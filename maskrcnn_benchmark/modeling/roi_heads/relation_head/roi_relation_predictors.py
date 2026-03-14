# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from SHA_GCL_extra.kl_divergence import KL_divergence
from .model_Hybrid_Attention import SHA_Context, SHA_Encoder, AFEM
from .model_Cross_Attention import CA_Context
from .utils_relation import layer_init
from .utils_motifs import obj_edge_vectors, encode_box_info
from maskrcnn_benchmark.data import get_dataset_statistics

from SHA_GCL_extra.utils_funcion import FrequencyBias_GCL
from SHA_GCL_extra.extra_function_utils import generate_num_stage_vector, generate_sample_rate_vector, \
    generate_current_sequence_for_bias, get_current_predicate_idx
from SHA_GCL_extra.group_chosen_function import get_group_splits
import random





@registry.ROI_RELATION_PREDICTOR.register("MotifsLikePredictor")
class MotifsLikePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLikePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _, tri_rep = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        add_losses = {}

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OI_V6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OI_V6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OI_V6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifsLike_GCL")
class MotifsLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLike_GCL, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OI_V6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OI_V6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OI_V6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()

        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.afem = AFEM(
            union_dim=self.pooling_dim,
            sem_dim=self.embed_dim,
            hidden_dim=self.pooling_dim,
            n_head=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD,
            d_k=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM,
            d_v=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM,
            d_inner=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM,
            dropout=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE,
        )

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = obj_edge_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed_t = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed_t = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed_t.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed_t.weight.copy_(rel_embed_vecs, non_blocking=True)
        self.rel_compress_init = nn.Linear(self.pooling_dim, self.num_rel_cls)
        layer_init(self.rel_compress_init, xavier=True)
        self.lin_visual_triplet = nn.Linear(self.hidden_dim * 2 + self.pooling_dim, self.hidden_dim)
        self.lin_semantic_triplet = nn.Linear(self.embed_dim * 3, self.hidden_dim)
        self.cat_encoder = SHA_Encoder(config, n_layers=2)
        self.cat_proj = nn.Linear(self.hidden_dim, self.pooling_dim)
        layer_init(self.lin_visual_triplet, xavier=True)
        layer_init(self.lin_semantic_triplet, xavier=True)
        layer_init(self.cat_proj, xavier=True)

        self.aum_tau1   = config.MODEL.ROI_RELATION_HEAD.AUM.TAU1
        self.aum_tau2   = config.MODEL.ROI_RELATION_HEAD.AUM.TAU2
        self.aum_lambda = config.MODEL.ROI_RELATION_HEAD.AUM.LAMBDA
        self.register_buffer('aum_iter', torch.zeros(1, dtype=torch.long))
        self.log_var_all = self.generate_aum_log_var_branches(self.num_groups)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        sbj_embed1_reps = []
        obj_embed2_reps = []
        sbj_glove_reps = []
        obj_glove_reps = []
        sbj_ctx_reps = []
        obj_ctx_reps = []
        for pair_idx, head_rep_i, tail_rep_i, obj_pred_i in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep_i[pair_idx[:, 0]], tail_rep_i[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred_i[pair_idx[:, 0]], obj_pred_i[pair_idx[:, 1]]), dim=1))
            sbj_embed1_reps.append(self.obj_embed1(obj_pred_i[pair_idx[:, 0]]))
            obj_embed2_reps.append(self.obj_embed2(obj_pred_i[pair_idx[:, 1]]))
            sbj_glove_reps.append(self.obj_embed_t(obj_pred_i[pair_idx[:, 0]]))
            obj_glove_reps.append(self.obj_embed_t(obj_pred_i[pair_idx[:, 1]]))
            sbj_ctx_reps.append(head_rep_i[pair_idx[:, 0]])
            obj_ctx_reps.append(tail_rep_i[pair_idx[:, 1]])
        prod_rep  = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        S_i       = cat(sbj_embed1_reps, dim=0)
        S_j       = cat(obj_embed2_reps, dim=0)
        sbj_glove = cat(sbj_glove_reps, dim=0)
        obj_glove = cat(obj_glove_reps, dim=0)
        sbj_ctx   = cat(sbj_ctx_reps, dim=0)
        obj_ctx   = cat(obj_ctx_reps, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_feat = self.up_dim(union_features)
        else:
            union_feat = union_features

        sem_pair = S_i + S_j
        enhanced_union = self.afem(union_feat, sem_pair, num_rels)

        init_rel_dists = self.rel_compress_init(prod_rep)
        rel_prob = F.softmax(init_rel_dists, dim=1)
        rel_mask = 1 - (rel_prob < 1e-3).long()
        rel_sem = (rel_prob * rel_mask) @ self.rel_embed_t.weight

        vis_triplet = self.lin_visual_triplet(
            torch.cat([sbj_ctx, union_feat, obj_ctx], dim=-1))
        sem_triplet = self.lin_semantic_triplet(
            torch.cat([sbj_glove, rel_sem, obj_glove], dim=-1))

        triplet_ctx, _ = self.cat_encoder(vis_triplet, sem_triplet, num_rels)

        prod_rep = prod_rep + self.cat_proj(triplet_ctx)

        if self.use_vision:
            prod_rep = prod_rep * enhanced_union

        add_losses = {}
        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            self.aum_iter += 1
            k_iter = self.aum_iter.item()
            if k_iter <= self.aum_tau1:
                gamma = 0.0
            elif k_iter <= self.aum_tau2:
                gamma = self.aum_lambda * (k_iter - self.aum_tau1) / (self.aum_tau2 - self.aum_tau1)
            else:
                gamma = self.aum_lambda

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                jdx = i
                rel_classier_now = self.rel_classifer_all[jdx]
                group_output_now = rel_classier_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                actual_label_now = self.pre_group_matrix[jdx][group_label]

                l_ce = self.CE_loss(group_output_now, actual_label_now)
                if gamma > 0.0:
                    y_onehot = torch.zeros_like(group_output_now).scatter_(
                        1, actual_label_now.unsqueeze(1), 1.0)
                    s = self.log_var_all[jdx](group_input)
                    sq_err = (group_output_now - y_onehot) ** 2
                    l_unc = (sq_err / s.exp() + 0.5 * s).mean()
                    add_losses['%d_CE_loss' % (jdx + 1)] = (1.0 - gamma) * l_ce + gamma * l_unc
                else:
                    add_losses['%d_CE_loss' % (jdx + 1)] = l_ce

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            return None, None, add_losses
        else:
            rel_classier_test = self.rel_classifer_all[-1]
            rel_dists = rel_classier_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all

    def generate_aum_log_var_branches(self, num_cls):
        """
        Generate one log-variance FC branch per group classifier.
        Each branch has the same output dimension as the corresponding
        rel_classifer, so the class-specific log-variance s \u2208 R^{R_g}
        aligns with the group logits f(z) \u2208 R^{R_g}.

        Weights/biases are zero-initialized: s = 0 at the start of training
        \u2192 exp(s) = 1, so the uncertainty term has no effect during Stage 1.
        """
        def _make_branch(out_dim):
            fc = nn.Linear(self.pooling_dim, out_dim)
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)
            return fc

        self.log_var_1 = _make_branch(self.max_group_element_number_list[0] + 1)
        self.log_var_2 = _make_branch(self.max_group_element_number_list[1] + 1)
        self.log_var_3 = _make_branch(self.max_group_element_number_list[2] + 1)
        self.log_var_4 = _make_branch(self.max_group_element_number_list[3] + 1)
        if num_cls == 4:
            log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3, self.log_var_4]
        elif num_cls < 4:
            exit('wrong num in log_var_all')
        else:
            self.log_var_5 = _make_branch(self.max_group_element_number_list[4] + 1)
            if num_cls == 5:
                log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3,
                               self.log_var_4, self.log_var_5]
            else:
                self.log_var_6 = _make_branch(self.max_group_element_number_list[5] + 1)
                if num_cls == 6:
                    log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3,
                                   self.log_var_4, self.log_var_5, self.log_var_6]
                else:
                    self.log_var_7 = _make_branch(self.max_group_element_number_list[6] + 1)
                    log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3,
                                   self.log_var_4, self.log_var_5, self.log_var_6, self.log_var_7]
                    if num_cls > 7:
                        exit('wrong num in log_var_all')
        return log_var_all


    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("VCTree_GCL")
class VCTree_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTree_GCL, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OI_V6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OI_V6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OI_V6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        self.num_groups = len(self.max_elemnt_list)
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()

        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.afem = AFEM(
            union_dim=self.pooling_dim,
            sem_dim=self.embed_dim,
            hidden_dim=self.pooling_dim,
            n_head=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD,
            d_k=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM,
            d_v=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM,
            d_inner=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM,
            dropout=config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE,
        )

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = obj_edge_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed_t = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed_t = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed_t.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed_t.weight.copy_(rel_embed_vecs, non_blocking=True)
        self.rel_compress_init = nn.Linear(self.pooling_dim, self.num_rel_cls)
        layer_init(self.rel_compress_init, xavier=True)
        self.lin_visual_triplet = nn.Linear(self.hidden_dim * 2 + self.pooling_dim, self.hidden_dim)
        self.lin_semantic_triplet = nn.Linear(self.embed_dim * 3, self.hidden_dim)
        self.cat_encoder = SHA_Encoder(config, n_layers=2)
        self.cat_proj = nn.Linear(self.hidden_dim, self.pooling_dim)
        layer_init(self.lin_visual_triplet, xavier=True)
        layer_init(self.lin_semantic_triplet, xavier=True)
        layer_init(self.cat_proj, xavier=True)

        self.aum_tau1   = config.MODEL.ROI_RELATION_HEAD.AUM.TAU1
        self.aum_tau2   = config.MODEL.ROI_RELATION_HEAD.AUM.TAU2
        self.aum_lambda = config.MODEL.ROI_RELATION_HEAD.AUM.LAMBDA
        self.register_buffer('aum_iter', torch.zeros(1, dtype=torch.long))
        self.log_var_all = self.generate_aum_log_var_branches(self.num_groups)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps      = []
        pair_preds     = []
        sbj_embed1_reps = []
        obj_embed2_reps = []
        sbj_glove_reps  = []
        obj_glove_reps  = []
        sbj_ctx_reps    = []
        obj_ctx_reps    = []
        for pair_idx, head_rep_i, tail_rep_i, obj_pred_i in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep_i[pair_idx[:, 0]], tail_rep_i[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred_i[pair_idx[:, 0]], obj_pred_i[pair_idx[:, 1]]), dim=1))
            sbj_embed1_reps.append(self.obj_embed1(obj_pred_i[pair_idx[:, 0]]))
            obj_embed2_reps.append(self.obj_embed2(obj_pred_i[pair_idx[:, 1]]))
            sbj_glove_reps.append(self.obj_embed_t(obj_pred_i[pair_idx[:, 0]]))
            obj_glove_reps.append(self.obj_embed_t(obj_pred_i[pair_idx[:, 1]]))
            sbj_ctx_reps.append(head_rep_i[pair_idx[:, 0]])
            obj_ctx_reps.append(tail_rep_i[pair_idx[:, 1]])
        prod_rep  = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        S_i       = cat(sbj_embed1_reps, dim=0)
        S_j       = cat(obj_embed2_reps, dim=0)
        sbj_glove = cat(sbj_glove_reps, dim=0)
        obj_glove = cat(obj_glove_reps, dim=0)
        sbj_ctx   = cat(sbj_ctx_reps, dim=0)
        obj_ctx   = cat(obj_ctx_reps, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_feat = self.up_dim(union_features)
        else:
            union_feat = union_features

        sem_pair       = S_i + S_j
        enhanced_union = self.afem(union_feat, sem_pair, num_rels)

        init_rel_dists = self.rel_compress_init(prod_rep)
        rel_prob = F.softmax(init_rel_dists, dim=1)
        rel_mask = 1 - (rel_prob < 1e-3).long()
        rel_sem  = (rel_prob * rel_mask) @ self.rel_embed_t.weight

        vis_triplet = self.lin_visual_triplet(
            torch.cat([sbj_ctx, union_feat, obj_ctx], dim=-1))
        sem_triplet = self.lin_semantic_triplet(
            torch.cat([sbj_glove, rel_sem, obj_glove], dim=-1))

        triplet_ctx, _ = self.cat_encoder(vis_triplet, sem_triplet, num_rels)

        prod_rep = prod_rep + self.cat_proj(triplet_ctx)

        if self.use_vision:
            prod_rep = prod_rep * enhanced_union

        add_losses = {}
        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label  = max(rel_labels)

            self.aum_iter += 1
            k_iter = self.aum_iter.item()
            if k_iter <= self.aum_tau1:
                gamma = 0.0
            elif k_iter <= self.aum_tau2:
                gamma = self.aum_lambda * (k_iter - self.aum_tau1) / (self.aum_tau2 - self.aum_tau1)
            else:
                gamma = self.aum_lambda

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                jdx = i
                rel_classier_now = self.rel_classifer_all[jdx]
                group_output_now = rel_classier_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                actual_label_now = self.pre_group_matrix[jdx][group_label]

                l_ce = self.CE_loss(group_output_now, actual_label_now)
                if gamma > 0.0:
                    y_onehot = torch.zeros_like(group_output_now).scatter_(
                        1, actual_label_now.unsqueeze(1), 1.0)
                    s = self.log_var_all[jdx](group_input)
                    sq_err = (group_output_now - y_onehot) ** 2
                    l_unc = (sq_err / s.exp() + 0.5 * s).mean()
                    add_losses['%d_CE_loss' % (jdx + 1)] = (1.0 - gamma) * l_ce + gamma * l_unc
                else:
                    add_losses['%d_CE_loss' % (jdx + 1)] = l_ce

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        jbef = i - 1
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            return None, None, add_losses
        else:
            rel_classier_test = self.rel_classifer_all[-1]
            rel_dists = rel_classier_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

    def generate_aum_log_var_branches(self, num_cls):
        def _make_branch(out_dim):
            fc = nn.Linear(self.pooling_dim, out_dim)
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)
            return fc

        self.log_var_1 = _make_branch(self.max_group_element_number_list[0] + 1)
        self.log_var_2 = _make_branch(self.max_group_element_number_list[1] + 1)
        self.log_var_3 = _make_branch(self.max_group_element_number_list[2] + 1)
        self.log_var_4 = _make_branch(self.max_group_element_number_list[3] + 1)
        if num_cls == 4:
            log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3, self.log_var_4]
        elif num_cls < 4:
            exit('wrong num in log_var_all')
        else:
            self.log_var_5 = _make_branch(self.max_group_element_number_list[4] + 1)
            if num_cls == 5:
                log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3,
                               self.log_var_4, self.log_var_5]
            else:
                self.log_var_6 = _make_branch(self.max_group_element_number_list[5] + 1)
                if num_cls == 6:
                    log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3,
                                   self.log_var_4, self.log_var_5, self.log_var_6]
                else:
                    self.log_var_7 = _make_branch(self.max_group_element_number_list[6] + 1)
                    log_var_all = [self.log_var_1, self.log_var_2, self.log_var_3,
                                   self.log_var_4, self.log_var_5, self.log_var_6, self.log_var_7]
                    if num_cls > 7:
                        exit('wrong num in log_var_all')
        return log_var_all

@registry.ROI_RELATION_PREDICTOR.register("TransLikePredictor")
class TransLikePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLikePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OI_V6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OI_V6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OI_V6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        add_losses = {}

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    import time
    result_str = '---'*20
    result_str += ('\n\nthe dataset we use is [ %s ]' % cfg.GLOBAL_SETTING.DATASET_CHOICE)
    if cfg.GLOBAL_SETTING.USE_BIAS:
        result_str += ('\nwe use [ bias ]!')
    else:
        result_str += ('\nwe do [ not ] use bias!')
    result_str += ('\nthe model we use is [ %s ]' % cfg.GLOBAL_SETTING.RELATION_PREDICTOR)
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ predcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ sgcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == False:
        result_str += ('\ntraining mode is [ sgdet ]')
    else:
        exit('wrong training mode!')
    result_str += ('\nlearning rate is [ %.5f ]' % cfg.SOLVER.BASE_LR)
    result_str += ('\nthe knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    assert cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE in ['None', 'KL_logit_Neighbor', 'KL_logit_None',
                                               'KL_logit_TopDown', 'KL_logit_BottomUp', 'KL_logit_BiDirection']
    if cfg.GLOBAL_SETTING.RELATION_PREDICTOR in ['TransLike_GCL', 'TransLikePredictor']:
        result_str += ('\nrel labels=0 is use [ %s ] to process' % cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE)
        assert cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE in ['rand_insert', 'rand_choose', 'all_include']
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Self-Attention', 'Cross-Attention', 'Hybrid-Attention']
        result_str += ('\n-----Transformer layer is [ %d ] in obj and [ %d ] in rel' %
                       (cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER,
                        cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER))
        result_str += ('\n-----Transformer mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)
    if cfg.GLOBAL_SETTING.RELATION_PREDICTOR in ['MotifsLike_GCL', 'MotifsLikePredictor']:
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Motifs', 'VTransE']
        result_str += ('\n-----Model mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)

    num_of_group_element_list, predicate_stage_count = get_group_splits(cfg.GLOBAL_SETTING.DATASET_CHOICE, cfg.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE)
    max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
    incre_idx_list, max_elemnt_list, group_matrix, kd_matrix = get_current_predicate_idx(
        num_of_group_element_list, cfg.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_PENALTY, cfg.GLOBAL_SETTING.DATASET_CHOICE)
    result_str += ('\n   the number of elements in each group is {}'.format(incre_idx_list))
    result_str += ('\n   incremental stage list is {}'.format(num_of_group_element_list))
    result_str += ('\n   the length of each line in group is {}'.format(predicate_stage_count))
    result_str += ('\n   the max number of elements in each group is {}'.format(max_group_element_number_list))
    result_str += ('\n   the knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    result_str += ('\n   the penalty for whole distillation loss is [ %.2f ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT)
    with open(os.path.join(cfg.OUTPUT_DIR, 'control_info.txt'), 'w') as outfile:
        outfile.write(result_str)
    result_str += '\n\n'
    result_str += '---'*20
    print(result_str)
    time.sleep(2)
    func = registry.ROI_RELATION_PREDICTOR[cfg.GLOBAL_SETTING.RELATION_PREDICTOR]
    return func(cfg, in_channels)
