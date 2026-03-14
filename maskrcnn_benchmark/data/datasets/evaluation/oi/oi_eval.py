import logging
import os
import torch
import numpy as np
import json

from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import (
    SGRecall, SGNoGraphConstraintRecall, SGPairAccuracy,
    SGMeanRecall, SGNGMeanRecall,
)


def do_oi_evaluation(cfg, dataset, predictions, output_folder, logger, iou_types):
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.OI_V6_NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES

    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        predictions[image_id] = prediction.resize((image_width, image_height))
        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)

    _save_output(output_folder, groundtruths, predictions, dataset)

    result_str = '\n' + '=' * 100 + '\n'

    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}

        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        eval_mean_recall = SGMeanRecall(
            result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        eval_ng_mean_recall = SGNGMeanRecall(
            result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        global_container = {
            'result_dict': result_dict,
            'mode': mode,
            'multiple_preds': multiple_preds,
            'num_rel_category': num_rel_category,
            'iou_thres': iou_thres,
            'attribute_on': attribute_on,
            'num_attributes': num_attributes,
        }

        for groundtruth, prediction in zip(groundtruths, predictions):
            _evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)

        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'
        logger.info(result_str)

        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return float(np.mean(result_dict[mode + cfg.GLOBAL_SETTING.CHOOSE_BEST_MODEL_BY_METRIC][100]))
    else:
        return -1


def _save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save(
            {'groundtruths': groundtruths, 'predictions': predictions},
            os.path.join(output_folder, "eval_results.pytorch"))
        visual_info = []
        for image_id, (gt, pred) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            gt_info = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]
                for b, l in zip(gt.bbox.tolist(), gt.get_field('labels').tolist())
            ]
            pred_info = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]
                for b, l in zip(pred.bbox.tolist(), pred.get_field('pred_labels').tolist())
            ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': gt_info,
                'prediction': pred_info,
            })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def _evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    mode = global_container['mode']
    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()
    local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()

    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])
    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode in ('sgdet', 'phrdet'):
        pass
    else:
        raise ValueError('invalid mode')

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
