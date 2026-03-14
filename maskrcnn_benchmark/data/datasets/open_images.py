import os
import torch
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import random

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


class OIDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, dict_file, train_file, test_file,
                 transforms=None, filter_empty_rels=True, num_im=-1,
                 num_val_im=1813, filter_duplicate_rels=True,
                 filter_non_overlap=True, flip_aug=False,
                 custom_eval=False, custom_path=''):
        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.train_file = train_file
        self.test_file = test_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.custom_eval = custom_eval

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        if self.split == 'train':
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = \
                load_graphs(self.train_file, self.split)
        else:
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = \
                load_graphs(self.test_file, self.split)

        if custom_eval:
            self.get_custom_imgs(custom_path)

    def __getitem__(self, index):
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = BoxList(torch.zeros((1, 4)), img.size, mode='xyxy')
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index

        img = Image.open(os.path.join(self.img_dir, self.filenames[index])).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('='*20, ' size mismatch at index ', str(index), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, index

    def get_img_info(self, index):
        if self.custom_eval:
            return self.img_info[index]
        return self.img_info[index]

    def get_statistics(self):
        fg_matrix, bg_matrix = get_OI_statistics(
            img_dir=self.img_dir,
            train_file=self.train_file,
            dict_file=self.dict_file,
            must_overlap=True,
        )
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_groundtruth(self, index, flip_img=False, evaluation=False):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        box = self.gt_boxes[index]
        box = torch.from_numpy(box).reshape(-1, 4)

        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax

        target = BoxList(box, (w, h), 'xyxy')
        tgt_labels = torch.from_numpy(self.gt_classes[index])
        target.add_field("labels", tgt_labels.long())

        relation = self.relationships[index].copy()

        if self.filter_duplicate_rels:
            assert self.split == 'train'
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


def get_OI_statistics(img_dir, train_file, dict_file, must_overlap=True):
    train_data = OIDataset(
        split='train', img_dir=img_dir, train_file=train_file,
        dict_file=dict_file, test_file=None,
        filter_duplicate_rels=False,
    )
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    from tqdm import tqdm
    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    n_cands = boxes.shape[0]
    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)
    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))
        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(
        boxes1.reshape([num_box1, 1, -1])[:, :, :2],
        boxes2.reshape([1, num_box2, -1])[:, :, :2])
    rb = np.minimum(
        boxes1.reshape([num_box1, 1, -1])[:, :, 2:],
        boxes2.reshape([1, num_box2, -1])[:, :, 2:])
    wh = (rb - lt + to_move).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter


def load_info(dict_file):
    info = json.load(open(dict_file, 'r'))
    ind_to_classes = info['ind_to_classes']
    ind_to_predicates = info['ind_to_predicates']
    return ind_to_classes, ind_to_predicates


def load_graphs(data_json_file, split):
    data_info_all = json.load(open(data_json_file, 'r'))
    filenames = data_info_all['filenames_all']
    img_info = data_info_all['img_info_all']
    gt_boxes = data_info_all['gt_boxes_all']
    gt_classes = data_info_all['gt_classes_all']
    relationships = data_info_all['relationships_all']

    output_filenames = []
    output_img_info = []
    output_boxes = []
    output_classes = []
    output_relationships = []

    for filename, imginfo, gt_b, gt_c, gt_r in zip(
            filenames, img_info, gt_boxes, gt_classes, relationships):
        if len(gt_r) > 0 and len(gt_b) > 0:
            output_filenames.append(filename)
            output_img_info.append(imginfo)
            output_boxes.append(np.array(gt_b))
            output_classes.append(np.array(gt_c))
            output_relationships.append(np.array(gt_r))

    if split == 'val':
        output_filenames = output_filenames[:1813]
        output_img_info = output_img_info[:1813]
        output_boxes = output_boxes[:1813]
        output_classes = output_classes[:1813]
        output_relationships = output_relationships[:1813]

    return output_filenames, output_img_info, output_boxes, output_classes, output_relationships
