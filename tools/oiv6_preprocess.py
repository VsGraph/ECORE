"""
Preprocess Open Images V6 Visual Relationship annotations into the JSON format
used by this codebase (same structure as GQA_200).

Expected input files (download from https://storage.googleapis.com/openimages/web/download_v6.html):
  - annotations/oidv6-class-descriptions.csv       : all object class labels
  - annotations/relationships_oidv6.txt             : all relationship labels (one per line)
  - annotations/train-annotations-vrd.csv           : training VRD annotations
  - annotations/validation-annotations-vrd.csv      : validation VRD annotations
  - annotations/test-annotations-vrd.csv            : test VRD annotations (if available)
  - annotations/train-images-boxable-with-rotation.csv : image metadata (width/height) for train
  - annotations/validation-images-with-rotation.csv    : image metadata for val
  - annotations/test-images-with-rotation.csv          : image metadata for test

Images should be organised as:
  images/
    train/   (or flat directory, set --flat_images accordingly)
    validation/
    test/

Usage:
  python tools/oiv6_preprocess.py \\
      --oi_dir /path/to/openimages \\
      --output_dir /path/to/output \\
      --min_rel_per_image 1
"""

import os
import csv
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------------------------
# Standard OIv6 VRD relationship labels (30 categories)
# ---------------------------------------------------------------------------
OI_RELATION_NAMES = [
    '__background__',
    'at', 'holds', 'is', 'interacts_with', 'inside_of',
    'on', 'under', 'hits', 'hangs_from', 'wears',
    'is_inside', 'larger_than', 'smaller_than', 'narrower_than', 'wider_than',
    'taller_than', 'shorter_than', 'is_zipped', 'is_open', 'is_on',
    'plays', 'talks_on_phone', 'reads', 'using', 'kicks',
    'catches', 'throws', 'eats', 'drinks', 'is_for',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Convert OIv6 VRD annotations to JSON')
    parser.add_argument('--oi_dir', required=True,
                        help='Root directory of Open Images dataset')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for JSON files')
    parser.add_argument('--min_rel_per_image', type=int, default=1,
                        help='Minimum number of relationships per image')
    parser.add_argument('--flat_images', action='store_true',
                        help='All images are in a single flat directory')
    return parser.parse_args()


def load_class_descriptions(desc_file):
    """Returns {label_name: readable_name} from oidv6-class-descriptions.csv."""
    label_to_name = {}
    with open(desc_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_to_name[row[0]] = row[1]
    return label_to_name


def load_image_metadata(meta_file):
    """Returns {image_id: {'width': W, 'height': H, 'filename': rel_path}}."""
    meta = {}
    with open(meta_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageID']
            meta[image_id] = {
                'width': int(float(row['OriginalWidth'])),
                'height': int(float(row['OriginalHeight'])),
            }
    return meta


def load_vrd_annotations(vrd_file):
    """
    Returns list of dicts:
      {image_id, label1, bbox1, label2, bbox2, relationship}
    where bbox = [xmin, ymin, xmax, ymax] in absolute pixels (after multiplying by w/h).
    """
    rows = []
    with open(vrd_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_split_json(vrd_rows, img_meta, label_to_name,
                     obj_to_idx, rel_to_idx,
                     split, flat_images, img_base_dir, min_rel):
    """
    Converts raw VRD rows to our JSON structure.
    Returns dict with keys: filenames_all, img_info_all, gt_boxes_all,
                            gt_classes_all, relationships_all
    """
    # Group annotations by image
    img_rels = defaultdict(list)
    for row in vrd_rows:
        img_rels[row['ImageID']].append(row)

    filenames_all = []
    img_info_all = []
    gt_boxes_all = []
    gt_classes_all = []
    relationships_all = []

    skipped = 0
    for image_id, rels in tqdm(img_rels.items(), desc=f'Building {split}'):
        if image_id not in img_meta:
            skipped += 1
            continue

        W = img_meta[image_id]['width']
        H = img_meta[image_id]['height']

        # Collect all unique boxes and their labels
        box_registry = {}  # (xmin,ymin,xmax,ymax,label_idx) -> box_idx
        box_list = []      # list of [xmin, ymin, xmax, ymax]
        cls_list = []      # list of class indices

        def get_or_add_box(label_name, xmin_rel, xmax_rel, ymin_rel, ymax_rel):
            readable = label_to_name.get(label_name, label_name)
            obj_idx = obj_to_idx.get(readable)
            if obj_idx is None:
                return None
            xmin = int(float(xmin_rel) * W)
            ymin = int(float(ymin_rel) * H)
            xmax = int(float(xmax_rel) * W)
            ymax = int(float(ymax_rel) * H)
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
            if xmax <= xmin or ymax <= ymin:
                return None
            key = (xmin, ymin, xmax, ymax, obj_idx)
            if key not in box_registry:
                box_registry[key] = len(box_list)
                box_list.append([xmin, ymin, xmax, ymax])
                cls_list.append(obj_idx)
            return box_registry[key]

        rel_triples = []
        for row in rels:
            rel_name = row.get('RelationshipLabel', row.get('RelationShipLabel', ''))
            rel_idx = rel_to_idx.get(rel_name)
            if rel_idx is None:
                continue

            idx1 = get_or_add_box(
                row['LabelName1'],
                row['XMin1'], row['XMax1'], row['YMin1'], row['YMax1'])
            idx2 = get_or_add_box(
                row['LabelName2'],
                row['XMin2'], row['XMax2'], row['YMin2'], row['YMax2'])

            if idx1 is None or idx2 is None or idx1 == idx2:
                continue
            rel_triples.append([idx1, idx2, rel_idx])

        if len(rel_triples) < min_rel or len(box_list) == 0:
            skipped += 1
            continue

        # Determine image file path
        if flat_images:
            filename = f'{image_id}.jpg'
        else:
            filename = os.path.join(split, f'{image_id}.jpg')

        filenames_all.append(filename)
        img_info_all.append({'width': W, 'height': H, 'image_id': image_id})
        gt_boxes_all.append(box_list)
        gt_classes_all.append(cls_list)
        relationships_all.append(rel_triples)

    print(f'[{split}] total: {len(filenames_all)}, skipped: {skipped}')
    return {
        'filenames_all': filenames_all,
        'img_info_all': img_info_all,
        'gt_boxes_all': gt_boxes_all,
        'gt_classes_all': gt_classes_all,
        'relationships_all': relationships_all,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ann_dir = os.path.join(args.oi_dir, 'annotations')

    # ---- Load class descriptions ----
    desc_file = os.path.join(ann_dir, 'oidv6-class-descriptions.csv')
    label_to_name = load_class_descriptions(desc_file)

    # ---- Build object vocabulary from VRD annotations ----
    print('Scanning object classes used in VRD annotations...')
    all_obj_readable = set()
    for split_name, vrd_fname in [
        ('train', 'train-annotations-vrd.csv'),
        ('validation', 'validation-annotations-vrd.csv'),
    ]:
        vrd_file = os.path.join(ann_dir, vrd_fname)
        if not os.path.exists(vrd_file):
            continue
        for row in load_vrd_annotations(vrd_file):
            for key in ['LabelName1', 'LabelName2']:
                name = label_to_name.get(row[key], row[key])
                all_obj_readable.add(name)

    obj_classes = ['__background__'] + sorted(all_obj_readable)
    obj_to_idx = {c: i for i, c in enumerate(obj_classes)}

    # ---- Use standard OIv6 relation vocabulary ----
    rel_to_idx = {r: i for i, r in enumerate(OI_RELATION_NAMES)}

    # ---- Save dict file ----
    dict_data = {
        'ind_to_classes': obj_classes,
        'ind_to_predicates': OI_RELATION_NAMES,
    }
    dict_file = os.path.join(args.output_dir, 'OI_V6_dict.json')
    with open(dict_file, 'w') as f:
        json.dump(dict_data, f)
    print(f'Saved dict: {dict_file}  '
          f'(obj: {len(obj_classes)}, rel: {len(OI_RELATION_NAMES)})')

    # ---- Process each split ----
    split_configs = [
        ('train',      'train-annotations-vrd.csv',
                       'train-images-boxable-with-rotation.csv'),
        ('validation', 'validation-annotations-vrd.csv',
                       'validation-images-with-rotation.csv'),
        ('test',       'test-annotations-vrd.csv',
                       'test-images-with-rotation.csv'),
    ]

    for split_name, vrd_fname, meta_fname in split_configs:
        vrd_file = os.path.join(ann_dir, vrd_fname)
        meta_file = os.path.join(ann_dir, meta_fname)
        if not os.path.exists(vrd_file):
            print(f'Skip {split_name}: {vrd_file} not found')
            continue
        if not os.path.exists(meta_file):
            print(f'Skip {split_name}: {meta_file} not found')
            continue

        print(f'\nProcessing {split_name}...')
        vrd_rows = load_vrd_annotations(vrd_file)
        img_meta = load_image_metadata(meta_file)
        split_json = build_split_json(
            vrd_rows, img_meta, label_to_name,
            obj_to_idx, rel_to_idx,
            split=split_name,
            flat_images=args.flat_images,
            img_base_dir=os.path.join(args.oi_dir, 'images'),
            min_rel=args.min_rel_per_image,
        )
        out_key = 'Train' if split_name == 'train' else 'Test'
        out_file = os.path.join(args.output_dir, f'OI_V6_{out_key}.json')
        with open(out_file, 'w') as f:
            json.dump(split_json, f)
        print(f'Saved: {out_file}')

    print('\nDone. Files saved to:', args.output_dir)
    print('Object classes:', len(obj_classes))
    print('Relation classes:', len(OI_RELATION_NAMES))
    print('\nUpdate defaults.py:')
    print(f'  _C.MODEL.ROI_BOX_HEAD.OI_V6_NUM_CLASSES = {len(obj_classes)}')
    print(f'  _C.MODEL.ROI_RELATION_HEAD.OI_V6_NUM_CLASSES = {len(OI_RELATION_NAMES)}')


if __name__ == '__main__':
    main()
