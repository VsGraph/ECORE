
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions. Recommended: cuda-10.1 & pytorch-1.6.

## Dataset
See  [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


**Tasks:**

```bash
# PredCls
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
# SGCls
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
# SGDet
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
## Training
### Example Command 

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 \
  tools/relation_train_net.py \
  --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" \
  GLOBAL_SETTING.DATASET_CHOICE 'VG' \
  GLOBAL_SETTING.RELATION_PREDICTOR 'MotifsLike_GCL' \
  GLOBAL_SETTING.BASIC_ENCODER 'Motifs' \
  GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' \
  GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 8 \
  DTYPE "float16" SOLVER.MAX_ITER 30000 \
  GLOVE_DIR /path/to/glove \
  OUTPUT_DIR /path/to/output
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 \
  tools/relation_test_net.py \
  --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" \
  GLOBAL_SETTING.DATASET_CHOICE 'VG' \
  GLOBAL_SETTING.RELATION_PREDICTOR 'MotifsLike_GCL' \
  GLOBAL_SETTING.BASIC_ENCODER 'Motifs' \
  GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' \
  GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  TEST.IMS_PER_BATCH 8 DTYPE "float16" \
  GLOVE_DIR /path/to/glove \
  OUTPUT_DIR /path/to/output
```
