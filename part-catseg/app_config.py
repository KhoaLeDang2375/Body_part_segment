# ============================================================
# app_config.py
# Static configuration for PartCATSeg Gradio Demo
# Using PascalPart116 (partcatseg_voc.pth) as sole checkpoint
# ============================================================

# Object classes from PascalPart116 dataset
# (these are the selectable options in the Gradio dropdown)
VOC_OBJ_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Part classes belonging to each object (for display in results table)
OBJ_TO_PARTS = {
    "aeroplane":    ["body", "stern", "wing", "tail", "engine", "wheel"],
    "bicycle":      ["wheel", "saddle", "handlebar", "chainwheel", "headlight"],
    "bird":         ["wing", "tail", "head", "eye", "beak", "torso", "neck", "leg", "foot"],
    "boat":         [],  # no part labels in dataset
    "bottle":       ["body", "cap"],
    "bus":          ["wheel", "headlight", "front", "side", "back", "roof", "mirror", "license plate", "door", "window"],
    "car":          ["wheel", "headlight", "front", "side", "back", "roof", "mirror", "license plate", "door", "window"],
    "cat":          ["tail", "head", "eye", "torso", "neck", "leg", "nose", "paw", "ear"],
    "chair":        [],  # no part labels
    "cow":          ["tail", "head", "eye", "torso", "neck", "leg", "ear", "muzzle", "horn"],
    "diningtable":  [],  # no part labels
    "dog":          ["tail", "head", "eye", "torso", "neck", "leg", "nose", "paw", "ear", "muzzle"],
    "horse":        ["tail", "head", "eye", "torso", "neck", "leg", "ear", "muzzle", "hoof"],
    "motorbike":    ["wheel", "saddle", "handlebar", "headlight"],
    "person":       ["head", "eye", "torso", "neck", "leg", "foot", "nose", "ear", "eyebrow",
                     "mouth", "hair", "lower arm", "upper arm", "hand"],
    "pottedplant":  ["pot", "plant"],
    "sheep":        ["tail", "head", "eye", "torso", "neck", "leg", "ear", "muzzle", "horn"],
    "sofa":         [],  # no part labels
    "train":        ["headlight", "head", "front", "side", "back", "roof", "coach"],
    "tvmonitor":    ["screen"],
}

# Model configuration
MODEL_CONFIG = {
    "config":  "configs/zero_shot/partcatseg_voc.yaml",
    "weights": "weights/partcatseg_voc.pth",
}

# Detectron2 dataset names used for metadata lookup
TRAIN_DATASET = "voc_obj_part_sem_seg_train"
TEST_DATASET  = "voc_obj_part_sem_seg_val_obj_condition"
