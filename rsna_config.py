import torchvision.models as models


SEED = 42
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10
TARGET_COLS  = [
    "bowel_healthy", "bowel_injury",
    "extravasation_healthy", "extravasation_injury",
    "kidney_healthy", "kidney_low", "kidney_high",
    "liver_healthy", "liver_low", "liver_high",
    "spleen_healthy", "spleen_low", "spleen_high",
    "any_injury"
]
loss_weights = {
    "bowel" : {"bowel_healthy" : 1, "bowel_injury" : 2},
    "extra" : {"extravasation_healthy" : 1, "extravasation_injury" : 6},
    "kidney" : {"kidney_healthy" : 1, "kidney_low" : 2, "kidney_high" : 4},
    "liver" : {"liver_healthy" : 1, "liver_low" : 2, "liver_high" : 4},
    "spleen" : {"spleen_healthy" : 1, "spleen_low" : 2, "spleen_high" : 4},
    "any_injury" : {"any_injury" : 6}
}

# Options for all models

HEAD_HIDDEN_SIZE = 32

#2D model options
BACKBONE = "efficientnet_v2_s"

#2D model parameters
resnet_backbones = {
"resnet18": {"model": models.resnet18, "weights": models.ResNet18_Weights.DEFAULT},
"resnet34": {"model": models.resnet34, "weights": models.ResNet34_Weights.DEFAULT},
"resnet50": {"model": models.resnet50, "weights": models.ResNet50_Weights.DEFAULT},
"resnet101": {"model": models.resnet101, "weights": models.ResNet101_Weights.DEFAULT},
"resnet152": {"model": models.resnet152, "weights": models.ResNet152_Weights.DEFAULT}
}

efficientnet_backbones = {
"efficientnet_v2_s": {"model": models.efficientnet_v2_s, "weights": models.EfficientNet_V2_S_Weights.DEFAULT},
"efficientnet_v2_m": {"model": models.efficientnet_v2_m, "weights": models.EfficientNet_V2_M_Weights.DEFAULT},
"efficientnet_v2_l": {"model": models.efficientnet_v2_l, "weights": models.EfficientNet_V2_L_Weights.DEFAULT}
}

    
backbone_dict = {**resnet_backbones, **efficientnet_backbones}

tasks=["bowel", "extra", "liver", "kidney", "spleen", "any_injury"]

#3D model options
RNN_HIDDEN_SIZE = 32s

DEBUGGIN = False