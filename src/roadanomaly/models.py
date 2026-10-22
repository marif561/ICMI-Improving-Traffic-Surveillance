"""ImageNet-pre-trained torchvision backbones with a new classification head."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

BACKBONES = {
    "inception_v3": ("inception_v3", "Inception_V3_Weights", 299),
    "resnet50": ("resnet50", "ResNet50_Weights", 224),
    "vgg16": ("vgg16", "VGG16_Weights", 224),
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights", 224),
    "densenet121": ("densenet121", "DenseNet121_Weights", 224),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", 224),
}


def default_image_size(name: str) -> int:
    return BACKBONES[name][2]


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.5,
                freeze_backbone: bool = False) -> nn.Module:
    fn, weights_cls, _ = BACKBONES[name]
    weights = getattr(models, weights_cls).DEFAULT if pretrained else None
    kwargs = {"aux_logits": True, "init_weights": not pretrained} if name == "inception_v3" else {}
    m = getattr(models, fn)(weights=weights, **kwargs)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False

    def head(in_f):
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    if name in ("inception_v3", "resnet50"):
        m.fc = head(m.fc.in_features)
        if name == "inception_v3":
            m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
    elif name == "densenet121":
        m.classifier = head(m.classifier.in_features)
    elif name == "vgg16":
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:  # mobilenet_v2, efficientnet_b0
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def main_logits(output):
    """InceptionV3 returns (logits, aux_logits) in training mode."""
    return output.logits if hasattr(output, "logits") else output
