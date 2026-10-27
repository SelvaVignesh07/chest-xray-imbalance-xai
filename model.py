"""
model.py
========
Phase 2: Model Architecture (Transfer Learning)

We use a pretrained ResNet (18 or 50) as a frozen/fine-tunable feature
extractor and replace the final fully-connected layer to output 2 logits
(NORMAL vs. PNEUMONIA) instead of ImageNet's 1000 classes.

Why transfer learning at all: chest X-ray datasets (thousands of images) are
tiny compared to ImageNet (1.2M images). Training a CNN from random
initialization on a dataset this small would overfit badly. A network
pretrained on ImageNet has already learned general-purpose low/mid-level
visual features (edges, textures, blob detectors) that transfer well to
X-ray images despite the domain being visually very different -- this is a
well established practice in the medical imaging literature, including the
paper this project reproduces.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_device():
    """
    Auto-select the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def build_model(architecture="resnet18", num_classes=2, freeze_backbone=False):
    """
    Build a ResNet transfer-learning model with a custom classification head.

    Args:
        architecture (str): 'resnet18' or 'resnet50'.
        num_classes (int): number of output classes (2 for Normal/Pneumonia).
        freeze_backbone (bool): if True, freezes all convolutional layers and
            only trains the new final FC head (fast, less prone to
            overfitting on very small datasets, but lower ceiling on
            accuracy). If False, the entire network is fine-tuned end-to-end
            (typically better performance when you have a few thousand
            images, which chest X-ray datasets usually provide).

    Returns:
        torch.nn.Module: the modified model.
    """
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT  # best available ImageNet weights
        model = models.resnet18(weights=weights)
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("architecture must be 'resnet18' or 'resnet50'")

    if freeze_backbone:
        # Disable gradient computation for all pretrained convolutional
        # weights -- they act as a fixed feature extractor. Only the new
        # head (added below) will be trained.
        for param in model.parameters():
            param.requires_grad = False

    # ResNet's final layer is `model.fc`, a single Linear layer mapping
    # (512 for resnet18 / 2048 for resnet50) -> 1000 ImageNet classes.
    # We replace it with a fresh Linear layer -> num_classes. New layers
    # are created with requires_grad=True by default, so even in the
    # freeze_backbone=True case this head is trainable.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Quick sanity check
    device = get_device()
    model = build_model("resnet18", num_classes=2).to(device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("Output shape (expect [4, 2]):", output.shape)
