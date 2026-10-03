import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 4


def create_model(mode):
    """
    Membuat ResNet-18 berdasarkan mode training.

    mode:
        feature -> ImageNet, hanya fc
        partial -> ImageNet, layer4 + fc
        scratch -> random, semua layer
    """

    if mode == "feature":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Bekukan seluruh backbone
        for param in model.parameters():
            param.requires_grad = False

        # Ganti classifier
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

        # fc tetap trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "partial":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Bekukan seluruh model terlebih dahulu
        for param in model.parameters():
            param.requires_grad = False

        # Buka layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Ganti classifier
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

        # fc trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "scratch":
        # Tidak menggunakan pretrained weights
        model = resnet18(weights=None)

        # Ganti classifier
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

        # Semua parameter trainable
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(
            "Mode harus 'feature', 'partial', atau 'scratch'"
        )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


if __name__ == "__main__":

    for mode in ["feature", "partial", "scratch"]:

        model = create_model(mode)

        total, trainable = count_parameters(model)

        print("=" * 50)
        print(f"MODE: {mode.upper()}")
        print(f"Total parameters     : {total:,}")
        print(f"Trainable parameters : {trainable:,}")
        print("=" * 50)