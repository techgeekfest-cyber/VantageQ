"""Official Kuro Siwo U-Net baseline (configs/method/unet/unet.json, models/model_utilities.py)."""

from __future__ import annotations

import segmentation_models_pytorch as smp

from ml.kurosiwo import NUM_CLASSES
from ml.preprocessing import CHANNELS


def build_unet(encoder_weights: str | None = "imagenet") -> smp.Unet:
    """smp.Unet, ResNet-18 encoder, 6 input channels, 3 classes (as the official config).

    With ImageNet weights, smp adapts the first convolution from 3 to 6 input channels.
    """
    return smp.Unet(encoder_name="resnet18", encoder_weights=encoder_weights,
                    in_channels=len(CHANNELS), classes=NUM_CLASSES)
