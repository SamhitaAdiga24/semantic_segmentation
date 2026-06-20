import segmentation_models_pytorch as smp

def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name="tu-convnext_tiny",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5
    )
    return model