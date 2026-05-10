import torch
import segmentation_models_pytorch as smp

def load_model(model_path, device):
    """
    Загружает обученную модель Unet++ EfficientNet-B4
    """

    # Создаем архитектуру с предобученными весами
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",  # Используйте предобученные веса
        in_channels=2,
        classes=1
    )

    # Загружаем веса модели
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Диагностика
    total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters:", total_params)
    return model