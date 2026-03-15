import torchvision.models as models

def load_model(name="resnet50"):

    model_dict = {
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "mobilenet_v2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    }

    if name not in model_dict:
        raise ValueError(f"Model {name} not supported")

    model = model_dict[name]()
    model.eval()

    return model