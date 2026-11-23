"""Shared model and image preprocessing."""

from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

ARCHITECTURE = "mobilenet_v3_small"
DOCUMENT_LABELS = ("ID_Card", "Passport")
QUALITY_LABELS = ("Clear", "Blur", "Glare", "Occluded")
IMAGE_SIZE = 224
MAX_IMAGE_SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def center_square(image):
    return transforms.functional.center_crop(image, min(image.size))


def spatial_preprocessing(size, crop_mode="full_frame"):
    operations = []
    if crop_mode == "center_square":
        operations.append(transforms.Lambda(center_square))
    operations.append(transforms.Resize((size, size)))
    return transforms.Compose(operations)


def preprocessing(size=IMAGE_SIZE, mean=MEAN, std=STD, crop_mode="full_frame"):
    return transforms.Compose([
        spatial_preprocessing(size, crop_mode),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_model(has_quality_head, weights=None):
    model = mobilenet_v3_small(weights=weights)
    features = model.classifier[-1].in_features
    outputs = len(DOCUMENT_LABELS)
    if has_quality_head:
        outputs += len(QUALITY_LABELS)
    model.classifier[-1] = nn.Linear(features, outputs)
    return model
