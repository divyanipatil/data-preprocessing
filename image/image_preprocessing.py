import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from io import BytesIO
import base64


class ImagePreprocessor:
    @staticmethod
    def normalize_standard(image):
        """Standard normalization using ImageNet statistics"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.ToPILImage()
        ])
        return transform(image)

    @staticmethod
    def resize_224(image):
        """Resize to 224x224"""
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        return transform(image)

    @staticmethod
    def grayscale(image):
        transform = transforms.Compose([
            transforms.Grayscale(3)  # 3 channels for compatibility
        ])
        return transform(image)

    @staticmethod
    def to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
