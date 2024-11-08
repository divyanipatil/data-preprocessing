import torchvision.transforms as transforms
import torch
import random


class ImageAugmenter:
    @staticmethod
    def horizontal_flip(image):
        """Flip image horizontally"""
        transform = transforms.RandomHorizontalFlip(p=1.0)
        return transform(image)

    @staticmethod
    def rotate(image):
        """Rotate image randomly between -30 and 30 degrees"""
        transform = transforms.RandomRotation(30)
        return transform(image)

    @staticmethod
    def color_jitter(image):
        """Randomly change brightness, contrast, and saturation"""
        transform = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        )
        return transform(image)
