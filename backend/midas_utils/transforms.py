import cv2
import numpy as np
import torch
from torchvision import transforms

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image

class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image.astype(np.float32) / 255.0
        image -= np.array(self.mean)
        image /= np.array(self.std)
        return image

class PrepareForNet(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img