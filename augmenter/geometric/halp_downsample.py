import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class HalpDownsample:
    def __init__(self, interpolation=None):
        self.interpolation = interpolation

    def __call__(self, images):
        img_shape = images[0].shape
        images[1] = cv2.resize(images[1], (img_shape[0] // 2, img_shape[1] // 2), interpolation=self.interpolation)
        images[2] = cv2.resize(images[2], (img_shape[0] // 2, img_shape[1] // 2), interpolation=self.interpolation)
        images[3] = cv2.resize(images[3], (img_shape[0] // 2, img_shape[1] // 2), interpolation=cv2.INTER_NEAREST)
        return images
        
        