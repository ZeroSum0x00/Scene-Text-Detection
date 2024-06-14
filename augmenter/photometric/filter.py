import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class ErodeDilate:
    def __init__(self, kernel_size=(1, 1)):
        self.kernel = np.ones(kernel_size, np.uint8)
        
    def __call__(self, images):
        if np.random.rand() <= 0.5:
            images[0] = cv2.erode(images[0], self.kernel, iterations=1)
        else:
            images[0] = cv2.dilate(images[0], self.kernel, iterations=1)
        return images


class RandomErodeDilate:
    def __init__(self, kernel_size=(1, 1), prob=0.5):
        self.kernel_size = kernel_size
        self.prob        = prob

    def __call__(self, images):
        self.aug = ErodeDilate(self.kernel_size)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            images = self.aug(images)
        return images