import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class Flip:
    def __init__(self, mode='horizontal'):
        self.mode       = mode

    def __call__(self, images):
        try:
            h, w, _ = images[0].shape
        except:
            h, w = images[0].shape
        horizontal_list = ['horizontal', 'h']
        vertical_list   = ['vertical', 'v']
        if self.mode.lower() in horizontal_list:
            for image in images:
                image = cv2.flip(image, 1)
        elif self.mode.lower() in vertical_list:
            for image in images:
                image = cv2.flip(image, 0)
        return images


class RandomFlip:
    def __init__(self, mode='horizontal', prob=0.5):
        self.mode = mode
        self.prob = prob
        
    def __call__(self, images):
        self.aug        = Flip(mode=self.mode)
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            images = self.aug(images)
        return images