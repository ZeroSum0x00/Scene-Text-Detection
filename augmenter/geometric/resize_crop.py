import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class ResizeCrop:
    def __init__(self, target_size=(768, 768, 3), interpolation=cv2.INTER_CUBIC):
        self.target_size   = target_size
        self.interpolation = interpolation
    
    def resized_crop(self, image, i, j, w, h, interpolation=None):
        xmin, ymin, xmax, ymax = np.round(i), np.round(j), np.round(i + w), np.round(j + h)
        try:
            check_point1 = image[ymin, xmin, ...]
            check_point2 = image[ymax-1, xmax-1, ...]
        except IndexError:        
            image = cv2.copyMakeBorder(image, 
                                       -min(0, ymin), max(ymax - image.shape[0], 0),
                                       -min(0, xmin), max(xmax - image.shape[1], 0), 
                                       cv2.BORDER_CONSTANT, 
                                       value=[0, 0, 0])
            xmax += -min(0, xmin)
            xmin += -min(0, xmin)
            ymax += -min(0, ymin)
            ymin += -min(0, ymin)
        finally:
            image = image[ymin:ymax, xmin:xmax, ...].copy()

        image = cv2.resize(image, (self.target_size[0], self.target_size[1]), interpolation=interpolation)
        return image
    
    def __call__(self, images):
        h, w, _ = images[0].shape
        th = tw = min(images[0].shape[:-1])
        if w == tw and h == th:
            i = j = 0
            c = h
            k = w
        
        try:
            i = random.randint(0, w - tw)
        except ValueError:
            i = random.randint(w - tw, 0)
        try:
            j = random.randint(0, h - th)
        except ValueError:
            j = random.randint(h - th, 0)
        c = tw
        k = th
        images[0] = self.resized_crop(images[0], i, j, c, k, interpolation=self.interpolation)
        images[1] = self.resized_crop(images[1], i, j, c, k, interpolation=self.interpolation)
        images[2] = self.resized_crop(images[2], i, j, c, k, interpolation=self.interpolation)
        images[3] = self.resized_crop(images[3], i, j, c, k, interpolation=cv2.INTER_NEAREST)
        return images
        
    
class RandomResizeCrop:
    def __init__(self, target_size=(768, 768, 3), interpolation=cv2.INTER_CUBIC, prob=0.5):
        self.target_size   = target_size
        self.interpolation = interpolation
        self.prob          = prob
        
    def __call__(self, images):
        self.aug        = ResizeCrop(target_size=self.target_size, interpolation=self.interpolation)
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            images = self.aug(images)
        return images