import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class Brightness:
    def __init__(self, delta=100):
        self.value = 1 + delta / 255
        
    def __call__(self, images):
        img_shape = images[0].shape
        if len(img_shape) > 2:
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_RGB2HSV)
            hsv = np.array(images[0], dtype=np.float32)
    
            hsv[:, :, 1] = hsv[:, :, 1] * self.value
            hsv[:, :, 2] = hsv[:, :, 2] * self.value
    
            hsv = np.uint8(np.clip(hsv, 0, 255))
    
            images[0] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return images
    

class RandomBrightness:
    def __init__(self, delta_range=100, prob=0.5):
        self.delta_range = delta_range
        self.prob        = prob

    def __call__(self, images):
        if isinstance(self.delta_range, (list, tuple)):
            delta = float(np.random.uniform(*self.delta_range))
        else:
            delta = float(np.random.uniform(-self.delta_range, self.delta_range))
            
        self.aug = Brightness(delta)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            images = self.aug(images)
        return images