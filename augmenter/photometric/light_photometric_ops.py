import cv2
import numpy as np


class LightIntensity:
    def __init__(self, hue=.1, sat=0.7, val=0.4, color_space='RGB'):
        self.hue         = hue
        self.sat         = sat
        self.val         = val
        self.color_space = color_space
        
    def __call__(self, images):
        images[0] = np.array(images[0], np.uint8)
        if self.color_space.lower() == "bgr":
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_RGB2HSV)
        elif self.color_space.lower() == "rgb":
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_RGB2HSV)
        r             = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        hue, sat, val = cv2.split(images[0])
        dtype         = images[0].dtype
        x             = np.arange(0, 256, dtype=r.dtype)
        lut_hue       = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        images[0]   = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        if self.color_space.lower() == "bgr":
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_HSV2RGB)
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
        elif self.color_space.lower() == "rgb":
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_HSV2RGB)
        return images


class RandomLightIntensity:
    def __init__(self, hue_range=.1, sat_range=0.7, val_range=0.4, color_space='RGB', prob=0.5):
        self.hue_range   = hue_range
        self.sat_range   = sat_range
        self.val_range   = val_range
        self.color_space = color_space
        self.prob        = prob

    def __call__(self, images):
        if isinstance(self.hue_range, (list, tuple)):
            hue = float(np.random.choice(self.hue_range))
        else:
            hue = float(np.random.uniform(-self.hue_range, self.hue_range))

        if isinstance(self.sat_range, (list, tuple)):
            sat = float(np.random.choice(self.sat_range))
        else:
            sat = float(np.random.uniform(-self.sat_range, self.sat_range))

        if isinstance(self.val_range, (list, tuple)):
            val = float(np.random.choice(self.val_range))
        else:
            val = float(np.random.uniform(-self.val_range, self.val_range))
            
        self.aug = LightIntensity(hue, sat, val, self.color_space)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            images = self.aug(images)
        return images
