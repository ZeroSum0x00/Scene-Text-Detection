import os
import cv2
import math
import types
import random
import numpy as np

from utils.files import extract_zip, verify_folder, get_files
from data_utils import ParseSynthText


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = verify_folder(data_destination) + folder_name 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def get_data(data_dirs, 
             annotation_dirs = None,
             data_type       = 'txt',
             phase           = 'train', 
             check_data      = False,
             load_memory     = False,
             *args, **kwargs):
    
    def load_data(data_dir, annotation_dir):
        if data_type.lower() == 'synthtext':
            annotation_dir = os.path.join(annotation_dir, f'{phase}_gt.mat')
            parser = ParseSynthText(data_dir, annotation_dir, load_memory, check_data=check_data, *args, **kwargs)
        return parser()
    
    assert data_type.lower() in ('synthtext')
    data_extraction = []
    
    data_dir = os.path.join(data_dirs, phase)
    parser = load_data(data_dir, annotation_dirs)
    data_extraction.extend(parser)
    return data_extraction


class Normalizer():
    def __init__(self, norm_type="divide", data_model="craft", mean=None, std=None):
        self.norm_type  = norm_type
        self.data_model = data_model
        self.mean       = mean
        self.std        = std
            
    def __get_standard_deviation(self, img):
        if self.mean is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.mean, float) or isinstance(self.mean, int):
                    img[..., i] -= self.mean
                else:
                    img[..., i] -= self.mean[i]

        if self.std is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.std, float) or isinstance(self.std, int):
                    img[..., i] /= (self.std + 1e-20)
                else:
                    img[..., i] /= (self.std[i] + 1e-20)
        return img

    def _sub_divide(self, images, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (images[0].shape[0] != target_size[0] or images[0].shape[1] != target_size[1]):
            images[0] = cv2.resize(images[0], (target_size[1], target_size[0]), interpolation=interpolation)
            if self.data_model.lower() == 'craft':
                images[1] = cv2.resize(images[1], (target_size[1] // 2, target_size[0] // 2), interpolation=interpolation)
                images[2] = cv2.resize(images[2], (target_size[1] // 2, target_size[0] // 2), interpolation=interpolation)
                images[3] = cv2.resize(images[3], (target_size[1] // 2, target_size[0] // 2), interpolation=cv2.INTER_NEAREST)

        if len(images[0].shape) == 2:
            images[0] = np.expand_dims(images[0], axis=-1)
        images[0] = images[0].astype(np.float32)
        images[0] = images[0] / 127.5 - 1
        images[0] = self.__get_standard_deviation(images[0])
        images[0] = np.clip(images[0], -1, 1)      
        return images
    
    def _divide(self, images, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (images[0].shape[0] != target_size[0] or images[0].shape[1] != target_size[1]):
            images[0] = cv2.resize(images[0], (target_size[1], target_size[0]), interpolation=interpolation)
            if self.data_model.lower() == 'craft':
                images[1] = cv2.resize(images[1], (target_size[1] // 2, target_size[0] // 2), interpolation=interpolation)
                images[2] = cv2.resize(images[2], (target_size[1] // 2, target_size[0] // 2), interpolation=interpolation)
                images[3] = cv2.resize(images[3], (target_size[1] // 2, target_size[0] // 2), interpolation=cv2.INTER_NEAREST)

        if len(images[0].shape) == 2:
            images[0] = np.expand_dims(images[0], axis=-1)
        images[0] = images[0].astype(np.float32)
        images[0] = images[0] / 255.0
        images[0] = self.__get_standard_deviation(images[0])
        images[0] = np.clip(images[0], 0, 1)
        return images

    def _basic(self, images, target_size=None, interpolation=None, keep_ratio_with_pad=False):
        if target_size and (images[0].shape[0] != target_size[0] or images[0].shape[1] != target_size[1]):
            images[0] = cv2.resize(images[0], (target_size[1], target_size[0]), interpolation=interpolation)
            if self.data_model.lower() == 'craft':
                images[1] = cv2.resize(images[1], (target_size[1] // 2, target_size[0] // 2), interpolation=interpolation)
                images[2] = cv2.resize(images[2], (target_size[1] // 2, target_size[0] // 2), interpolation=interpolation)
                images[3] = cv2.resize(images[3], (target_size[1] // 2, target_size[0] // 2), interpolation=cv2.INTER_NEAREST)

        if len(images[0].shape) == 2:
            images[0] = np.expand_dims(images[0], axis=-1)
        images[0] = images[0].astype(np.float32)
        images[0] = self.__get_standard_deviation(images[0])
        images[0] = images[0].astype(np.uint8)
        images[0] = np.clip(images[0], 0, 255)
        return images

    def __call__(self, input, *args, **kargs):
        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide(input, *args, **kargs)
            elif self.norm_type == "sub_divide":
                return self._sub_divide(input, *args, **kargs)
        elif isinstance(self.norm_type, types.FunctionType):
            return self._func_calc(input, self.norm_type, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)
