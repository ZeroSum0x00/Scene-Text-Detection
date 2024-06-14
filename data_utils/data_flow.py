import os
import re
import cv2
import random
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

from augmenter import build_augmenter
from data_utils.data_augmentation import Augmentor
from data_utils.craft_processing import get_craft_groundtruth
from data_utils.db_processing import get_dbnet_groundtruth
from data_utils.data_processing import extract_data_folder, get_data, Normalizer
from utils.auxiliary_processing import change_color_space
from utils.boxEnlarge import polygonArea
from utils.logger import logger


def get_train_test_data(data_dirs,
                        annotation_dirs,
                        target_size,
                        batch_size,
                        perspective_transfrom,
                        data_model_type='craft',
                        color_space='RGB',
                        load_bbox='character',
                        augmentor=None,
                        normalizer='divide',
                        mean_norm=None,
                        std_norm=None,
                        data_type='txt',
                        check_data=False,
                        load_memory=False,
                        dataloader_mode=0,
                        *args, **kwargs):
    """
        dataloader_mode = 0:   train - validation - test
        dataloader_mode = 1:   train - validation
        dataloader_mode = 2:   train
    """

    data_train = get_data(data_dirs, 
                          annotation_dirs   = annotation_dirs,
                          data_type         = data_type,
                          phase             = 'train',
                          check_data        = check_data,
                          load_memory       = load_memory)
                            
    train_generator = Data_Sequence(data_train, 
                                    target_size           = target_size, 
                                    batch_size            = batch_size, 
                                    perspective_transfrom = perspective_transfrom,
                                    data_model_type       = data_model_type,
                                    color_space           = color_space,
                                    load_bbox             = load_bbox,
                                    augmentor             = augmentor,
                                    normalizer            = normalizer,
                                    mean_norm             = mean_norm,
                                    std_norm              = std_norm,
                                    phase                 = 'train',
                                    *args, **kwargs)

    if dataloader_mode != 2:
        data_valid = get_data(data_dirs,
                              annotation_dirs   = annotation_dirs,
                              data_type         = data_type,
                              phase             = 'validation',
                              check_data        = check_data,
                              load_memory       = load_memory)
        valid_generator = Data_Sequence(data_valid, 
                                        target_size           = target_size, 
                                        batch_size            = batch_size, 
                                        perspective_transfrom = perspective_transfrom,
                                        data_model_type       = data_model_type,
                                        color_space           = color_space,
                                        load_bbox             = load_bbox,
                                        augmentor             = augmentor,
                                        normalizer            = normalizer,
                                        mean_norm             = mean_norm,
                                        std_norm              = std_norm,
                                        phase                 = 'valid',
                                        *args, **kwargs)
    else:
        valid_generator = None

    if dataloader_mode != 1:
        data_test  = get_data(data_dirs,
                              annotation_dirs   = annotation_dirs,
                              data_type         = data_type,
                              phase             = 'test',
                              check_data        = check_data,
                              load_memory       = load_memory)
        test_generator  = Data_Sequence(data_valid, 
                                        target_size           = target_size, 
                                        batch_size            = batch_size, 
                                        perspective_transfrom = perspective_transfrom,
                                        data_model_type       = data_model_type,
                                        color_space           = color_space,
                                        load_bbox             = load_bbox,
                                        augmentor             = augmentor,
                                        normalizer            = normalizer,
                                        mean_norm             = mean_norm,
                                        std_norm              = std_norm,
                                        phase                 = 'test',
                                        *args, **kwargs)
    else:
        test_generator = None
        
    logger.info('Load data successfully')
    return train_generator, valid_generator, test_generator


class Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size, 
                 batch_size, 
                 perspective_transfrom,
                 data_model_type="CRAFT",
                 color_space='RGB',
                 load_bbox='char',
                 augmentor=None, 
                 normalizer=None,
                 mean_norm=None, 
                 std_norm=None, 
                 phase='train', 
                 debug_mode=False):
        self.dataset = dataset
        self.target_size = target_size
        self.batch_size = batch_size

        if phase == "train":
            self.dataset = shuffle(self.dataset)
        self.N = self.n = len(self.dataset)
        
        self.data_model_type = data_model_type
        self.color_space     = color_space
        self.load_bbox       = load_bbox
        self.phase           = phase
        
        additional_config = {
             'RandomResizeCrop': {'target_size': target_size},
             'LightIntensityChange': {'color_space': color_space}
        }

        
        self.debug_mode      = debug_mode
                     
        self.normalizer = Normalizer(normalizer, 
                                     data_model=data_model_type,
                                     mean=mean_norm, 
                                     std=std_norm)

        if augmentor[phase] and isinstance(augmentor[phase], (tuple, list)):
            self.augmentor = Augmentor(augment_objects=build_augmenter(augmentor[phase],
                                                                       additional_config=additional_config ))
        else:
            self.augmentor = augmentor[phase]
        
        self.perspective_transfrom = perspective_transfrom
        
    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

#     def validate_polygons(self, polygons, size):
#         h, w = size
#         for polygon in polygons:
#             for poly in polygon:
#                 poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
#                 poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                
#                 area = polygonArea(poly)
#                 if area > 0:
#                     poly = poly[::-1, :]
                    
#         return polygons
    
    def __getitem__(self, index):
        batch_image = []
        debug_image = []
        batch_region_score = []
        batch_affinity_score = []
        batch_confidence_mask = []
        
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i           = i % self.N
            sample = self.dataset[i]

            if len(self.target_size) == 2 or self.target_size[-1] == 1:
                deep_channel = 0
            else:
                deep_channel = 1

            if sample['image'] is not None:
                image = sample['image']
                image = change_color_space(image, 'bgr', self.color_space if deep_channel else 'gray')
            else:
                img_path = os.path.join(sample['path'], sample['filename'])
                image = cv2.imread(img_path, deep_channel)
                image = change_color_space(image, 'bgr' if deep_channel else 'gray', self.color_space)
            
            if self.load_bbox.lower() in ['char', 'character']:
                bboxes = sample['char_bboxes']
            else:
                bboxes = sample['word_bboxes']
                
#             fixed_bboxes = []
#             for j, bbox in enumerate(bboxes.copy()):
#                 if np.any(bbox):
#                     fixed_bboxes.append(bbox)
#             fixed_bboxes = self.validate_polygons(fixed_bboxes, sample['image_size'])
            
            words = sample['words']
            
            if self.data_model_type.lower() == "craft":
                region_score, affinity_score, confidence_mask = get_craft_groundtruth(bboxes, words, self.perspective_transfrom, sample['image_size'])

                if self.augmentor:
                    image, region_score, affinity_score, confidence_mask = self.augmentor([image, region_score, affinity_score, confidence_mask])

                image, region_score, affinity_score, confidence_mask = self.normalizer([image, region_score, affinity_score, confidence_mask],
                                                                                       target_size=self.target_size,
                                                                                       interpolation=cv2.INTER_CUBIC)

            else:
                fixed_bboxes = np.concatenate(fixed_bboxes, axis=0)
                    
            batch_image.append(image)
            batch_region_score.append(region_score)
            batch_affinity_score.append(affinity_score)
            batch_confidence_mask.append(confidence_mask)
            
        batch_image = np.array(batch_image)
        batch_region_score = np.array(batch_region_score)
        batch_affinity_score = np.array(batch_affinity_score)
        batch_confidence_mask = np.array(batch_confidence_mask)
        
        return batch_image, [batch_region_score, batch_affinity_score, batch_confidence_mask]