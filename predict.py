import os
import cv2
import math
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from models import build_models
from utils.files import get_files
from utils.post_processing import detect_image
from utils.config_processing import load_config


def predict(file_config=None):
    config = load_config(file_config)
    test_config = config['Test']
    data_config  = config['Dataset']
    data_path = test_config['data_infer']
    
    model, _ = build_models(config['Model'], './20240607-085744/weights/best_weights_hmean')

    images = get_files(data_path, extensions=['jpg', 'png'])
    target_shape = config['Model']['input_shape']

    total_num = len(images)
    n_correct = 0
    for name in tqdm(images):
        if len(target_shape) == 2 or target_shape[-1] == 1:
            read_mode = 0
        else:
            read_mode = 1
        
        boxes = detect_image(f"{data_path}/{name}",
                             model,
                             color_space='rgb',
                             normalize='divide',
                             text_threshold=0.7, 
                             link_threshold=0.4,
                             low_text=0.4, 
                             estimate_num_chars=None,
                             canvas_size=2560, 
                             mag_ratio=1., 
                             min_size=20,
                             slope_ths=0.1,
                             ycenter_ths=0.5, 
                             height_ths=0.5, 
                             width_ths=0.5, 
                             add_margin=0.1,
                             save_result=False, 
                             verbose=False)
        import pickle

        a = {'rboxes': boxes}

        # with open('bboxes_predict_processed.pickle', 'wb') as handle:
        #     pickle.dump(a, handle)
if __name__ == '__main__':
    predict('./configs/craft.yaml')