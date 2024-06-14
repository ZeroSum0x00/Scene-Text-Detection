import re
import os
import cv2
import itertools
import imagesize
import numpy as np
import scipy.io as sio
from tqdm import tqdm


class ParseSynthText:
    def __init__(self, 
                 data_dir,
                 annotation_file,
                 load_memory=False,
                 check_data=False):
        self.data_dir = data_dir
        try:
            self.mat_env = sio.loadmat(annotation_file)
        except BaseException as e:
            print(e)
        
        self.load_memory = load_memory
        self.check_data = check_data

    def get_bboxes(self, bbox_list, words):
        compare_bbox = []
        if np.any(bbox_list):
            if len(bbox_list.shape) == 2:
                bbox_list = np.expand_dims(bbox_list, axis=-1)

            bbox_list = bbox_list.transpose((2, 1, 0))

            idx = 0
            for i in range(len(words)):
                length_of_word = len(words[i])
                bbox = bbox_list[idx : idx + length_of_word]
                idx += length_of_word
                bbox = np.array(bbox)
                compare_bbox.append(bbox)
        return compare_bbox
            
    def __call__(self):
        data_extraction = []
        filename_list = self.mat_env["imnames"][0]
        word_list = self.mat_env["txt"][0]
        char_bbox_list = self.mat_env["charBB"][0]
        word_bbox_list = self.mat_env["wordBB"][0]
            
        for idx in tqdm(range(len(filename_list))):
            info_dict = {}
            filename = str(filename_list[idx][0]).strip()
            info_dict['filename'] = os.path.basename(filename)
            info_dict['image'] = None
            image_path = os.path.join(self.data_dir, filename)
            info_dict['path'] = os.path.dirname(image_path)
            width, height = imagesize.get(image_path)
            info_dict['image_size'] = (height, width)
            
            if self.check_data:
                try:
                    # valid_image(image_path)
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    shape = img.shape
                except Exception as e:
                    os.remove(image_path)
                    print(f"Error: File {filename} is can't loaded: {e}")
                    continue
                    
            if self.load_memory:
                img = cv2.imread(image_path)
                info_dict['image'] = img
                
            words = [re.split(" \n|\n |\n| ", word.strip()) for word in word_list[idx]]
            words = list(itertools.chain(*words))
            words = [word for word in words if len(word) > 0]
            
            if len(char_bbox_list) > 0:
                char_compare_bbox = self.get_bboxes(np.array(char_bbox_list[idx]), words)
            else:
                char_compare_bbox = []

            word_compare_bbox = self.get_bboxes(np.array(word_bbox_list[idx]), words)

            info_dict['char_bboxes'] = char_compare_bbox
            info_dict['word_bboxes'] = word_compare_bbox
            info_dict['words'] = words

            if len(info_dict['char_bboxes']) > 0 or len(info_dict['word_bboxes']) > 0:
                data_extraction.append(info_dict)
        return data_extraction