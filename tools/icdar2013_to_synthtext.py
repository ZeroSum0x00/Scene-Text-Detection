import os
import sys
import cv2
import shutil
import numpy as np
import scipy.io as scio

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.files import get_files


def ICDAR2013_to_SynthText(data_path,
                           result_path='./',
                           mat_temp_path='./configs/synthtext_temple.mat',
                           phase='train',
                           ):
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, phase), exist_ok=True)

    temp_data = scio.loadmat(mat_temp_path)
    wordBB = temp_data['wordBB'][0][:1]
    charBB = temp_data['charBB'][0][:1]
    
    if phase == 'train':
        img_folder = 'Challenge2_Training_Task12_Images'
        gt_folder = 'Challenge2_Training_Task1_GT'
    else:
        img_folder = 'Challenge2_Test_Task12_Images'
        gt_folder = 'Challenge2_Test_Task1_GT'

    image_path = os.path.join(data_path, img_folder)
    label_path = os.path.join(data_path, gt_folder)

    image_files = get_files(image_path, ['jpg', 'png'])
    label_files = get_files(label_path, ['txt'])
    
    
    new_wordBB = np.concatenate([wordBB for i in range(len(label_files))])
    new_charBB = np.concatenate([wordBB for i in range(len(label_files))])
    txt = []
    filename = []

    for idx, label_file in enumerate(label_files):
        image_name = '_'.join(label_file.split('_')[1:]).replace('txt', 'jpg')

        if image_name in image_files:
            with open(os.path.join(label_path, label_file)) as f:
                lines = f.readlines()

            text = []
            char_bbox = []
            word_bbox = []
            for data in lines:
                if phase == "train":
                    data = data.strip().split(' ')
                else:
                    data = data.strip().split(', ')

                if len(data) == 5:
                    xmin, ymin, xmax, ymax = [round(float(p)) for p in data[:4]]
                    label = data[-1].replace('"', '')
                    bbox = np.array([[xmin, ymin],
                                     [xmax, ymin],
                                     [xmax, ymax],
                                     [xmin, ymax]])
                    text.append("#")
                    word_bbox.append(bbox)

            filename.append([image_name])
            txt.append(text)
            word_bbox = np.array(word_bbox)
            word_bbox = word_bbox.transpose((2, 1, 0))

            if word_bbox.shape[-1] >= 1:
                new_wordBB[idx] = word_bbox
                new_charBB[idx] = np.array([])

            shutil.copy(os.path.join(image_path, image_name), os.path.join(result_path, phase, image_name))
            
            
    new_data = {
        '__header__': temp_data['__header__'],
        '__version__': '1.0',
        '__globals__': [],
        'charBB': [new_charBB],
        'wordBB': [new_wordBB],
        'imnames': [filename],
        'txt': [txt],
    }
    
    scio.savemat(os.path.join(result_path, f'{phase}_gt.mat'), new_data)
    
    
if __name__ == "__main__":
    phase = ['train', 'validation']
    for p in phase:
        ICDAR2013_to_SynthText(data_path="/home/vbpo-101386/Desktop/TuNIT/Datasets/Object Detection/ICDAR2013",
                               result_path="/home/vbpo-101386/Desktop/TuNIT/Datasets/Object Detection/ICDAR2013_new",
                               phase=p)