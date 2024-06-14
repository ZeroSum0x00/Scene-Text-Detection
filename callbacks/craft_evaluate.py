import os
import cv2
import shutil
import colorsys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils.detect_iou_calculator import DetectionIoUEvaluator
from utils.post_processing import image_preprocessing, get_detect_result, adjust_result_coordinates, get_heatmap_from_image
from utils.auxiliary_processing import change_color_space
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



class CRAFTEvaluate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path         = None,
                 color_space         = 'BGR',
                 data_normalize      = 'divide',
                 eval_canvas_size    = 1280,
                 eval_mag_ratio      = 1.5,
                 eval_text_threshold = 0.7,
                 eval_link_threshold = 0.4,
                 eval_low_text       = 0.4,
                 return_poly         = False,
                 min_ratio           = 0.2,
                 saved_best          = True,
                 show_frequency      = 100):
        super(CRAFTEvaluate, self).__init__()
        self.result_path         = result_path
        self.color_space         = color_space
        self.data_normalize      = data_normalize
        self.eval_canvas_size    = eval_canvas_size
        self.eval_mag_ratio      = eval_mag_ratio
        self.eval_text_threshold = eval_text_threshold
        self.eval_link_threshold = eval_link_threshold
        self.eval_low_text       = eval_low_text
        self.return_poly         = return_poly
        self.min_ratio           = min_ratio
        self.saved_best          = saved_best
        self.show_frequency      = show_frequency
        num_metrics              = 3
        self.metric_values       = [[0.0] for i in range(num_metrics)]
        self.epoches             = [0]
        self.current_metric      = [0.0 for i in range(num_metrics)]
        self.eval_dataset        = None
        self.data_path           = None
        self.evaluator           = DetectionIoUEvaluator()
        
    def pass_data(self, data):
        self.eval_dataset = data
        self.total_imgs_bboxes_gt = self.get_groundtruth_normed()
        
    def get_groundtruth_normed(self):
        gt_results = []
        if self.eval_dataset is not None:
            for ann_dataset in self.eval_dataset.dataset:
                img_bboxes = []
                words = ann_dataset['words']
                bboxes = ann_dataset['word_bboxes']
                filename = ann_dataset['filename']
                for word, bbox in zip(words, np.concatenate(bboxes, 0)):
                    gt_box_dict = {"text": None, "points": None, "ignore": None}
                    gt_box_dict["text"] = word
                    gt_box_dict["points"] = np.array(bbox).reshape(-1, 2)
                    gt_box_dict["ignore"] = False
                    gt_box_dict['filename'] = filename
                    img_bboxes.append(gt_box_dict)
                gt_results.append(img_bboxes)
        return gt_results
    
    def basic_model_process(self, image, target_ratio, size_heatmap):
        ratio_h = ratio_w = 1 / target_ratio
        region_score, affinity_score = self.model.predict(image)
        region_score = region_score[0, : size_heatmap[0], : size_heatmap[1]].numpy()
        affinity_score = affinity_score[0, : size_heatmap[0], : size_heatmap[1]].numpy()
        boxes, polys, mapper = get_detect_result(region_score, 
                                                 affinity_score, 
                                                 text_threshold=self.eval_text_threshold,
                                                 link_threshold=self.eval_link_threshold,
                                                 low_text=self.eval_low_text,
                                                 poly=self.return_poly,
                                                 estimate_num_chars=False)
        boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        render_region_score = get_heatmap_from_image(region_score)
        render_affinity_score = get_heatmap_from_image(affinity_score)
        render_img = [render_region_score, render_affinity_score]
        return boxes, polys, render_img
        
    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        if temp_epoch % self.show_frequency == 0:
            if self.eval_dataset is not None:
                print("\nGet IOU.")
                
                total_imgs_bboxes_pre = []
                for ann_dataset in tqdm(self.eval_dataset.dataset):
                    single_img_bbox = []
                    img_path = os.path.join(ann_dataset['path'], ann_dataset['filename'])
                    
                    image = cv2.imread(img_path)
                    image = change_color_space(image, 'bgr', self.color_space)
                    original_image_shape = image.shape
                    image, target_ratio, size_heatmap = image_preprocessing(image,
                                                                            normalize=self.data_normalize,
                                                                            target_size=None, 
                                                                            square_size=self.eval_canvas_size, 
                                                                            mag_ratio=self.eval_mag_ratio,
                                                                            interpolation=cv2.INTER_LINEAR)
                    boxes, polys, score_text = self.basic_model_process(image, target_ratio, size_heatmap)
                    
                    for box in boxes:
                        box_info = {"text": "###", "points": box, "ignore": False, 'filename': ann_dataset['filename']}
                        single_img_bbox.append(box_info)
                    total_imgs_bboxes_pre.append(single_img_bbox)

                print("Calculate IoU.")
                results = []
                for i, (gt, pred) in enumerate(zip(self.total_imgs_bboxes_gt, total_imgs_bboxes_pre)):
                    perSampleMetrics_dict = self.evaluator.evaluate_image(gt, pred)
                    results.append(perSampleMetrics_dict)
                    
                eval_metrics = self.evaluator.combine_results(results, self.model.architecture.__class__.__name__)
                
                metric_title = []
                metric_color = ['deeppink', 'orange', 'purple']
                for i, (key, value) in enumerate(eval_metrics.items()):
                    metric_title.append(key)
                    self.metric_values[i].append(value)
                    
                if self.saved_best:
                    for j, (key, value) in enumerate(eval_metrics.items()):
                        if value > self.current_metric[j] and value > self.min_ratio:
                            logger.info(f'{key} metric increase {self.current_metric[j]*100:.2f}% to {value*100:.2f}%')
                            logger.info(f'Save best {key} weights to {os.path.join(self.result_path, "weights", f"best_weights_{key}")}')                    
                            self.model.save_weights(os.path.join(self.result_path, "weights", f"best_weights_{key}"))
                            self.current_metric[j] = value
                            
                self.epoches.append(temp_epoch)
                for k, (key, value) in enumerate(eval_metrics.items()):
                    with open(os.path.join(self.result_path, 'summary', f"epoch_{key}.txt"), 'a') as f:
                        f.write(f"{key} metric in epoch {epoch + 1}: {value}\n")
                        
                f = plt.figure()
                max_height = np.max(self.metric_values)
                max_width  = np.max(self.epoches)

                for i in range(len(self.metric_values)):
                    max_index = np.argmax(self.metric_values[i])

                    linewidth = 2
                    plt.plot(self.epoches, self.metric_values[i], linewidth=linewidth, color=metric_color[i], label=metric_title[i])
                        
                    if round(np.max(self.metric_values[i]), 3) <= 0.:
                        continue

                    temp_text = plt.text(0, 0, 
                                         f'{self.metric_values[i][max_index]:0.3f}', 
                                         alpha=0,
                                         fontsize=8, 
                                         fontweight=600,
                                         color='white')
                    r = f.canvas.get_renderer()
                    bb = temp_text.get_window_extent(renderer=r)
                    width = bb.width
                    height = bb.height
                    text = plt.text(self.epoches[max_index] + (width * 0.00027 + 0.01) * max_width, 
                                    self.metric_values[i][max_index] + (height * 0.0017 + 0.012) * max_height, 
                                    f'{self.metric_values[i][max_index]:0.3f}', 
                                    fontsize=8, 
                                    fontweight=600,
                                    color='white')
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (self.epoches[max_index] + width * 0.00027 * max_width, self.metric_values[i][max_index] + height * 0.0017 * max_height),
                            width * 0.003 * max_width,
                            height * 0.005 * max_height,
                            # alpha=0.85,
                            facecolor='hotpink'
                    ))
                    plt.scatter(self.epoches[max_index], self.metric_values[i][max_index], s=80, facecolor='red')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Metrics')
                plt.title('A metrics graph')
                plt.legend(fontsize=7, loc="upper left")
    
                plt.savefig(os.path.join(self.result_path, 'summary', "epoch_metrics.png"))
                plt.cla()
                plt.close("all")
            else:
                print('\nYou need to pass data in using the pass_data function.')