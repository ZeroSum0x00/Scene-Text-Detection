import os
import cv2
import math
import numpy as np
import tensorflow as tf
from utils.auxiliary_processing import diff, change_color_space


def resize_aspect_ratio(image, square_size, mag_ratio=1, interpolation=None):
    height, width, channel = image.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(image, (target_w, target_h), interpolation = interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_h/2), int(target_w/2))
    return resized, ratio, size_heatmap


def image_preprocessing(image, normalize='sub_divide', target_size=None, square_size=2560, mag_ratio=1, interpolation=None):
    if target_size is None:
        image, ratio, size_heatmap = resize_aspect_ratio(image, square_size, mag_ratio, interpolation)
    else:
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
            ratio = 1
            size_heatmap = (int(image.shape[0] / 2), int(image.shape[1] / 2))
            
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        
    if normalize == "sub_divide":
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = np.clip(image, -1, 1)
    elif normalize == "divide":
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.clip(image, 0, 1)
    else:
        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
    
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    return image, ratio, size_heatmap


def get_heatmap_from_image(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def get_detect_bboxes(text_map, link_map, text_threshold=0.7, link_threshold=0.4, low_text=0.4, estimate_num_chars=False):
    img_h, img_w = text_map.shape

    """ labeling method """
    ret, text_score = cv2.threshold(text_map, low_text, 1, 0)
    ret, link_score = cv2.threshold(link_map, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1, n_labels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(text_map[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(text_map.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        if estimate_num_chars:
            _, character_locs = cv2.threshold((text_map - link_map) * segmap /255., text_threshold, 1, 0)
            _, n_chars = label(character_locs)
            mapper.append(n_chars)
        else:
            mapper.append(k)
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        n_iter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - n_iter, x + w + n_iter + 1, y - n_iter, y + h + n_iter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + n_iter, 1 + n_iter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        
        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)

    return det, labels, mapper


def get_detect_polygons(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 30 or h < 30:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def get_detect_result(text_map, link_map, text_threshold, link_threshold, low_text, poly=False, estimate_num_chars=False):
    boxes, labels, mapper = get_detect_bboxes(text_map, 
                                              link_map, 
                                              text_threshold, 
                                              link_threshold, 
                                              low_text, 
                                              estimate_num_chars=estimate_num_chars)

    if poly:
        polys = get_detect_polygons(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def group_text_box(polys, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 1.0, add_margin = 0.05, sort_output = True):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list,combined_list, merged_list = [],[],[],[]

    for poly in polys:
        slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
        slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0],poly[2],poly[4],poly[6]])
            x_min = min([poly[0],poly[2],poly[4],poly[6]])
            y_max = max([poly[1],poly[3],poly[5],poly[7]])
            y_min = min([poly[1],poly[3],poly[5],poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
        else:
            height = np.linalg.norm([poly[6]-poly[0],poly[7]-poly[1]])
            width = np.linalg.norm([poly[2]-poly[0],poly[3]-poly[1]])

            margin = int(1.44*add_margin*min(width, height))

            theta13 = abs(np.arctan( (poly[1]-poly[5])/np.maximum(10, (poly[0]-poly[4]))))
            theta24 = abs(np.arctan( (poly[3]-poly[7])/np.maximum(10, (poly[2]-poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13)*margin
            y1 = poly[1] - np.sin(theta13)*margin
            x2 = poly[2] + np.cos(theta24)*margin
            y2 = poly[3] - np.sin(theta24)*margin
            x3 = poly[4] + np.cos(theta13)*margin
            y3 = poly[5] + np.sin(theta13)*margin
            x4 = poly[6] - np.cos(theta24)*margin
            y4 = poly[7] + np.sin(theta24)*margin

            free_list.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1: # one box per line
            box = boxes[0]
            margin = int(add_margin*min(box[1]-box[0],box[5]))
            merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin])
        else: # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [],[]
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths*np.mean(b_height)) and ((box[0]-x_max) < width_ths *(box[3]-box[2])): # merge boxes
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) >0: merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1: # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin])
                else: # non adjacent box in same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin])
    # may need to check if box is really in image
    return merged_list, free_list


def get_boxes_from_model_result(region_score, affinity_score, target_ratio, text_threshold, link_threshold, low_text, optimal_num_chars):
    result = []
    ratio_h = ratio_w = 1 / target_ratio
    estimate_num_chars = optimal_num_chars is not None
    
    if not isinstance(region_score, np.ndarray):
        try:
            region_score = region_score.numpy()
        except:
            region_score = np.array(region_score)
            
    if not isinstance(affinity_score, np.ndarray):
        try:
            affinity_score = affinity_score.numpy()
        except:
            affinity_score = np.array(affinity_score)
            
    boxes_list, polys_list = [], []
    for region, affinity in zip(region_score, affinity_score):
        boxes, polys, mapper = get_detect_result(region, 
                                                 affinity, 
                                                 text_threshold=text_threshold,
                                                 link_threshold=link_threshold,
                                                 low_text=low_text,
                                                 poly=False,
                                                 estimate_num_chars=optimal_num_chars)

        boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
        
        if optimal_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if optimal_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)
    return result


def merge_boxes(boxes_list, min_size=20, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, add_margin=0.1, optimal_num_chars=None):
    horizontal_list_agg, free_list_agg = [], []
    for text_box in boxes_list:
        horizontal_list, free_list = group_text_box(text_box, 
                                                    slope_ths,
                                                    ycenter_ths, 
                                                    height_ths,
                                                    width_ths, 
                                                    add_margin,
                                                    (optimal_num_chars is None))
        if min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
        horizontal_list_agg.append(horizontal_list)
        free_list_agg.append(free_list)
    return horizontal_list_agg, free_list_agg

        
def detect_image(image, model, 
                 color_space='rgb', normalize='divide', 
                 text_threshold=0.7, link_threshold=0.4,
                 low_text=0.4, estimate_num_chars=None,
                 canvas_size=2560, mag_ratio=1., 
                 min_size=20, slope_ths=0.1, 
                 ycenter_ths=0.5, height_ths=0.5, 
                 width_ths=0.5, add_margin=0.1, 
                 save_result=False, verbose=False):

    if isinstance(image, str):
        image = cv2.imread(image)
        image = change_color_space(image, 'bgr', color_space)
    
    height, width, _ = image.shape
    image, target_ratio, size_heatmap = image_preprocessing(image, 
                                                            normalize=normalize,
                                                            target_size=None, 
                                                            square_size=canvas_size, 
                                                            mag_ratio=mag_ratio,
                                                            interpolation=cv2.INTER_LINEAR)
    region_score, affinity_score = model.predict(image)
    boxes_list = get_boxes_from_model_result(region_score, 
                                             affinity_score,
                                             target_ratio,
                                             text_threshold=text_threshold,
                                             link_threshold=link_threshold,
                                             low_text=low_text,
                                             optimal_num_chars=estimate_num_chars)
    horizontal_list, free_list = merge_boxes(boxes_list, 
                                             min_size=min_size,
                                             slope_ths=slope_ths, 
                                             ycenter_ths=ycenter_ths, 
                                             height_ths=height_ths, 
                                             width_ths=width_ths, 
                                             add_margin=add_margin, 
                                             optimal_num_chars=estimate_num_chars)
    
    bboxes = []
    for box in horizontal_list[0]:
        xmin = max(0, box[0])
        ymin = max(0, box[2])
        xmax = min(box[1], width)
        ymax = min(box[3], height)
        bboxes.append([xmin, ymin, xmax, ymax])
    
    visual_image = np.squeeze(image.copy(), 0)
    for box in bboxes:
        cv2.rectangle(visual_image, (box[0], box[1]), (box[2], box[3]), (255, 0,0), 2) 
    print(visual_image.shape)
    cv2.imwrite('saved_weights/results.jpg', visual_image* 255)
    return bboxes

