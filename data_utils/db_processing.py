import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


def distance_caculator(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = square_distance_1 * square_distance_2 * square_sin / square_distance
    result[np.less_equal(result, 0)] = 0
    result = np.sqrt(result)
    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def extend_line(point_1, point_2, result, shrink_ratio=0.4):
    ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
                  int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + shrink_ratio))))
    cv2.line(result, tuple(ex_point_1), tuple(point_1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
    ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
                  int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + shrink_ratio))))
    cv2.line(result, tuple(ex_point_2), tuple(point_2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
    return ex_point_1, ex_point_2


def get_dbnet_groundtruth(polygons, size, min_text_size=8, shrink_ratio=0.4, thresh_max=0.7, thresh_min=0.3):
    height, width = size
    gt_map = np.zeros((1, height, width), dtype=np.float32)
    mask_map = np.ones((height, width), dtype=np.float32)
    thresh_map = np.zeros((height, width), dtype=np.float32)
    thresh_mask_map = np.zeros((height, width), dtype=np.float32)

    for polygon in polygons:
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        bh = max(polygon[:, 1]) - min(polygon[:, 1])
        bw = max(polygon[:, 0]) - min(polygon[:, 0])

        if min(bh, bw) < min_text_size:
            cv2.fillPoly(mask_map, polygon.astype(np.int32)[np.newaxis, :, :], 0)
        else:
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygon]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            padded_polygon = np.array(padding.Execute(distance)[0])
            cv2.fillPoly(thresh_mask_map, [padded_polygon.astype(np.int32)], 1.0)

            shrinked = padding.Execute(-distance)
            if shrinked == []:
                cv2.fillPoly(mask_map, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
            cv2.fillPoly(gt_map[0], [shrinked.astype(np.int32)], 1)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        iw = xmax - xmin + 1
        ih = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, iw - 1, num=iw).reshape(1, iw), (ih, iw))
        ys = np.broadcast_to(np.linspace(0, ih - 1, num=ih).reshape(ih, 1), (ih, iw))

        distance_map = np.zeros((polygon.shape[0], ih, iw), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = distance_caculator(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), thresh_map.shape[1] - 1)
        xmax_valid = min(max(0, xmax), thresh_map.shape[1] - 1)
        ymin_valid = min(max(0, ymin), thresh_map.shape[0] - 1)
        ymax_valid = min(max(0, ymax), thresh_map.shape[0] - 1)
        thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin: ymax_valid - ymax + ih,
                             xmin_valid - xmin: xmax_valid - xmax + iw],
            thresh_map[ymin_valid: ymax_valid + 1, xmin_valid: xmax_valid + 1])

    thresh_map = thresh_map * (thresh_max - thresh_min) + thresh_min
    return gt_map, mask_map, thresh_map, thresh_mask_map