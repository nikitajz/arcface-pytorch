# coding: utf-8
# YuanYang
import math

import cv2
import numpy as np


def non_maximum_suppression(boxes, overlap_threshold, mode='Union'):
    """
    non max suppression
    重なりのある Bounding Box から同一の物体と判断したものを一つにまとめるアルゴリズム

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return pick


def normalize_and_centering_image(input_image):
    """
    adjust the input from (h, w, c) to ( 1, c, h, w) for network input
    * details
        * normalizing, centering : (x - 128) / 128
        * change to channel first
        * reshape to 4-dimension tensor

    Parameters:
    ----------
        input_image: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if input_image.dtype is not np.dtype('float32'):
        out_data = input_image.astype(np.float32)
    else:
        out_data = input_image

    out_data = out_data.transpose((2, 0, 1))
    # 4次元テンソルに変換
    out_data = np.expand_dims(out_data, 0)
    # 正規化
    out_data = (out_data - 127.5) * 0.0078125
    return out_data


def generate_bbox(map, reg, scale, threshold):
    """
    generate bbox from feature map
    
    Parameters:
    ----------
        map: numpy array , n x m x 1
            detect score for each position
        reg: numpy array , n x m x 4
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """
    stride = 2
    cellsize = 12

    t_index = np.where(map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = map[t_index[0], t_index[1]]
    boundingbox = np.vstack([np.round((stride * t_index[1] + 1) / scale),
                             np.round((stride * t_index[0] + 1) / scale),
                             np.round((stride * t_index[1] + 1 + cellsize) / scale),
                             np.round((stride * t_index[0] + 1 + cellsize) / scale),
                             score,
                             reg])

    return boundingbox.T


def detect_first_stage(img, net, scale, threshold):
    """
    run PNet for first stage
    
    Parameters:
    ----------
        img: numpy array, bgr order
            input image
        scale: float number
            how much should the input image scale
        net: PNet
            worker
    Returns:
    -------
        total_boxes : bboxes
    """
    height, width, _ = img.shape

    # math.ceil: x 以上の最大の整数値を返す
    hs = math.ceil(height * scale)
    ws = math.ceil(width * scale)

    # 画像をスケールしたサイズに縮める
    im_data = cv2.resize(img, (ws, hs))

    # adjust for the network input
    input_buf = normalize_and_centering_image(im_data)
    output = net.predict(input_buf)
    boxes = generate_bbox(output[1][0, 1, :, :], output[0], scale, threshold)

    if boxes.size == 0:
        return None

    # nms
    pick = non_maximum_suppression(boxes[:, 0:5], 0.5, mode='Union')
    boxes = boxes[pick]
    return boxes


def detect_first_stage_warpper(args):
    return detect_first_stage(*args)
