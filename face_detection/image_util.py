# coding: utf-8
"""
face detection 内での画像IOの定義
"""
import cv2


def load_image(path):
    """
    画像の読み込み
    :param str path:　
    :return:
    """
    return cv2.imread(path)


def save_image(img, path):
    """
    画像の保存
    :param img:
    :param str path:
    :return:
    """
    return cv2.imwrite(path, img)
