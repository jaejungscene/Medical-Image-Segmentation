from copy import deepcopy


def fineNotMatch(list1: list, list2: list) -> list:
    """
    return: list of elements that dosen't match
    """
    if len(list1)>len(list2):
        Llist, Slist = list1, list2
    else:
        Llist, Slist = list2, list1

    flag = False
    result = []
    for elem in Llist:
        for i in range(len(Slist)):
            if Slist[i] == elem:
                flag = True
                del Slist[i]
                break
        if flag == False:
            result.append(elem)
        flag = False

    return result


import numpy as np
from typing import *
import cv2
from skimage import io, color
import matplotlib.pyplot as plt

def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    # color: Tuple[int, int, int] = (255,0,0),
    alpha: float=0.5,
    resize: Tuple[int, int]=(512,512)
) -> np.ndarray:
    """
    return the image with segmentation masks 
    """
    if len(image.shape) == 2:   # gray color
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # if image.shape[-1] == min(image.shape):
    #     image = image.transpose(2,0,1)

    return color.label2rgb(label=mask, image=image, alpha=alpha)
    # color = np.asarray(color).reshape(3,1,1)
    # colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    # masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    # image_overlay = masked.filled()

    # if resize is not None:
    #     image = cv2.resize(image.transpose(1,2,0), resize)
    #     image_overlay = cv2.resize(image_overlay.transpose(1,2,0), resize)
    
    # image_combined = cv2.addWeighted(image, 1-alpha, image_overlay, alpha, 0)
    # return image_combined
        