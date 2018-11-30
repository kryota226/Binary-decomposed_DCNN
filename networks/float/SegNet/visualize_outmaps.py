# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import glob as gb

NPY_PATH = 'C:/cityscapes/segnet/BinaryCNN-without_1_1/result/bit6/basis6/out_maps'

npy_list = sorted(gb.glob(NPY_PATH + '/*'))
palette = np.array([[180, 130, 70],[70,70,70],[153,153,153],[128,64,128],
                    [125,69,120],[225,70,224],[54,140,115],[0,220,220],[35,142,107],[136,17,0],
                    [66,49,203],[35,25,110]], dtype=np.uint8)

def load_map(npy_img):
    cls_img = np.load(npy_img)
    segmented_img = segmentation(cls_img)
    img_name = os.path.splitext(os.path.basename(npy_img))[0]
    cv2.imwrite(img_name, segmented_img)

def segmentation(cls_img):
    h, w = cls_img.shape
    seg_img = np.empty((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if cls_img[i, j] == 0:
                seg_img[i, j] = palette[0]
            elif cls_img[i, j] == 1:
                seg_img[i, j] = palette[1]
            elif cls_img[i, j] == 2:
                seg_img[i, j] = palette[2]
            elif cls_img[i, j] == 3:
                seg_img[i, j] = palette[3]
            elif cls_img[i, j] == 4:
                seg_img[i, j] = palette[4]
            elif cls_img[i, j] == 5:
                seg_img[i, j] = palette[5]
            elif cls_img[i, j] == 6:
                seg_img[i, j] = palette[6]
            elif cls_img[i, j] == 7:
                seg_img[i, j] = palette[7]
            elif cls_img[i, j] == 8:
                seg_img[i, j] = palette[8]
            elif cls_img[i, j] == 9:
                seg_img[i, j] = palette[9]
            elif cls_img[i, j] == 10:
                seg_img[i, j] = palette[10]
            elif cls_img[i, j] == 11:
                seg_img[i, j] = palette[11]
    return seg_img

if __name__ == '__main__':
    for i, npy_file in enumerate(npy_list):
        load_map(npy_file)
        print('\routput_file: {0} / {1}'.format(i + 1, len(npy_list)), end='')