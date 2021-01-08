# -*- coding: utf-8 -*-
'''
@author: Zhenyuan Shen
@email: zshen52@wisc.edu 
        zhenyuanshen@microsoft.com
@date: 2021/1/7

'''

import os
import cv2
import glob
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
# plt.set_cmap('binary')

root = os.path.dirname(os.path.dirname(__file__))
input_dir = os.path.join(root, 'figures')
output_dir = input_dir

def maximum(img_list):
    return np.max(img_list, axis=0)

def average(img_list):
    # Calculate blended image
    assert len(img_list) > 0
    h, w = img_list[0].shape[:2]
    mask = np.zeros((h, w))

    # dst = np.sum(img_list, axis=0)
    # for i in range(h):
    #     for j in range(w):
    #         if mask[i, j] == 0: continue
    #         dst[i, j, :] /= mask[i, j]

    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_i = gray > 0
        mask_i = mask_i.astype(np.float)
        # plt.imshow(mask_i * 255)
        # plt.show()
        mask += mask_i

    dst = np.sum(img_list, axis=0)
    print('dst shape: ', dst.shape)
    print('mask shape: ', mask.shape)

    # for i in range(h):
    #     for j in range(w):
    #         if mask[i, j] == 0: continue
    #         dst[i, j, :] /= mask[i, j]
    mask = np.clip(mask, 1, len(img_list))

    plt.imshow(mask * 255)
    plt.show()

    dst = dst / mask[..., np.newaxis]
    dst = dst.astype(np.uint8)
    return dst

def sift_blending():
    left = cv2.imread(os.path.join(input_dir, 'wrap_2D_前广角镜头.jpg')) # 'wrap_2D_车头视角2.jpg'
    right = cv2.imread(os.path.join(input_dir, 'wrap_2D_车轮视角2_right.jpg'))

    images = [left, right]
    stitcher = cv2.Stitcher.create()
    ret, pano = stitcher.stitch(images)

    if ret == cv2.STITCHER_OK:
        cv2.imshow('Panoroma', pano)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print("Failed to stitch.")

if __name__ == '__main__':
    img_path_list = glob.glob(os.path.join(input_dir, 'wrap_2D_*.jpg'))
    img_list = [cv2.imread(img_path) for img_path in img_path_list if os.path.exists(img_path)]
    for img in img_list:
        print(img.shape)
    res_img = maximum(img_list)

    cv2.imshow('result', res_img)
    cv2.imwrite(os.path.join(output_dir, 'combined_avg_{}.jpg'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))),
                res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # sift_blending()