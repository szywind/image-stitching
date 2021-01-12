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
import enum
import numpy as np
from tqdm import tqdm

from datetime import datetime
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt
# plt.set_cmap('binary')

from utils import bfs

root = os.path.dirname(os.path.dirname(__file__))
input_dir = os.path.join(root, 'figures')
output_dir = input_dir
sub_dir = 'set0'
thresh = 10

class BLENDER_TYPE(enum.Enum):
    """
    - MAX:
    - AVG:
    - FEATHER:
    - MULTIBAND:
    """

    MAX = 'max'
    AVG = 'avg'
    FEATHER = 'feather'
    MULTIBAND = 'multiband'

def get_mask(img_list):
    h, w = img_list[0].shape[:2]
    mask = np.zeros((h, w))

    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray: ', gray)
        mask_i = gray > thresh
        mask_i = mask_i.astype(np.float)
        # plt.imshow(mask_i * 255)
        # plt.show()
        mask += mask_i
    return mask.astype(np.uint8)

def maximum(img_list):
    return np.max(img_list, axis=0)

def average(img_list):
    # Calculate blended image
    assert len(img_list) > 0
    dst = np.sum(img_list, axis=0)
    print('dst shape: ', dst.shape)

    mask = get_mask(img_list)
    mask = np.clip(mask, 1, len(img_list))
    print('mask shape: ', mask.shape)

    # kernel = np.ones((5, 5), np.float32) / 25
    # mask = cv2.filter2D(mask, -1, kernel)

    # mask = cv2.GaussianBlur(mask, (5, 5), 1, sigmaY=1)
    mask_3d = mask[..., np.newaxis]

    dst = dst / mask_3d
    dst = dst.astype(np.uint8)

    return dst

def sift_blending():
    left = cv2.imread(os.path.join(input_dir, sub_dir, 'wrap_2D_前广角镜头.jpg')) # 'wrap_2D_车头视角2.jpg'
    right = cv2.imread(os.path.join(input_dir, sub_dir, 'wrap_2D_车轮视角2_right.jpg'))

    images = [left, right]
    stitcher = cv2.Stitcher.create()
    ret, pano = stitcher.stitch(images)

    if ret == cv2.STITCHER_OK:
        cv2.imshow('Panoroma', pano)
        cv2.destroyAllWindows()
    else:
        print("Failed to stitch.")

def feather_blending(img_list):
    if len(img_list) == 0:
        raise ValueError('Error: Input empty image list!')
    blender = cv2.detail_FeatherBlender()
    blender.setSharpness(0.02)
    h, w = img_list[0].shape[:2]
    blender.prepare((0, 0, w, h)) # called once at start

    for img in tqdm(img_list, desc='process image ...', ncols=100):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > thresh
        blender.feed(img.astype(np.int16), mask.astype(np.uint8), (0,0))

    res, res_mask = blender.blend(None, None)
    return res.astype(np.uint8)

def multiband_blending(img_list):
    if len(img_list) == 0:
        raise ValueError('Error: Input empty image list!')
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(4)
    h, w = img_list[0].shape[:2]
    blender.prepare((0, 0, w, h)) # called once at start

    for img in tqdm(img_list, desc='process image ...', ncols=100):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray > thresh) * 255
        blender.feed(img.astype(np.int16), mask.astype(np.uint8), (0,0))

    res, res_mask = blender.blend(None, None)
    return res.astype(np.uint8)

def image_inpainting(img, do_refine_mask=True, num_iter=1, do_smooth=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < thresh).astype(np.uint8)

    if do_refine_mask:
        TOP, BOTTOM, LEFT, RIGHT = 350, 1050, 340, 650
        mask[:, LEFT:RIGHT] = 0
        bfs(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.dilate(mask, kernel, iterations=10)
        # mask = mask * ((gray < 50) * (gray > 0))
        mask = mask * (gray < 50)
        mask[:, LEFT:RIGHT] = 0


    # cv2.imshow('initial', gray)
    cv2.imshow("mask", mask * 255)
    # cv2.imshow("bar0", gray + mask * 255) # wrong
    cv2.imshow("bar", (1 - mask) * gray + mask * 255)

    mask0 = mask.copy()
    res = img.copy()
    for i in range(num_iter):
        ## image inpainting by opencv
        # res = cv2.inpaint(res, mask, 10, cv2.INPAINT_NS)
        res = cv2.inpaint(res, mask, 3, cv2.INPAINT_TELEA)
        # cv2.xphoto.inpaint(res, (mask == 0).astype(np.uint8), res, cv2.xphoto.INPAINT_FSR_FAST)
        # cv2.xphoto.inpaint(res, (mask == 0).astype(np.uint8), res, cv2.xphoto.INPAINT_FSR_BEST)

        ## update mask
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray {}'.format(i), gray)
        mask = mask * (gray < thresh)
        # cv2.imshow("mask {}".format(i), mask*255)

    cv2.imshow("0", res)

    # TODO: filtered edges
    if do_smooth:
        mask = mask0
        # kernel = np.ones((5,5), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,5))
        # mask = cv2.dilate(mask, kernel, iterations=10)
        # mask = mask * ((gray < 60) * (gray > 0))
        # cv2.imshow("gray", gray)
        # cv2.imshow("mask", mask * 255)
        # cv2.imshow("bar", (1 - mask) * gray + mask * 255)

        mask_3d = mask[..., np.newaxis]

        ## max filter
        filtered = Image.fromarray(res).filter(ImageFilter.MaxFilter(size=3))
        # filtered = filtered.filter(ImageFilter.MaxFilter(size=3))
        # img_copy = cv2.cvtColor(np.asarray(img_copy), cv2.COLOR_RGB2BGR)
        filtered = np.asarray(filtered)

        # filtered = cv2.GaussianBlur(res, (3,7), 1)
        # filtered = cv2.medianBlur(filtered, 3)
        # filtered = cv2.bilateralFilter(res, 3, 75, 75)

        res = (1 - mask_3d) * res + mask_3d * filtered
        cv2.imshow("1", res)
    return res

def run(method, img_list, do_save=False, do_inpaint=True):
    if method == BLENDER_TYPE.MAX:
        res_img = maximum(img_list)
    elif method == BLENDER_TYPE.AVG:
        res_img = average(img_list)
    elif method == BLENDER_TYPE.FEATHER:
        res_img = feather_blending(img_list)
    elif method == BLENDER_TYPE.MULTIBAND:
        res_img = multiband_blending(img_list)
    else:
        raise ValueError('Unsupported blending moethod.')
    cv2.imshow('result_{}'.format(method.value), res_img)

    if do_inpaint:
        res_img = image_inpainting(res_img)
        cv2.imshow('after impainting: result_{}'.format(method.value), res_img)

    if do_save:
        cv2.imwrite(os.path.join(output_dir, 'combined_{}_{}_{}.jpg'.format(sub_dir, method.value, datetime.now().strftime("%d-%m-%Y %H-%M-%S"))), res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_path_list = glob.glob(os.path.join(input_dir, sub_dir, 'wrap_2D_*.jpg'))
    img_list = [cv2.imread(img_path) for img_path in img_path_list if os.path.exists(img_path)]
    for img in img_list:
        print(img.shape)

    run(BLENDER_TYPE.MAX, img_list)
    run(BLENDER_TYPE.AVG, img_list)
    run(BLENDER_TYPE.FEATHER, img_list)
    run(BLENDER_TYPE.MULTIBAND, img_list)

    # sift_blending()
