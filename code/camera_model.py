# -*- coding: utf-8 -*-
'''
@author: Zhenyuan Shen
@email: zshen52@wisc.edu 
        zhenyuanshen@microsoft.com
@date: 2021/1/6

'''
import os
import cv2
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

root = os.path.dirname(os.path.dirname(__file__))
pts_dir = os.path.join(root, 'keypoints')
input_dir = os.path.join(root, 'input')
output_dir = os.path.join(root, 'figures')
sub_dir = 'set0'

def evaluate_homography_matrix(im_src, pts_src, im_dst, pts_dst):
    '''
    :param im_src: source image
    :param pts_src: key points in the source image
    :param im_dst: destination image
    :param pts_dst: key points in the destination image
    :return:
    '''
    h, status = cv2.findHomography(pts_src, pts_dst)
    print('homography status', status.shape, status.flatten())
    ''' 
    The calculated homography can be used to warp 
    the source image to destination. Size is the 
    size (width, height) of im_dst
    '''
    size = (im_dst.shape[1], im_dst.shape[0])
    im_wrap = cv2.warpPerspective(im_src, h, size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return im_wrap

if __name__ == '__main__':
    img_fullname_dst = u'全景拼接画面.jpg'
    img_dst = cv2.imread(os.path.join(root, 'output', img_fullname_dst))
    img_name_dst = os.path.splitext(img_fullname_dst)[0]
    df_kps_dst = pd.read_csv(os.path.join(pts_dir, 'labels_{}.csv'.format(img_name_dst)), header=None)
    df_kps_dst.sort_values(by=0, inplace=True)
    assert len(set(df_kps_dst[0].tolist())) == len(df_kps_dst[0].tolist()) # check uniqueness

    img_list = glob.glob(os.path.join(input_dir, sub_dir, '2D*.jpg'))
    # img_list = [img for img in img_list if u'2D_车头视角2' in img]
    for img_path in tqdm(img_list, desc='process image ...', ncols=100):
        img_fullname_src = os.path.basename(img_path)
        img_src = cv2.imread(img_path)
        img_name_src = os.path.splitext(img_fullname_src)[0]
        df_kps_src = pd.read_csv(os.path.join(pts_dir, 'labels_{}.csv'.format(img_name_src)), header=None)
        df_kps_src.sort_values(by=0, inplace=True)
        assert len(set(df_kps_src[0].tolist())) == len(df_kps_src[0].tolist()) # check uniqueness


        # check completeness
        mask = df_kps_dst[0].isin(df_kps_src[0])
        assert mask.sum() == len(df_kps_src)

        kps_dst = df_kps_dst.loc[mask, 1:2].to_numpy()
        kps_src = df_kps_src.loc[:, 1:2].to_numpy()

        # if u'2D_后视角' in img_name_src:
        #     # flip image and kps
        #     img_src = cv2.flip(img_src, 1)
        #     h, w = img_src.shape[:2]
        #     kps_src[:, 0] = w - 1 - kps_src[:, 0]

        res_img = evaluate_homography_matrix(img_src, kps_src, img_dst, kps_dst)

        ## TODO: use line detection later
        TOP, BOTTOM, LEFT, RIGHT = 350, 1050, 340, 650
        if u'2D_后视角' in img_name_src:
            res_img[:BOTTOM, :] = 0
        if u'2D_车头视角' in img_name_src:
            res_img[TOP:, :] = 0

        if '_left' in img_name_src:
            res_img[:, LEFT:] = 0
        if '_right' in img_name_src:
            res_img[:, :RIGHT] = 0

        cv2.imshow('result', res_img)
        cv2.imwrite(os.path.join(output_dir, sub_dir, 'wrap_' + img_fullname_src), res_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

