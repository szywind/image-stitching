# -*- coding: utf-8 -*-
'''
@author: Zhenyuan Shen
@email: zshen52@wisc.edu
        zhenyuanshen@microsoft.com
@date: 2021/1/5

'''
import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

root = os.path.dirname(os.path.dirname(__file__))
input_dir = os.path.join(root, 'raw')
output_dir = os.path.join(root, 'input')
sub_dir = 'set0'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
print('input_dir: ', input_dir)
print('output_dir: ', output_dir)

def check_line(lines, epsilon=1e-5):
    count_vertical, count_horizon = 0, 0
    for line in lines:
        if abs(line[1]) < epsilon: # horizon line
            count_horizon += 1
        elif abs(line[1] * 2 - np.pi) < epsilon:  # vertical line
            count_vertical += 1
    return count_vertical > 0 or count_horizon > 0

def get_lines(lines, epsilon=1e-5):
    lines_vertical = []
    lines_horizon = []
    for line in lines:
        _line = line.squeeze() # flatten()
        if abs(_line[1]) < epsilon: # horizon line
            lines_horizon.append(line)
        elif abs(_line[1] * 2 - np.pi) < epsilon:  # vertical line
            lines_vertical.append(line)
    return lines_vertical, lines_horizon

def line_detection(image, do_visualize=True):
    '''
    Standard Hough Line Detector (ref: https://www.cnblogs.com/FHC1994/p/9138315.html)
    :param image:
    :return:
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    cv2.imshow("edges", edges)
    alpha = 0.5
    num_lines = 0
    line_length = 200

    while num_lines < 5 and line_length > 100:
        line_length = int(min(gray.shape[:2]) * alpha)
        lines = cv2.HoughLines(edges, 1, np.pi/180, line_length)
        try:
            lines_vertical, lines_horizon = get_lines(lines)
            num_lines = len(lines_horizon) + len(lines_vertical)
        except:
            num_lines = 0
        alpha -= 0.01

    if do_visualize:
        for line in tqdm(lines_vertical + lines_horizon, desc='\tprocess lines', ncols=100):
            rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
            a = np.cos(theta)   #theta是弧度
            b = np.sin(theta)
            x0 = a * rho    #代表x = r * cos（theta）
            y0 = b * rho    #代表y = r * sin（theta）
            x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
            y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
            x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
            y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
            print(x1, y1, x2, y2)
            color = (0, 0, 255)
            cv2.line(image, (x1, y1), (x2, y2), color, 2)    #点的坐标必须是元组，不能是列表。
        cv2.imshow("image-lines", image)
    return lines

def line_detect_possible_demo(image, do_visualize=True):
    '''
    statistical Hough Line Detector
    :param image:
    :return:
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    line_length = int(min(gray.shape[:2]) * 0.5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=line_length, maxLineGap=5)
    if do_visualize:
        for line in tqdm(lines, desc='process lines', ncols=100):
            x1, y1, x2, y2 = line[0]
            print(line[0])
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("line_detect_possible_demo", image)
    return lines

def crop_image(src, lines, epsilon=1e-5, img_full_name=None, prefix='', do_save=True, do_split=False):
    if do_save and img_full_name == None:
        raise ValueError('invalid file name')
    img_name, img_ext = os.path.splitext(img_full_name)

    h, w = src.shape[:2]
    y, x = 0, 0
    mid = None
    for line in lines:
        _line = line[0]
        if abs(_line[1]) < epsilon: # horizontal line
            x = max(x, _line[0])
            if do_split and abs(_line[0] + 1 - w // 2) < epsilon: # TODO: hard code， 495 == w // 2
                mid = int(_line[0])
        elif abs(_line[1] * 2 - np.pi) < epsilon: # vertical line
            y = max(y, _line[0])


    if y == 0:
        y = h
    if x == 0:
        x = w
    dst = src[:int(y), :int(x)]
    if do_save:
        if do_split:
            cv2.imwrite(os.path.join(output_dir, prefix + '{}_left{}'.format(img_name, img_ext)), dst[:, :mid+1])
            cv2.imwrite(os.path.join(output_dir, prefix + '{}_right{}'.format(img_name, img_ext)), dst[:, mid+1:])
        else:
            cv2.imwrite(os.path.join(output_dir, prefix + img_full_name), dst)

def process_batch(img_list):
    for img_path in tqdm(img_list, desc='process images ...', ncols=100):
        img_full_name = os.path.basename(img_path)

        print('input image: ', img_path)
        src = cv2.imread(img_path)
        print('image shape: ', src.shape)
        cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('input_image', src)
        lines = line_detection(src.copy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        do_split = u'车轮' in img_full_name
        crop_image(src, lines, img_full_name=img_full_name, do_split=do_split)

def visualize_keypoints(img_path, pts_path):
    img = cv2.imread(img_path)
    kps = pd.read_csv(pts_path, header=None)
    cv2.imshow('input', img)
    # Radius of circle
    radius = 2
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = -1

    print(pts_path)
    for idx, kp in kps.iterrows():
        cv2.circle(img, (kp[1], kp[2]), radius, color, thickness, lineType=8, shift=0)
        cv2.putText(img, str(kp[0]), (kp[1]-10, kp[2]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255))
    cv2.imshow('result', img)
    return img

if __name__ == '__main__':
    ## data cropping
    # img_list = glob.glob(os.path.join(input_dir, '2D*.jpg'))
    # img_list_wheel = list(filter(lambda x: u'车轮' in x, img_list))
    # print('img_list: ', len(img_list), img_list[0])
    # process_batch(img_list)

    ## vizualize annotations
    input_dir, output_dir = output_dir, os.path.join(root, 'output')
    img_list = glob.glob(os.path.join(input_dir, sub_dir, '2D*.jpg'))
    # img_list = [img for img in img_list if u'2D_车头视角2' in img]

    fig_dir = os.path.join(root, 'figures', sub_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    for img_path in tqdm(img_list, desc='process image ...', ncols=100):
        img_fullname = os.path.basename(img_path)
        img_name = os.path.splitext(img_fullname)[0]
        pts_path = os.path.join(root, 'keypoints', 'labels_' + img_name + '.csv')

        output_path = os.path.join(fig_dir, 'anno_' + img_fullname)
        if not os.path.exists(pts_path):
            raise FileNotFoundError('File {} Not Found!'.format(pts_path))
        if not os.path.exists(img_path):
            raise FileNotFoundError('File {} Not Found!'.format(img_path))

        res_img = visualize_keypoints(img_path, pts_path)
        cv2.imwrite(output_path, res_img)
        cv2.waitKey(0)

    img_name = u'全景拼接画面'
    img_path = os.path.join(output_dir, img_name + '.jpg')
    pts_path = os.path.join(root, 'keypoints', 'labels_' + img_name + '.csv')
    output_path = os.path.join(fig_dir, 'anno_' + img_name + '.jpg')
    if not os.path.exists(pts_path) or not os.path.exists(img_path):
        raise FileNotFoundError('File Not Found!')
    res_img = visualize_keypoints(img_path, pts_path)
    cv2.imwrite(output_path, res_img)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()