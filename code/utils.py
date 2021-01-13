# -*- coding: utf-8 -*-
'''
@author: Zhenyuan Shen
@email: zshen52@wisc.edu 
        zhenyuanshen@microsoft.com
@date: 2021/1/12

'''

import cv2
from collections import defaultdict

def bfs(mask):
    assert mask.ndim == 2
    h, w = mask.shape
    queue = []
    for i in range(h):
        if mask[i, 0]:
            mask[i, 0] = 0
            queue.append((i, 0))
        if mask[i, w-1]:
            mask[i, w-1] = 0
            queue.append((i, w-1))

    for j in range(w):
        if mask[0, j]:
            mask[0, j] = 0
            queue.append((0, j))

        if mask[h-1, j]:
            mask[h-1, j] = 0
            queue.append((h-1, j))

    dirs = [-1, 0, 1, 0, -1]
    print(len(queue))
    while queue:
        s = queue.pop(0)
        for d in range(4):
            row = s[0] + dirs[d]
            col = s[1] + dirs[d+1]
            if row >= 0 and row < h and col >= 0 and col < w and mask[row, col]:
                mask[row, col] = 0
                queue.append((row, col))

def max_filter_1d(img, radius):
    h, w = img.shape
    dst = img.copy()
    padded = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_CONSTANT, cv2.Scalar(0));

    for r in range(h):
        for c in range(w):
            highest = 0;
            for i in range(-radius, radius+1, 1):
                for j in range(-radius, radius+1, 1):
                    val = padded[radius + r + i, radius + c + j]
                    highest = max(highest, val)

            dst[r, c] = highest
    return dst