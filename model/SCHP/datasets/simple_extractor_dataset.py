#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   dataset.py
@Time    :   8/30/19 9:12 PM
@Desc    :   Dataset Definition
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import cv2
import numpy as np

from torch.utils import data
from utils.transforms import get_affine_transform


class SimpleFolderDataset(data.Dataset):
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        self.file_list=[]
        self.root_list=[]
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".jpg"):
                    source_file_path = os.path.join(root, file)
                    self.file_list.append(source_file_path)
                    self.root_list.append(root)

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        img_path = self.file_list[index]
        root = self.root_list[index]
        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            return self.__getitem__(index+1)
        else:
            h, w, _ = img.shape

            # Get person center and scale
            person_center, s = self._box2cs([0, 0, w - 1, h - 1])
            r = 0
            trans = get_affine_transform(person_center, s, r, self.input_size)
            input = cv2.warpAffine(
                img,
                trans,
                (int(self.input_size[1]), int(self.input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))

            input = self.transform(input)
            meta = {
                'img_path': img_path,
                'name': img_name,
                'root': root,
                'center': person_center,
                'height': h,
                'width': w,
                'scale': s,
                'rotation': r
            }

            return input, meta
