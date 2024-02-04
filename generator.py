"""
Authors : inzapp

Github url : https://github.com/inzapp/image-to-image

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 image_paths_x,
                 image_paths_y,
                 input_rows,
                 input_cols,
                 input_channels,
                 output_channels,
                 batch_size,
                 nv12=False,
                 dtype='float32'):
        self.image_paths_y = image_paths_y
        self.image_paths_x = image_paths_x
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.batch_size = batch_size
        self.nv12 = nv12
        self.dtype = dtype

        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        self.x_image_paths_of = self.get_x_image_paths_of(self.image_paths_y, self.image_paths_x)
        np.random.shuffle(self.image_paths_y)

    def load(self):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image_pair, self.next_image_path()))
        batch_x, batch_y = [], []
        for f in fs:
            img_x, img_y = f.result()
            batch_x.append(self.preprocess(img_x, image_type='x'))
            batch_y.append(self.preprocess(img_y, image_type='y'))
        batch_x = np.asarray(batch_x).astype(self.dtype)
        batch_y = np.asarray(batch_y).astype(self.dtype)
        return batch_x, batch_y

    def preprocess(self, img, image_type):
        assert image_type in ['x', 'y']
        channels = self.input_channels if image_type == 'x' else self.output_channels
        if self.nv12:
            channels = 1
            input_rows_for_nv12 = self.input_rows // 3 * 2
            img = self.resize(img, (self.input_cols, input_rows_for_nv12))
            img = self.convert_bgr2yuv420sp(img)
        else:
            img = self.resize(img, (self.input_cols, self.input_rows))
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.asarray(img).reshape((self.input_rows, self.input_cols, channels)).astype(self.dtype) / 255.0
        return x

    def postprocess(self, y):
        img = np.asarray(np.clip((y * 255.0), 0.0, 255.0)).astype('uint8')
        if self.nv12:
            input_rows_for_nv12 = self.input_rows // 3 * 2
            img = self.convert_yuv420sp2bgr(img)
            img = img.reshape((input_rows_for_nv12, self.input_cols, 3))
        else:
            if self.output_channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.reshape((self.input_rows, self.input_cols, self.output_channels))
        return img

    def key(self, image_path, image_type):
        if image_type == 'x':
            key = f'{os.path.basename(image_path)[:-11]}'
        else:
            key = f'{os.path.basename(image_path)[:-4]}'
        return key

    def get_x_image_paths_of(self, image_paths_y, image_paths_x):
        x_image_paths_of = {}
        for path in image_paths_y:
            x_image_paths_of[self.key(path, 'y')] = []
        for path in image_paths_x:
            x_image_paths_of[self.key(path, 'x')].append(path)
        return x_image_paths_of

    def next_image_path(self):
        path = self.image_paths_y[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths_y):
            self.img_index = 0
            np.random.shuffle(self.image_paths_y)
        return path

    def resize(self, img, size=(-1, -1), scale=1.0):
        interpolation = None
        img_height, img_width = img.shape[:2]
        if scale != 1.0:
            if scale > 1.0:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
        else:
            if size[0] > img_width or size[1] > img_height:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, size, interpolation=interpolation)

    def convert_bgr2yuv420sp(self, img, yuv_type='nv12'):
        assert yuv_type in ['nv12', 'nv21']
        h, w = img.shape[:2]
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_YV12)
        y = yuv[:h]
        uv = yuv[h:]
        uvf = uv.flatten()
        v = uvf[:int(uvf.shape[0] / 2)].reshape(uv.shape[0], -1)
        u = uvf[int(uvf.shape[0] / 2):].reshape(uv.shape[0], -1)
        new_uv_or_vu = np.zeros(uv.shape, dtype=uv.dtype)
        if yuv_type == 'nv12':
            new_uv_or_vu[:,::2] = u
            new_uv_or_vu[:,1::2] = v
        elif yuv_type == 'nv21':
            new_uv_or_vu[:,::2] = v
            new_uv_or_vu[:,1::2] = u
        return np.vstack((y, new_uv_or_vu))

    def convert_yuv420sp2bgr(self, img, yuv_type='nv12'):
        assert yuv_type in ['nv12', 'nv21']
        if yuv_type == 'nv12':
            conversion_type = cv2.COLOR_YUV2BGR_NV12
        else:
            conversion_type = cv2.COLOR_YUV2BGR_NV21
        return cv2.cvtColor(img, conversion_type)

    def get_x_path_of(self, path_y):
        return np.random.choice(self.x_image_paths_of[self.key(path_y, 'y')])

    def load_image(self, path, image_type):
        assert image_type in ['x', 'y']
        data = np.fromfile(path, dtype=np.uint8)
        if self.nv12:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            channels = self.input_channels if image_type == 'x' else self.output_channels
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
        return img

    def load_image_x(self, path_y):
        return self.load_image(self.get_x_path_of(path_y), image_type='x')

    def load_image_y(self, path_y):
        return self.load_image(path_y, image_type='y')

    def load_image_pair(self, path_y):
        img_x = self.load_image_x(path_y)
        img_y = self.load_image_y(path_y)
        return img_x, img_y

