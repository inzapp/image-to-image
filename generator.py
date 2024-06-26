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
                 input_shape,
                 output_shape,
                 batch_size,
                 nv12,
                 g_model=None,
                 training=False):
        self.image_paths_y = image_paths_y
        self.image_paths_x = image_paths_x
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.nv12 = nv12
        self.g_model = g_model
        self.training = training

        self.img_index = 0
        self.pool = ThreadPoolExecutor(8)
        self.x_image_paths_of = self.get_x_image_paths_of(self.image_paths_y, self.image_paths_x)
        np.random.shuffle(self.image_paths_y)

    def load(self, use_adversarial_loss):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image_pair, self.next_image_path()))
        batch_x, batch_y = [], []
        for f in fs:
            img_x, img_y = f.result()
            batch_x.append(self.preprocess(img_x, image_type='x'))
            batch_y.append(self.preprocess(img_y, image_type='y'))
        batch_x = np.asarray(batch_x).astype(np.float32)
        batch_y = np.asarray(batch_y).astype(np.float32)

        dx, dy, gx, gy = None, None, None, None
        if use_adversarial_loss:
            from image_to_image import ImageToImage
            real_dx = batch_y[:self.half_batch_size]
            fake_dx = np.asarray(ImageToImage.graph_forward(self.g_model, batch_x[:self.half_batch_size]))
            dx = np.concatenate([real_dx, fake_dx], axis=0)
            real_dy = np.ones((self.half_batch_size, 1))
            fake_dy = np.zeros((self.half_batch_size, 1))
            dy = np.concatenate([real_dy, fake_dy])
            gx = batch_x
            gy = np.ones((self.batch_size, 1))

            dx = np.asarray(dx).reshape((self.batch_size,) + self.output_shape).astype(np.float32)
            dy = np.asarray(dy).reshape((self.batch_size, 1)).astype(np.float32)
            gx = np.asarray(gx).reshape((self.batch_size,) + self.input_shape).astype(np.float32)
            gy = np.asarray(gy).reshape((self.batch_size, 1)).astype(np.float32)
        return batch_x, batch_y, dx, dy, gx, gy

    def preprocess(self, img, image_type):
        assert image_type in ['x', 'y']
        target_shape = self.input_shape if image_type == 'x' else self.output_shape
        channels = target_shape[-1]
        if self.nv12:
            channels = 1
            target_rows_for_nv12 = target_shape[0] // 3 * 2
            img = self.resize(img, (target_shape[1], target_rows_for_nv12))
            img = self.convert_bgr2yuv420sp(img)
        else:
            img = self.resize(img, (target_shape[1], target_shape[0]))
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.asarray(img).reshape(target_shape).astype(np.float32) / 255.0
        return x

    def postprocess(self, y):
        img = np.asarray(np.clip((y * 255.0), 0.0, 255.0)).astype('uint8')
        if self.nv12:
            output_rows_for_nv12 = self.output_shape[0] // 3 * 2
            img = self.convert_yuv420sp2bgr(img)
            img = img.reshape((output_rows_for_nv12, self.output_shape[1], 3))
        else:
            if self.output_shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.reshape(self.output_shape)
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
        for path in image_paths_y:
            if len(x_image_paths_of[self.key(path, 'y')]) == 0:
                x_image_paths_of[self.key(path, 'y')].append(path)
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
        img_h, img_w = img.shape[:2]
        interpolation_upscaling = cv2.INTER_CUBIC
        interpolation_downscaling = cv2.INTER_CUBIC
        if scale != 1.0:
            if scale > 1.0:
                interpolation = interpolation_upscaling
            else:
                interpolation = interpolation_downscaling
            return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
        else:
            if size[0] > img_w or size[1] > img_h:
                interpolation = interpolation_upscaling
            else:
                interpolation = interpolation_downscaling
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
        x_paths = self.x_image_paths_of[self.key(path_y, 'y')]
        return np.random.choice(x_paths) if self.training else x_paths[0]

    def load_image(self, path, image_type):
        assert image_type in ['x', 'y']
        data = np.fromfile(path, dtype=np.uint8)
        if self.nv12:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            channels = self.input_shape[-1] if image_type == 'x' else self.output_shape[-1]
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
        return img

    def load_image_x(self, path_y, no_x=False):
        if no_x:
            img = self.load_image(path_y, image_type='x')
        else:
            img = self.load_image(self.get_x_path_of(path_y), image_type='x')
        return img

    def load_image_y(self, path_y):
        return self.load_image(path_y, image_type='y')

    def load_image_pair(self, path_y):
        img_x = self.load_image_x(path_y)
        img_y = self.load_image_y(path_y)
        return img_x, img_y

