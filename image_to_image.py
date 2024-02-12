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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import random
import warnings
import numpy as np
import shutil as sh
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self,
                 train_image_path,
                 validation_image_path,
                 input_shape,
                 output_shape,
                 model_name,
                 lr,
                 warm_up,
                 batch_size,
                 unet_depth,
                 iterations,
                 save_interval,
                 pretrained_model_path='',
                 nv12=False,
                 training_view=False):
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_name = model_name
        self.lr = lr
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.unet_depth = unet_depth
        self.iterations = iterations
        self.save_interval = save_interval
        self.pretrained_model_path = pretrained_model_path
        self.nv12 = nv12
        self.training_view = training_view


class ImageToImage(CheckpointManager):
    def __init__(self, config, training):
        super().__init__()
        assert config.save_interval >= 1000
        assert config.input_shape[0] % 32 == 0
        assert config.input_shape[1] % 32 == 0
        assert config.input_shape[2] in [1, 3]
        assert config.output_shape[0] % 32 == 0
        assert config.output_shape[1] % 32 == 0
        assert config.output_shape[2] in [1, 3]
        if config.nv12:
            assert config.input_shape[-1] == 1, 'input channels must be 1 if use nv12 format'
            assert config.output_shape[-1] == 1, 'output channels must be 1 if use nv12 format'
        self.train_image_path = config.train_image_path
        self.validation_image_path = config.validation_image_path
        self.input_shape = config.input_shape
        self.output_shape = config.output_shape
        self.model_name = config.model_name
        self.lr = config.lr
        self.warm_up = config.warm_up
        self.batch_size = config.batch_size
        self.unet_depth = config.unet_depth
        self.iterations = config.iterations
        self.save_interval = config.save_interval
        self.pretrained_model_path = config.pretrained_model_path
        self.nv12 = config.nv12
        self.training_view = config.training_view

        warnings.filterwarnings(action='ignore')
        self.set_model_name(config.model_name)
        self.live_view_previous_time = time()
        if not training:
            self.set_global_seed()

        if not self.is_valid_path(self.train_image_path):
            print(f'train image path is not valid : {self.train_image_path}')
            exit(0)

        if not self.is_valid_path(self.validation_image_path):
            print(f'validation image path is not valid : {self.validation_image_path}')
            exit(0)

        self.train_image_paths_x, self.train_image_paths_y = self.init_image_paths(self.train_image_path)
        self.validation_image_paths_x, self.validation_image_paths_y = self.init_image_paths(self.validation_image_path)

        self.pretrained_iteration_count = 0
        if self.pretrained_model_path != '':
            if not (os.path.exists(self.pretrained_model_path) and os.path.isfile(self.pretrained_model_path)):
                print(f'file not found : {model_path}')
                exit(0)
            self.model = tf.keras.models.load_model(self.pretrained_model_path, compile=False, custom_objects={'tf': tf})
            self.input_shape = self.model.input_shape[1:]
            self.output_shape = self.model.output_shape[1:]
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(self.pretrained_model_path)
            self.nv12 = self.pretrained_model_path.find('nv12') > -1
        else:
            self.model = Model(
                input_shape=self.input_shape,
                output_shape=self.output_shape).build(unet_depth=self.unet_depth)

        self.train_data_generator = DataGenerator(
            image_paths_x=self.train_image_paths_x,
            image_paths_y=self.train_image_paths_y,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            batch_size=self.batch_size,
            nv12=self.nv12,
            training=True)
        self.validation_data_generator = DataGenerator(
            image_paths_x=self.validation_image_paths_x,
            image_paths_y=self.validation_image_paths_y,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            batch_size=1,
            nv12=self.nv12)

    def set_global_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def is_valid_path(self, path):
        return os.path.exists(path) and os.path.isdir(path)

    def exit_if_no_images(self, image_paths, path):
        if len(image_paths) == 0:
            print(f'no images found in {path}')
            exit(0)

    def init_image_paths(self, image_path):
        from x_data_generator import X_KEYWORD
        paths_all = glob(f'{image_path}/**/*.jpg', recursive=True)
        paths_x, paths_y = [], []
        for path in paths_all:
            if os.path.basename(path).find(f'_{X_KEYWORD}_') > -1:
                paths_x.append(path)
            else:
                paths_y.append(path)
        return paths_x, paths_y

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def concat(self, images):
        need_gray_to_bgr = self.input_shape[-1] != self.output_shape[-1]
        for i in range(len(images)):
            if len(images[i].shape) == 2:
                images[i] = images[i].reshape(images[i].shape + (1,))
            if need_gray_to_bgr and images[i].shape[-1] == 1:
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
            image_h, image_w = images[i].shape[:2]
            if image_h != self.output_shape[0] and image_w != self.output_shape[1]:
                images[i] = self.train_data_generator.resize(images[i], (self.output_shape[1], self.output_shape[0]))
            if len(images[i].shape) == 2:
                images[i] = images[i].reshape(images[i].shape + (1,))
        return np.concatenate(images, axis=1)

    def predict(self, img_x):
        x = self.train_data_generator.preprocess(img_x, image_type='x')
        x = x.reshape((1,) + x.shape)
        img_pred = self.train_data_generator.postprocess(np.array(self.graph_forward(self.model, x)[0]))
        return img_pred

    def predict_video(self, video_path):
        if not (os.path.exists(video_path) and os.path.isfile(video_path)):
            print(f'video not found. video video_path : {video_path}')
            exit(0)
        cap = cv2.VideoCapture(video_path)
        while True:
            frame_exist, bgr = cap.read()
            if not frame_exist:
                print('frame not exists')
                break
            bgr = self.train_data_generator.resize(bgr, (self.user_input_shape[1], self.user_input_shape[0]))
            img_x = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if self.input_type == 'gray' else bgr
            img_pred = self.predict(img_x)
            img_concat = self.concat([img_x, img_pred])
            img_concat = self.train_data_generator.resize(img_concat, scale=0.5)
            cv2.imshow('img_pred', img_concat)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cap.release()
        cv2.destroyAllWindows()

    def predict_rtsp(self, rtsp_url):
        import threading
        from time import sleep
        def read_frames(rtsp_url, frame_queue, end_flag, lock):
            cap = cv2.VideoCapture(rtsp_url)
            while True:
                with lock:
                    frame_exist, bgr = cap.read()
                    if not frame_exist:
                        break
                    if len(frame_queue) == 0:
                        frame_queue.append(bgr)
                    else:
                        frame_queue[0] = bgr
                    print(f'[read_frames] frame updated')
                sleep(0)
            end_flag[0] = True
            cap.release()

        lock, frame_queue, end_flag = threading.Lock(), [], [False]
        read_thread = threading.Thread(target=read_frames, args=(rtsp_url, frame_queue, end_flag, lock))
        read_thread.daemon = True
        read_thread.start()
        while True:
            if end_flag[0]:
                print(f'[main] end flag is True')
                break
            bgr = None
            with lock:
                if frame_queue:
                    bgr = frame_queue[0].copy()
            if bgr is None:
                print(f'[main] bgr is None')
                sleep(0.1)
                continue

            print(f'[main] frame get success')
            bgr = self.train_data_generator.resize(bgr, (self.user_input_shape[1], self.user_input_shape[0]))
            img_x = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if self.input_type == 'gray' else bgr
            img_pred = self.predict(img_x)
            img_concat = self.concat([img_x, img_pred])
            img_concat = self.train_data_generator.resize(img_concat, scale=0.5)
            cv2.imshow('img_pred', img_concat)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cv2.destroyAllWindows()

    def print_loss(self, progress_str, loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' loss : {loss:>8.4f}'
        print(loss_str, end='')

    def psnr(self, img_a, img_b, max_val):
        mse = np.mean((img_a.astype('float32') - img_b.astype('float32')) ** 2.0)
        return 20 * np.log10(max_val / np.sqrt(mse)) if mse!= 0.0 else 100.0

    def evaluate(self, dataset='validation', image_path='', show=False, save_count=0, no_x=False):
        image_paths_y, image_paths_x = [], []
        if image_path != '':
            if not os.path.exists(image_path):
                print(f'image path not found : {image_path}')
                return
            if os.path.isdir(image_path):
                image_paths_x, image_paths_y = self.init_image_paths(image_path)
            else:
                image_paths_x, image_paths_y = [], [image_path]
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths_x = self.train_image_paths_x
                image_paths_y = self.train_image_paths_y
            else:
                image_paths_x = self.validation_image_paths_x
                image_paths_y = self.validation_image_paths_y

        if no_x:
            image_paths_y += image_paths_x
            image_paths_x = []

        if len(image_paths_y) == 0:
            print(f'no images found')
            return

        data_generator = DataGenerator(
            image_paths_x=image_paths_x,
            image_paths_y=image_paths_y,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            batch_size=self.batch_size,
            nv12=self.nv12)

        if save_count > 0:
            save_path = 'result_images'
            os.makedirs(save_path, exist_ok=True)

        print()
        count = 0
        psnr_sum = 0.0
        ssim_sum = 0.0
        show_or_save_images = show or save_count > 0
        paths = image_paths_y if show_or_save_images else tqdm(image_paths_y)
        for path in paths:
            img_x = data_generator.load_image_x(path, no_x=no_x)
            img_y = data_generator.load_image_y(path)
            img_pred = self.predict(img_x)
            img_pred_h, img_pred_w = img_pred.shape[:2]
            img_y = data_generator.resize(img_y, (img_pred_w, img_pred_h))
            img_y = img_y.reshape((img_pred_h, img_pred_w, -1))
            img_pred = img_pred.reshape((img_pred_h, img_pred_w, -1))
            if show_or_save_images:
                img_x = data_generator.resize(img_x, (img_pred_w, img_pred_h))
                img_x = img_x.reshape((img_pred_h, img_pred_w, -1))
                img_concat = self.concat([img_x, img_pred])
                if show:
                    cv2.imshow('img', img_concat)
                    key = cv2.waitKey(0)
                    if key == 27:
                        exit(0)
                if save_count > 0:
                    basename = os.path.basename(path)
                    save_img_path = f'{save_path}/{basename}'
                    cv2.imwrite(save_img_path, img_concat, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    count += 1
                    print(f'[{count} / {save_count}] save success : {save_img_path}')
                    if count == save_count:
                        exit(0)
            psnr = self.psnr(img_y, img_pred, 255.0)
            ssim = tf.image.ssim(img_y, img_pred, 255.0)
            psnr_sum += psnr
            ssim_sum += ssim
        if not show_or_save_images:
            avg_psnr = psnr_sum / float(len(image_paths_y))
            avg_ssim = ssim_sum / float(len(image_paths_y))
            print(f'psnr : {avg_psnr:.2f}, ssim : {avg_ssim:.4f}\n')
        return avg_psnr, avg_ssim

    def train(self):
        self.exit_if_no_images(self.train_image_paths_x, self.train_image_path)
        self.exit_if_no_images(self.train_image_paths_y, self.train_image_path)
        self.exit_if_no_images(self.validation_image_paths_x, self.validation_image_path)
        self.exit_if_no_images(self.validation_image_paths_y, self.validation_image_path)
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths_y)} gt, {len(self.train_image_paths_x)} input samples.')
        print('start training')
        nv12_content = '_nv12' if self.nv12 else ''
        self.init_checkpoint_dir()
        iteration_count = self.pretrained_iteration_count
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy='step')
        eta_calculator = ETACalculator(iterations=self.iterations)
        eta_calculator.start()
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(optimizer, iteration_count)
            loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss)
            if self.training_view:
                self.training_view_function()
            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count, content=nv12_content)
            if iteration_count % self.save_interval == 0:
                psnr, ssim = self.evaluate()
                content = f'_psnr_{psnr:.2f}_ssim_{ssim:.4f}{nv12_content}'
                self.save_best_model(self.model, iteration_count, content=content, metric=psnr)
            if iteration_count == self.iterations:
                print('train end successfully')
                return

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            path = np.random.choice(self.validation_image_paths_y)
            img = self.validation_data_generator.load_image_x(path)
            img_pred = self.predict(img)
            img_concat = self.concat([img, img_pred])
            cv2.imshow('training view', img_concat)
            cv2.waitKey(1)

