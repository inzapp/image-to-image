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
import argparse
import numpy as np
import shutil as sh
import albumentations as A

from glob import glob
from tqdm import tqdm


X_KEYWORD = 'I2IX'


class XDataGenerator:
    def __init__(self, generate_count_per_image):
        self.use_noise = True
        self.use_blur = True
        self.use_jpeg_compression = True

        self.random_noise_max_range = 40
        self.transform_jpeg_compression = A.Compose([A.ImageCompression(quality_lower=50, quality_upper=75, always_apply=True)])
        self.generate_count_per_image = generate_count_per_image
        self.x_functions = []
        if self.use_noise:
            self.x_functions.append(self.add_noise)
        if self.use_blur:
            self.x_functions.append(self.random_blur)
        self.max_x_pair_count = 10  # fixed count for data loader, do not change
        self.x_index_candidates = list(map(str, list(range(self.max_x_pair_count))))

    def init_image_paths(self, path):
        x_paths, y_paths = [], []
        for path in glob(f'{path}/**/*.jpg', recursive=True):
            if os.path.basename(path).find(f'_{X_KEYWORD}_') > -1:
                x_paths.append(path)
            else:
                y_paths.append(path)
        return x_paths, y_paths

    def add_noise(self, img):
        img = np.array(img).astype('float32')
        noise_power = np.random.uniform() * self.random_noise_max_range
        img_h, img_w, channels = img.shape
        img += np.random.uniform(-noise_power, noise_power, size=(img_h, img_w, channels))
        img = np.clip(img, 0.0, 255.0).astype('uint8')
        return img

    def random_blur(self, img):
        if np.random.uniform() < 0.5:
            img = cv2.blur(img, (2, 2))
        else:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def jpeg_compression(self, img):
        return self.transform_jpeg_compression(image=img)['image']

    def generate_x_image(self, img):
        img = np.random.choice(self.x_functions)(img)
        if self.use_jpeg_compression and np.random.uniform() < 0.5:
            img = self.jpeg_compression(img)
        return img

    def show(self, path):
        _, y_paths = self.init_image_paths(path)
        for path in y_paths:
            print(path)
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_x = self.generate_x_image(img)
            img_cat = np.concatenate([img, img_x], axis=1)
            cv2.imshow('x image sample', img_cat)
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)

    def generate(self, path):
        assert 1 <= self.generate_count_per_image <= self.max_x_pair_count
        _, y_paths = self.init_image_paths(path)
        for path in tqdm(y_paths):
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            path_without_extension = f'{path[:-4]}'
            save_count = 0
            while True:
                save_success = False
                img_x = self.generate_x_image(img)
                for i in range(self.max_x_pair_count):
                    save_path = f'{path_without_extension}_{X_KEYWORD}_{i}.jpg'
                    if not (os.path.exists(save_path) and os.path.isfile(save_path)):
                        cv2.imwrite(save_path, img_x, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        save_count += 1
                        save_success = True
                        break
                if not save_success:  # case all x index image is generated
                    break
                if save_count == self.generate_count_per_image:
                    break

    def remove(self, path):
        x_paths, _ = self.init_image_paths(path)
        if len(x_paths) == 0:
            print(f'no x images found in {path}')
            exit(0)
        for path in tqdm(x_paths):
            os.remove(path)

    def check(self, path):
        x_paths, y_paths = self.init_image_paths(path)
        if len(y_paths) == 0:
            print(f'no gt images found in {path}')
            exit(0)
        not_paired_y_paths = []
        for y_path in tqdm(y_paths):
            x_exists = False
            for i in range(self.max_x_pair_count):
                x_path = f'{y_path[:-4]}_{X_KEYWORD}_{i}.jpg'
                if os.path.exists(x_path) and os.path.isfile(x_path):
                    x_exists = True
            if not x_exists:
                not_paired_y_paths.append(y_path)

        print(f'y images : {len(y_paths)}')
        print(f'x images : {len(x_paths)}')
        if len(not_paired_y_paths) == 0:
            print('\nall images has x pairs at least once')
        else:
            # for path in not_paired_y_paths:
            #     print(path)
            print(f'\nno x pair image count : {len(not_paired_y_paths)}')

    def rename(self, path):
        x_paths, _ = self.init_image_paths(path)
        if len(x_paths) == 0:
            print(f'no x images found in {path}')
            exit(0)

        for path in tqdm(x_paths):
            path = path.replace('\\', '/')
            path_sp = path.split('/')
            basename = path_sp[-1]
            basename_without_extension = basename[:-4]
            basename_sp = basename_without_extension.split('_')

            x_keyword_position = -1
            x_index_position = -1
            for i in range(len(basename_sp)):
                if basename_sp[i] == X_KEYWORD and i + 1 < len(basename_sp):
                    if basename_sp[i+1] in self.x_index_candidates:
                        x_keyword_position = i
                        x_index_position = i + 1
                        break

            new_basename = basename
            if x_keyword_position > -1 and x_index_position > -1:
                x_index_str = basename_sp.pop(x_index_position)
                basename_sp.pop(x_keyword_position)
                basename_sp.append(X_KEYWORD)
                basename_sp.append(x_index_str)
                new_basename = '_'.join(basename_sp) + '.jpg'
            path_sp[-1] = new_basename
            new_path = '/'.join(path_sp)
            if path != new_path:
                sh.move(path, new_path)

    def split(self, path):
        _, y_paths = self.init_image_paths(path)
        if len(y_paths) == 0:
            print(f'no gt images found in {path}')
            exit(0)

        train_dir_path = f'{path}/train'
        if not (os.path.exists(train_dir_path) and os.path.isdir(train_dir_path)):
            os.makedirs(train_dir_path, exist_ok=True)
        validation_dir_path = f'{path}/validation'
        if not (os.path.exists(validation_dir_path) and os.path.isdir(validation_dir_path)):
            os.makedirs(validation_dir_path, exist_ok=True)

        np.random.shuffle(y_paths)
        validation_rate = 0.2
        validation_count = int(len(y_paths) * validation_rate)
        train_gt_image_paths = y_paths[validation_count:]
        validation_gt_image_paths = y_paths[:validation_count]

        for path in tqdm(train_gt_image_paths):
            sh.move(path, train_dir_path)
            for index in self.x_index_candidates:
                x_path = f'{path[:-4]}_{X_KEYWORD}_{index}.jpg'
                if os.path.exists(x_path) and os.path.isfile(x_path):
                    sh.move(x_path, train_dir_path)
        for path in tqdm(validation_gt_image_paths):
            sh.move(path, validation_dir_path)
            for index in self.x_index_candidates:
                x_path = f'{path[:-4]}_{X_KEYWORD}_{index}.jpg'
                if os.path.exists(x_path) and os.path.isfile(x_path):
                    sh.move(x_path, validation_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='path where generate or remove x image')
    parser.add_argument('--count', type=int, default=1, help='generate count per image')
    parser.add_argument('--show', action='store_true', help='show generated x images for preview')
    parser.add_argument('--generate', action='store_true', help='generate x images')
    parser.add_argument('--remove', action='store_true', help='remove x images')
    parser.add_argument('--check', action='store_true', help='check dataset has paired x data')
    parser.add_argument('--rename', action='store_true', help='move x keyword to end of name')
    parser.add_argument('--split', action='store_true', help='split train, validation dataset with x pairs')
    args = parser.parse_args()
    check_count = 0
    check_count = check_count + 1 if args.show else check_count
    check_count = check_count + 1 if args.generate else check_count
    check_count = check_count + 1 if args.remove else check_count
    check_count = check_count + 1 if args.rename else check_count
    check_count = check_count + 1 if args.check else check_count
    check_count = check_count + 1 if args.split else check_count
    if check_count == 0:
        print('use with one of [--show, --generate, --remove, --rename, --check, --split]')
        exit(0)
    if check_count > 1:
        print('[--show, --generate, --remove, --rename, --check, --split] cannot be used at the same time')
        exit(0)
    x_data_generator = XDataGenerator(args.count)
    if args.show:
        x_data_generator.show(args.path)
    elif args.generate:
        x_data_generator.generate(args.path)
    elif args.remove:
        x_data_generator.remove(args.path)
    elif args.check:
        x_data_generator.check(args.path)
    elif args.rename:
        x_data_generator.rename(args.path)
    elif args.split:
        x_data_generator.split(args.path)

