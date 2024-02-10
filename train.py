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
import argparse

from image_to_image import TrainingConfig, ImageToImage


if __name__ == '__main__':
    config = TrainingConfig(
        train_image_path='/train_data/coco/train',
        validation_image_path='/train_data/coco/validation',
        model_name='model',
        input_shape=(256, 256, 3),
        output_shape=(256, 256, 3),
        lr=0.001,
        warm_up=0.5,
        batch_size=8,
        iterations=100000,
        save_interval=10000,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', help='evaluate using given dataset')
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='image or video path for prediction or evaluation')
    parser.add_argument('--save-count', type=int, default=0, help='count for save images')
    parser.add_argument('--show', action='store_true', help='show predicted images instead evaluate')
    parser.add_argument('--no-x', action='store_true', help='predict using given gt dataset as no x pair')
    args = parser.parse_args()
    if args.model != '':
        config.pretrained_model_path = args.model
    image_to_image = ImageToImage(config=config, training=not args.evaluate)
    if args.evaluate:
        if args.path.endswith('.mp4'):
            image_to_image.predict_video(video_path=args.path)
        elif args.path.startswith('rtsp://'):
            image_to_image.predict_rtsp(rtsp_url=args.path)
        else:
            image_to_image.evaluate(dataset=args.dataset, image_path=args.path, show=args.show, save_count=args.save_count, no_x=args.no_x)
    else:
        image_to_image.train()

