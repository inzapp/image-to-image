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
    parser.add_argument('--rows', type=int, default=0, help='input rows for model')
    parser.add_argument('--cols', type=int, default=0, help='input cols for model')
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--type', type=str, default='', help='pretrained model input type : [gray, rgb, nv12, nv21]')
    parser.add_argument('--predict', action='store_true', help='predict using given x dataset')
    parser.add_argument('--predict-gt', action='store_true', help='predict using given gt dataset')
    parser.add_argument('--evaluate', action='store_true', help='evaluate using given dataset')
    parser.add_argument('--evaluate-gt', action='store_true', help='evaluate using given dataset without model forwarding')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='image or video path for prediction or evaluation')
    parser.add_argument('--save-count', type=int, default=0, help='count for save images')
    args = parser.parse_args()
    if args.rows > 0 and args.cols > 0:
        config.input_rows = args.rows
        config.input_cols = args.cols
    if args.model != '':
        config.pretrained_model_path = args.model
    if args.type != '':
        config.input_type = args.type
    image_to_image = ImageToImage(config=config, training=not (args.predict or args.predict_gt or args.evaluate or args.evaluate_gt))
    if args.predict or args.predict_gt:
        if args.path.endswith('.mp4'):
            image_to_image.predict_video(video_path=args.path)
        elif args.path.startswith('rtsp://'):
            image_to_image.predict_rtsp(rtsp_url=args.path)
        else:
            image_to_image.predict_images(image_path=args.path, dataset=args.dataset, save_count=args.save_count, predict_gt=args.predict_gt)
    elif args.evaluate or args.evaluate_gt:
        image_to_image.evaluate(image_path=args.path, dataset=args.dataset, evaluate_gt=args.evaluate_gt)
    else:
        image_to_image.train()

