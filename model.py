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
import tensorflow as tf


class Model:
    def __init__(self, input_shape, output_shape, build_gan):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.build_gan = build_gan
        self.output_scale = self.calc_output_scale(self.input_shape, self.output_shape)
        self.infos = None

    def calc_output_scale(self, input_shape, output_shape):
        min_rows = min(input_shape[0], output_shape[0])
        max_rows = max(input_shape[0], output_shape[0])
        min_cols = min(input_shape[1], output_shape[1])
        max_cols = max(input_shape[1], output_shape[1])
        assert max_rows % min_rows == 0, f'max rows must be multiple of min_rows, max_rows : {max_rows}, min_rows : {min_rows}'
        assert max_cols % min_cols == 0, f'max cols must be multiple of min_cols, max_cols : {max_cols}, min_cols : {min_cols}'
        row_scale = int(output_shape[0] / input_shape[0])
        col_scale = int(output_shape[1] / input_shape[1])
        assert row_scale == col_scale, f'row scale, col scale must be same. row_scale : {row_scale}, col_scale : {col_scale}'
        output_scale = row_scale
        assert output_scale in [1, 2, 4]
        return output_scale

    def build(self, unet_depth, g_model, d_model, bn=False, dts=True):
        if g_model is None:
            g_input, g_output = self.build_g(unet_depth, bn, dts, activation='leaky' if self.build_gan else 'relu')
        else:
            g_input, g_output = g_model.input, g_model.output
        g_model = tf.keras.models.Model(g_input, g_output)

        d_model, gan = None, None
        if self.build_gan:
            if d_model is None:
                d_input, d_output = self.build_d(bn=False)
            else:
                d_input, d_output = d_model.input, d_model.output
            d_model = tf.keras.models.Model(d_input, d_output)
            gan_output = d_model(g_output)
            gan = tf.keras.models.Model(g_input, gan_output)
        return g_model, d_model, gan

    def build_g(self, unet_depth, bn, dts, activation):
        if unet_depth == 0:
            self.infos = [[32, 3]]
        else:
            self.infos = [[16, 1], [32, 1], [64, 2], [128, 2], [256, 2], [256, 2]]
        g_input = tf.keras.layers.Input(shape=self.input_shape, name='i2i_input')
        x = g_input
        xs = []
        channels, n_convs = self.infos[0]
        for _ in range(n_convs):
            x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)
        for i in range(unet_depth):
            xs.append(x)
            x = self.maxpooling2d(x)
            channels, n_convs = self.infos[i+1]
            for _ in range(n_convs):
                x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)
        for i in range(unet_depth, 0, -1):
            channels, n_convs = self.infos[i-1]
            x = self.conv2d(x, channels, 1, 1, bn=bn, activation=activation)
            x = self.upsampling2d(x)
            x = self.add([x, xs.pop()])
            for _ in range(n_convs):
                x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)

        if self.output_scale == 1:
            g_output = self.g_output(x, g_input, name='i2i_output')
        else:
            if dts:
                output_channels = self.output_shape[-1] * self.output_scale * self.output_scale
                x = self.conv2d(x, output_channels, 1, 1, bn=bn, activation='sigmoid')
                # use hard-coded constant for avoiding Not JSON Serializable error at onnx conversion
                if self.output_scale == 2:
                    g_output = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2), name='i2i_output')(x)
                else:
                    g_output = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 4), name='i2i_output')(x)
            else:
                if self.output_scale >= 2:
                    x = self.conv2dtranspose(x, max(self.infos[0][0]//2, 4), 4, 2, bn=bn, activation=activation)
                if self.output_scale >= 4:
                    x = self.conv2dtranspose(x, max(self.infos[0][0]//4, 4), 4, 2, bn=bn, activation=activation)
                g_output = self.output_layer(x, g_input, name='i2i_output')
        return g_input, g_output

    def build_d(self, bn, activation='leaky'):
        d_input = tf.keras.layers.Input(shape=self.output_shape)
        x = d_input
        x = self.conv2d(x, 16, 3, 2, bn=bn, activation=activation)
        x = self.conv2d(x, 32, 3, 2, bn=bn, activation=activation)
        x = self.conv2d(x, 64, 3, 2, bn=bn, activation=activation)
        x = self.conv2d(x, 128, 3, 2, bn=bn, activation=activation)
        x = self.conv2d(x, 256, 3, 2, bn=bn, activation=activation)
        x = self.flatten(x)
        d_output = self.dense(x, 1, activation='linear', bn=False)
        return d_input, d_output

    def output_layer(self, x, input_layer, bn=False, additive=False, name='i2i_output'):
        if additive:
            assert self.input_shape == self.output_shape
            x = self.conv2d(x, self.input_shape[-1], 1, 1, bn=bn, activation='linear')
            x = self.add([x, input_layer], name=name)
        else:
            x = self.conv2d(x, self.output_shape[-1], 1, 1, bn=bn, activation='sigmoid')
        return x

    def csp_block(self, x, filters, kernel_size, depth, bn=False, activation='relu'):
        half_filters = filters // 2
        x_0 = self.conv2d(x, half_filters, 1, 1, bn=bn, activation=activation)
        x_1 = self.conv2d(x, filters, 1, 1, bn=bn, activation=activation)
        for _ in range(depth):
            x_0 = self.conv2d(x_0, half_filters, kernel_size, 1, bn=bn, activation=activation)
        x_0 = self.conv2d(x_0, filters, 1, 1, bn=bn, activation=activation)
        x = self.add([x_0, x_1])
        x = self.conv2d(x, filters, 1, 1, bn=bn, activation=activation)
        return x

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def conv2dtranspose(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='linear'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def maxpooling2d(self, x):
        return tf.keras.layers.MaxPooling2D()(x)

    def upsampling2d(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, x, name=None):
        return tf.keras.layers.Add(name=name)(x)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        if self.build_gan:
            return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        else:
            return tf.keras.initializers.GlorotNormal()

    def kernel_regularizer(self, l2=0.01):
        return tf.keras.regularizers.l2(l2=l2)

    def activation(self, x, activation, name=None):
        if activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x

