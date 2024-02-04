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
    def __init__(self, input_rows, input_cols, input_channels, output_channels):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_shape = (self.input_rows, self.input_cols, self.input_channels)

    def build(self, bn=False):
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='i2i_input')
        x = input_layer
        x = self.conv2d(x, 16, 3, 1, bn=bn, activation='relu')
        f0 = x
        x = self.maxpooling2d(x)
        x = self.csp_block(x, 32, 3, 2, bn=bn, activation='relu')
        f1 = x
        x = self.maxpooling2d(x)
        x = self.csp_block(x, 64, 3, 3, bn=bn, activation='relu')
        x = self.conv2d(x, 32, 1, 1, bn=bn, activation='relu')
        x = self.upsampling2d(x)
        x = self.add([x, f1])
        x = self.csp_block(x, 32, 3, 2, bn=bn, activation='relu')
        x = self.conv2d(x, 16, 1, 1, bn=bn, activation='relu')
        x = self.upsampling2d(x)
        x = self.add([x, f0])
        x = self.csp_block(x, 16, 3, 1, bn=bn, activation='relu')
        output_layer = self.output_layer(x, input_layer, name='i2i_output')
        return tf.keras.models.Model(input_layer, output_layer)

    def output_layer(self, x, input_layer, bn=False, additive=False, name='i2i_output'):
        if additive:
            assert self.input_channels == self.output_channels
            x = self.conv2d(x, self.input_channels, 1, 1, bn=bn, activation='linear')
            x = self.add([x, input_layer], name=name)
        else:
            x = self.conv2d(x, self.output_channels, 1, 1, bn=bn, activation='sigmoid')
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

    def maxpooling2d(self, x):
        return tf.keras.layers.MaxPooling2D()(x)

    def upsampling2d(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, x, name=None):
        return tf.keras.layers.Add(name=name)(x)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        return tf.keras.initializers.he_normal()

    def kernel_regularizer(self, l2=0.01):
        return tf.keras.regularizers.l2(l2=l2)

    def activation(self, x, activation, name=None):
        if activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x

