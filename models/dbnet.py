import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate
from models import ResNet50_backbone, ResNet50


class SegDetector(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 k_factor     = 10,
                 use_bias     = False,
                 use_adaptive = False,
                 use_smooth   = False,
                 use_serial   = False,
                 activation   = 'relu',
                 normalizer   = 'batch-norm',
                 *args, **kwargs):
        super(SegDetector, self).__init__(*args, **kwargs)
        self.filters      = filters
        self.k_factor     = k_factor
        self.use_bias     = use_bias
        self.use_adaptive = use_adaptive
        self.use_smooth   = use_smooth
        self.use_serial   = use_serial
        self.activation   = activation
        self.normalizer   = normalizer

    def build(self, input_shape):
        self.conv_in5 = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=self.use_bias)
        self.conv_in4 = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=self.use_bias)
        self.conv_in3 = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=self.use_bias)
        self.conv_in2 = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=self.use_bias)

        self.upsample5 = UpSampling2D(size=(2, 2))
        self.upsample4 = UpSampling2D(size=(2, 2))
        self.upsample3 = UpSampling2D(size=(2, 2))

        self.conv_up_block5 = Sequential([
            Conv2D(filters=self.filters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=self.use_bias),
            UpSampling2D(size=(8, 8))
        ])
        self.conv_up_block4 = Sequential([
            Conv2D(filters=self.filters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=self.use_bias),
            UpSampling2D(size=(4, 4))
        ])
        self.conv_up_block3 = Sequential([
            Conv2D(filters=self.filters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=self.use_bias),
            UpSampling2D(size=(2, 2))
        ])
        self.conv_up_block2 = Conv2D(filters=self.filters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=self.use_bias)

        self.binarize = Sequential([
            Conv2D(filters=self.filters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=self.use_bias),
            BatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=self.filters // 4, kernel_size=(2, 2), strides=(2, 2), padding='valid'),
            BatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2), padding='valid'),
            Activation('sigmoid')
        ])

        if self.use_adaptive:
            self.thresh = Sequential([
                Conv2D(filters=self.filters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=self.use_bias),
                BatchNormalization(),
                Activation('relu'),
                self.upsample_thresh_block(self.filters // 4, use_bias=self.use_bias),
                BatchNormalization(),
                Activation('relu'),
                self.upsample_thresh_block([self.filters // 4, 1], use_bias=self.use_bias),
                Activation('sigmoid')
            ])

    def upsample_thresh_block(self, filters, use_bias=False):
        if isinstance(filters, int):
            filters = [filters]

        sequence_layers = []
        if self.use_smooth:
            for f in filters:
                if f > 1:
                    sequence_layers.append(UpSampling2D(size=(2, 2)))
                    sequence_layers.append(Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=use_bias))
                else:
                    sequence_layers.append(Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True))
        else:
            for f in filters:
                sequence_layers.append(Conv2DTranspose(filters=f, kernel_size=(2, 2), strides=(2, 2), padding='valid'))
        return Sequential(sequence_layers)

    def call(self, inputs, training=False):
        c2, c3, c4, c5 = inputs
        x5 = self.conv_in5(c5, training=training)
        x4 = self.conv_in4(c4, training=training)
        x3 = self.conv_in3(c3, training=training)
        x2 = self.conv_in2(c2, training=training)

        out4 = self.upsample5(x5, training=training) + x4
        out3 = self.upsample4(x4, training=training) + x3
        out2 = self.upsample3(x3, training=training) + x2

        p5 = self.conv_up_block5(x5, training=training)
        p4 = self.conv_up_block4(out4, training=training)
        p3 = self.conv_up_block3(out3, training=training)
        p2 = self.conv_up_block2(out2, training=training)

        fuse = concatenate([p5, p4, p3, p2], axis=-1)
        binary = self.binarize(fuse, training=training)

        if not training:
            return binary

        if self.use_adaptive:
            if self.use_serial:
                scale_binary = tf.image.resize(binary, size=(tf.shape(fuse)[1], tf.shape(fuse)[2]), method=tf.image.ResizeMethod.BILINEAR)
                fuse = concatenate([fuse, scale_binary], axis=-1)
            thresh = self.thresh(fuse, training=training)
            thresh_binary = 1 + tf.math.exp(-self.k_factor * (binary - thresh))
            thresh_binary = tf.math.reciprocal(thresh_binary)
        return binary, thresh, thresh_binary
