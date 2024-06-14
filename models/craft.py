import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

from utils.train_processing import losses_prepare
from utils.logger import logger


class CRAFT(tf.keras.Model):
    def __init__(self, backbone, num_classes=2, *args, **kwargs):
        super(CRAFT, self).__init__(*args, **kwargs)
        self.backbone      = backbone
        self.num_classes = num_classes

    def build(self, input_shape):
        self.block1 = Sequential([
            MaxPooling2D(3, strides=1, padding='same'),
            Conv2D(1024, kernel_size=3, padding='same', dilation_rate=6),
            Conv2D(1024, kernel_size=1)
        ])
        self.conv1 = self.convolution_block([512, 256])
        self.conv2 = self.convolution_block([256, 128])
        self.conv3 = self.convolution_block([128, 64])
        self.conv4 = self.convolution_block([64, 32])
        self.head = self.model_head()

    def convolution_block(self, filters):
        return Sequential([
            Conv2D(filters[0], kernel_size=1),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters[1], kernel_size=3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
        ])
    
    def model_head(self):
        return Sequential([
            Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            Conv2D(16, kernel_size=3, padding='same', activation='relu'),
            Conv2D(16, kernel_size=1, padding='same', activation='relu'),
            Conv2D(self.num_classes, kernel_size=1, padding='same', activation='sigmoid')
        ])

    def upsample_size(self, x, size):
        return tf.image.resize(x, size=size, method='nearest')

    def call(self, inputs, training=False):
        feature_maps = self.backbone(inputs, training=training)
        S2, S4, S8, S16 = feature_maps[:4]

        x = self.block1(S16, training=training)
        x = self.upsample_size(x, [tf.shape(S16)[1], tf.shape(S16)[2]])
        x = concatenate([x, S16])
        x = self.conv1(x)

        x = self.upsample_size(x, [tf.shape(S8)[1], tf.shape(S8)[2]])
        x = concatenate([x, S8])
        x = self.conv2(x)
        
        x = self.upsample_size(x, [tf.shape(S4)[1], tf.shape(S4)[2]])
        x = concatenate([x, S4])
        x = self.conv3(x)

        x = self.upsample_size(x, [tf.shape(S2)[1], tf.shape(S2)[2]])
        x = concatenate([x, S2])
        x = self.conv4(x)
        
        x = self.head(x)
        region_score, affinity_score = tf.split(x, num_or_size_splits=2, axis=-1)
        region_score = tf.squeeze(region_score, axis=-1)
        affinity_score = tf.squeeze(affinity_score, axis=-1)
        return region_score, affinity_score

    @tf.function
    def predict(self, inputs):
        region_score, affinity_score = self(inputs, training=False)
        return region_score, affinity_score

    def calc_loss(self, y_true, y_pred, loss_object):
        loss = losses_prepare(loss_object)
        loss_value = 0
        if loss:
            loss_value += loss(y_true, y_pred)
        return loss_value
