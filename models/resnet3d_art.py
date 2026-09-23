
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def bottleneck_block_art(x, filters, strides=(1, 1, 1), downsample=False):
    shortcut = x

    x = layers.Conv3D(filters, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters, 3, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters * 4, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if downsample or strides != (1, 1, 1):
        shortcut = layers.Conv3D(filters * 4, 1, strides=strides, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50_art(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv3D(64, 7, strides=(1, 2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Stage 2
    x = bottleneck_block_art(x, 64, downsample=True)
    for _ in range(2):
        x = bottleneck_block_art(x, 64)
    stage2_conv = x
    stage2 = layers.GlobalAveragePooling3D()(x)

    # Stage 3
    x = bottleneck_block_art(x, 128, strides=(2, 2, 2), downsample=True)
    for _ in range(3):
        x = bottleneck_block_art(x, 128)
    stage3_conv = x
    stage3 = layers.GlobalAveragePooling3D()(x)

    # Stage 4
    x = bottleneck_block_art(x, 256, strides=(2, 2, 2), downsample=True)
    for _ in range(5):
        x = bottleneck_block_art(x, 256)
    stage4_conv = x
    stage4 = layers.GlobalAveragePooling3D()(x)

    # Stage 5
    x = bottleneck_block_art(x, 512, strides=(2, 2, 2), downsample=True)
    for _ in range(2):
        x = bottleneck_block_art(x, 512)
    stage5_conv = x
    stage5 = layers.GlobalAveragePooling3D()(x)

    model = keras.Model(inputs, [stage2, stage3, stage4, stage5,
                                 stage2_conv, stage3_conv, stage4_conv, stage5_conv],
                        name="3d_resnet50_art")
    return model
