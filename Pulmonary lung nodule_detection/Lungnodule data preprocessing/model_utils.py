# model_utils.py

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import Input
from config import IMG_SIZE

# ──────────────────────────────────────────────────────────────
# For traditional CNN model (if used)
# ──────────────────────────────────────────────────────────────
def conv_block(filters, act='relu'):
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPooling2D())
    return block

def dense_block(units, dropout_rate, act='relu'):
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))
    return block

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(conv_block(32))
    model.add(conv_block(64))
    model.add(conv_block(128))
    model.add(conv_block(256))
    model.add(Flatten())
    model.add(dense_block(128, 0.5))
    model.add(dense_block(64, 0.3))
    model.add(dense_block(32, 0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adamax(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ──────────────────────────────────────────────────────────────
# Functional dense block for EfficientNetB3
# ──────────────────────────────────────────────────────────────
def dense_block_fn(x, units, dropout_rate, act='relu'):
    x = Dense(units, activation=act)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x

# ──────────────────────────────────────────────────────────────
# EfficientNetB3 Model - Grad-CAM Compatible
# ──────────────────────────────────────────────────────────────
def build_efficientnetb3_model(input_shape, num_classes):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape, pooling=None)
    base_model.trainable = False  # Optionally True for fine-tuning

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = dense_block_fn(x, 128, 0.5)
    x = dense_block_fn(x, 32, 0.2)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adamax(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
