import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.misc import imread

from metrics import f1 as f1_score
from utils import make_dir

import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, Cropping2D, Input, merge
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K  # want the Keras modules to be compatible
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def UNet(filters_dims, activation='relu', kernel_initializer='glorot_uniform', padding='same'):
    inputs = Input((480, 640, 3))
    new_inputs = inputs
    conv_layers = []
    # Encoding Phase
    for i in range(len(filters_dims) - 1):
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(new_inputs)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(conv)
        conv_layers.append(conv)
        new_inputs = MaxPooling2D(pool_size=(2, 2))(conv)
        # op = BatchNormalization()(op)

    # middle phase
    conv = Conv2D(filters_dims[-1], 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_initializer)(new_inputs)
    conv = Conv2D(filters_dims[-1], 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_initializer)(conv)
    new_inputs = Dropout(0.5)(conv)

    filters_dims.reverse()
    conv_layers.reverse()

    # Decoding Phase
    for i in range(1, len(filters_dims)):
        up = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                    kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(new_inputs))
        concat = merge([conv_layers[i-1], up], mode='concat', concat_axis=3)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(concat)
        new_inputs = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                            kernel_initializer=kernel_initializer)(conv)
    outputs = Conv2D(1, 1, activation='softmax', padding='same',
                     kernel_initializer='glorot_uniform')(new_inputs)

    model = Model(input=inputs, output=outputs, name='UNet')
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse', f1_score])
    return model


def train(model, x_dir, y_dir, batch_size, epochs):
    # Running on multi GPU
    print('Tensorflow backend detected; Applying memory usage constraints')
    ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True),
                                              log_device_placement=True))
    K.set_session(ss)
    ss.run(K.tf.global_variables_initializer())
    K.set_learning_phase(1)

    # print("Getting data.. Image shape: {}. Masks shape : {}".format(x.shape,
    #                                                                 y.shape))
    # print("The data will be split to Train Val: 80/20")

    # saving weights and logging
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    make_dir(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_weights_only=True, save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/')

    # history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
    #                     verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)

    # getting image data generator
    seed = 1

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(x_dir,
                                                        (480, 640),
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(y_dir,
                                                      (480, 640),
                                                      color_mode='grayscale',
                                                      class_mode=None,
                                                      batch_size=batch_size,
                                                      seed=seed)
    train_generator = zip(image_generator, mask_generator)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=2000,
                                  epochs=epochs)

    return history


if __name__ == '__main__':
    unet_config = 'config/unet.json'
    print('unet json: {}'.format(os.path.abspath(unet_config)))
    with open(unet_config) as json_file:
        config = json.load(json_file)
    print("Initializing UNet model")
    model = UNet(filters_dims=config['filters_dims'],
                 activation=config['activation'],
                 kernel_initializer=config['kernel_initializer'],
                 padding=config['padding'])

    training_config = 'config/training.json'
    print('training json: {}'.format(os.path.abspath(training_config)))
    with open(training_config) as json_file:
        config = json.load(json_file)

    print("Loading data")
    data_path = os.path.join(os.getcwd(), 'data', 'Augments_train')
    Original_image_path = os.path.join(data_path, 'Original')
    # Original_image_names = os.listdir(Original_image_path)
    # Original_image_names.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    # Original_image_names = Original_image_names[:700]

    SegMask_path = os.path.join(data_path, 'SegMask_Wound_Bg')
    # SegMask_names = os.listdir(SegMask_path)
    # SegMask_names.sort(key=lambda x: int(x.split('_')[4].split('.')[0]))
    # SegMask_names = SegMask_names[:700]

    # Original_images = np.array([imread(os.path.join(Original_image_path, x))
    #                             for x in Original_image_names])
    # SegMasks = np.array([imread(os.path.join(SegMask_path, x)) for x in SegMask_names])

    # Checking imports
    # Check labeling
    # assert list(map(lambda x: x.split('_')[4], SegMask_names)) == list(
    #     map(lambda x: x.split('_')[2], Original_image_names)), "Original and Masks are not in sync. Please recheck"

    # Check shape
    # assert list(map(lambda x: x.shape[:2], Original_images)
    #             ) == list(map(lambda x: x.shape[:2], SegMasks)), "Original and Masks do not have the same WxH. Please recheck"

    # transformed_images, transformed_masks = resizing(
    #     image_path, mask_path, verbose=False, plot=False)

    # transformed_images = np.array(transformed_images)
    # transformed_masks = np.array(transformed_masks)
    # print("Performing one hot encoding")
    # SegMasks = to_categorical(SegMasks, 2)
    # print("SegMasks shape: ", SegMasks.shape)
    print("Initializing training instance")

    train(model=model,
          x_dir=Original_image_path,
          y_dir=SegMask_path,
          batch_size=config["batch_size"],
          epochs=config["epochs"])
