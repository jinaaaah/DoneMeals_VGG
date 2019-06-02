import numpy as np
import os
import math
import matplotlib.pyplot as plt

from keras import models
from keras import Input
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau

n_classes = 23

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = os.path.join('./data/images/train')
val_dir = os.path.join('./data/images/validate')

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=16,
                                                    target_size=(224, 224),
                                                    color_mode='rgb')
val_generator = val_datagen.flow_from_directory(val_dir,
                                                batch_size=16,
                                                target_size=(224, 224),
                                                color_mode='rgb')

input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')

pre_trained_vgg = VGG19(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3))
pre_trained_vgg.trainable = False

add_model = models.Sequential()
add_model.add(pre_trained_vgg)
add_model.add(Dropout(.4))
add_model.add(Flatten())
add_model.add(Dense(4096, activation='relu'))
add_model.add(Dense(2048, activation='relu'))
add_model.add(Dense(1024, activation='relu'))
add_model.add(Dense(n_classes,
                    init='glorot_uniform',
                    W_regularizer=l2(.0005),
                    activation='softmax'))

checkpoint = ModelCheckpoint(filepath='./model/model1.hdf5',
                             monitor='loss',
                             mode='min',
                             save_best_only=True)
opt = SGD(lr=.01, momentum=.9)

add_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

history = add_model.fit_generator(train_generator,
                                  steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
                                  epochs=300,
                                  validation_data=val_generator,
                                  validation_steps=math.ceil(val_generator.n / val_generator.batch_size),
                                  callbacks=[checkpoint])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()

plt.show()
