
#import sys
import os
#from tensorflow import keras.*
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras import callbacks

#DEV = False
#argvs = sys.argv
#argc = len(argvs)
#
#if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
#  DEV = True
#
#if DEV:
#  epochs = 2
#else:
#  epochs = 20
epochs=15

train_data_path = './train'
validation_data_path = './test'

"""
Parameters
"""
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 100
validation_steps = 30
nb_filters1 = 64
nb_filters2 = 128
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 6
learning_rate = 0.0004

model = Sequential()
model.add(Conv2D(nb_filters1, conv1_size, conv1_size, padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, conv2_size, conv2_size, padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

#"""
#Tensorboard log
#"""
#log_dir = './tf-log/'
#tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#cbks = [tb_cb]

model.fit(
    train_generator,
#    samples_per_epoch=samples_per_epoch,
    steps_per_epoch = 100,
    epochs=epochs,
    validation_data=validation_generator,
#    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')
