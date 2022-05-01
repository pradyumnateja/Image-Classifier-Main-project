# -*- coding: utf-8 -*-
"""ImageClassifier_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ndU8IPktnZL4ezD9ZjV_hJtoMgoBSBlI
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/NSP_Project'

# 1. Import necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from utils import *
import numpy
import math
import matplotlib.pyplot as plt
!pip install split-folders
import splitfolders
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(1234)
numpy.random.default_rng(2022)

# configurations
DATA_DIR = "./natural_images"
NUM_CLASSES = 8
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

class_labels = {0: 'airplane',
                1: 'car',
                2: 'cat',
                3: 'dog',
                4: 'flower',
                5: 'fruit',
                6: 'motorbike',
                7: 'person'}

# Splitting the dataset into train and test sets
data = DATA_DIR
splitfolders.ratio(data, output='/content/drive/MyDrive/NSP_Project/data',
                   seed=2022, ratio=(.8, .2), group_prefix=None)

# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# splitting the train data into train and validation data loading the images
train_data_generator = ImageDataGenerator(rescale=1./255,
                                          validation_split=0.2)

val_data_generator = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.2)

train_data = train_data_generator.flow_from_directory("/content/drive/MyDrive/NSP_Project/data/train", 
                                               target_size=(IMG_WIDTH, IMG_HEIGHT), 
                                               color_mode='rgb',
                                               batch_size=BATCH_SIZE, 
                                               class_mode='sparse',
                                               shuffle=True,
                                               subset = 'training') 

val_data = val_data_generator.flow_from_directory("/content/drive/MyDrive/NSP_Project/data/train", 
                                           target_size=(IMG_WIDTH, IMG_HEIGHT), 
                                           color_mode='rgb',
                                           batch_size=BATCH_SIZE, 
                                           class_mode='sparse',
                                           shuffle=False,
                                           subset = 'validation')

test_data_generator = ImageDataGenerator(rescale=1./255)
test_data = test_data_generator.flow_from_directory("/content/drive/MyDrive/NSP_Project/data/val",
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='sparse',
                                                    shuffle=False,
                                                    subset=None)

# getting the class labels
classes = train_data.class_indices
class_labels = {label: class_ for class_, label in classes.items()}
class_labels

# Reference: https://www.tensorflow.org/tutorials/images/classification#visualize_the_data
def display_images(train_data):
  plt.figure(figsize=(10, 10))
  for i in range(25):
      ax = plt.subplot(5, 5, i + 1)
      plt.imshow(train_data[0][0][i])
      plt.title(class_labels[train_data[0][1][i]])
      plt.axis("off")

# displaying samples from train_data
display_images(train_data)

# Train and Validation samples
TRAIN_SAMPLES = train_data.samples
VALIDATION_SAMPLES = val_data.samples

"""#### Model 1"""

# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/images/classification#model_summary
# https://www.tensorflow.org/tutorials/images/classification#visualize_training_results
# CNN Base model
def base_model():
  input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
  conv1 = Conv2D(32, kernel_size =(5, 5), activation ='relu')(input)
  pooling1 = MaxPooling2D(pool_size =(2, 2))(conv1)
  conv2 = Conv2D(64, kernel_size =(5, 5), activation ='relu')(pooling1)
  pooling2 = MaxPooling2D(pool_size =(2, 2))(conv2)
  conv3 = Conv2D(32, kernel_size =(3, 3), activation ='relu')(pooling2)
  pooling3 = MaxPooling2D(pool_size =(2, 2))(conv3)
  flatten = Flatten()(pooling3)
  dense = Dense(64, activation='relu')(flatten)
  dropout = Dropout(0.5)(dense)
  output = Dense(NUM_CLASSES, activation='softmax')(dropout)

  return Model(inputs=input, outputs=output)

base_model = base_model()
base_model.summary()

# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/images/classification#model_summary
# https://www.tensorflow.org/tutorials/images/classification#visualize_training_results
# https://www.tensorflow.org/tutorials/images/classification#train_the_model
# 3. Train the model
base_model.compile(loss='sparse_categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(0.001),
             metrics=['acc'])
history_base_model = base_model.fit(
    train_data,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE),
    epochs=10,
    validation_data=val_data,
    validation_steps=math.ceil(float(VALIDATION_SAMPLES)/BATCH_SIZE))

model_metrics(history_base_model)

#Reference: https://www.programcreek.com/python/example/89223/keras.preprocessing.image.load_img
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
def image_prediction(model, img_path):
    plt.figure(figsize=(5, 5))
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = expanded_img_array/255.
    prediction = model.predict(preprocessed_img)
    prediction_class = prediction.argmax(axis=1)
    plt.imshow(img)
    plt.title(str(class_labels[int(prediction_class)])+'|'+str(100*np.max(prediction)))

# Saving the model
base_model.save('/content/drive/MyDrive/NSP_Project/base_model.h5')

# 5. Model Prediction
model = load_model('/content/drive/MyDrive/NSP_Project/base_model.h5')

# Evaluating the model on test set
print(f'Accuracy on test set: {model.evaluate(test_data)[1]}')

# Classifying random images
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/motorbike/motorbike_0011.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/dog/dog_0200.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/person/person_0319.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/flower/flower_0475.jpg')

"""#### MobileNet pretrained model"""

# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
def model_maker():
    base_model = MobileNet(include_top=False,
                          input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable=False
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)

mobilenet_model = model_maker()
mobilenet_model.summary()

# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/images/classification#train_the_model
# 3. Train the model
mobilenet_model.compile(loss='sparse_categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(0.001),
             metrics=['acc'])
history_mobilenet = mobilenet_model.fit(
    train_data,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE),
    epochs=10,
    validation_data=val_data,
    validation_steps=math.ceil(float(VALIDATION_SAMPLES)/BATCH_SIZE))

# Reference: https://www.tensorflow.org/tutorials/keras/save_and_load
# 4. Save the model
mobilenet_model.save('/content/drive/MyDrive/NSP_Project/mobilenet_model.h5')

# Reference: https://www.tensorflow.org/tutorials/keras/save_and_load
# 5. Loading the model
model = load_model('/content/drive/MyDrive/NSP_Project/mobilenet_model.h5')

# Evaluating the model on test set
print(f'Accuracy on test set: {model.evaluate(test_data)[1]}')

# Classifying random images
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/motorbike/motorbike_0011.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/dog/dog_0200.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/person/person_0319.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/flower/flower_0475.jpg')

# Reference: https://www.tensorflow.org/tutorials/images/classification#visualize_training_results
def model_metrics(history):
  epochs=10
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

model_metrics(history_mobilenet)

"""### Data Augmentation"""

# Reference: https://www.tensorflow.org/tutorials/images/data_augmentation

train_data_generator = ImageDataGenerator(rescale=1./255,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          rotation_range=20,
                                          validation_split=0.2)

val_data_generator = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.2)

train_data = train_data_generator.flow_from_directory(DATA_DIR, 
                                               target_size=(IMG_WIDTH, IMG_HEIGHT), 
                                               color_mode='rgb',
                                               batch_size=BATCH_SIZE, 
                                               class_mode='sparse',
                                               shuffle=True,
                                               subset = 'training') 

val_data = val_data_generator.flow_from_directory(DATA_DIR, 
                                           target_size=(IMG_WIDTH, IMG_HEIGHT), 
                                           color_mode='rgb',
                                           batch_size=BATCH_SIZE, 
                                           class_mode='sparse',
                                           shuffle=False,
                                           subset = 'validation')

# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/images/classification#train_the_model
mobilenet_model_1 = model_maker()
mobilenet_model.compile(loss='sparse_categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(0.001),
             metrics=['acc'])
history_mobilenet_1 = mobilenet_model.fit(
    train_data,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE),
    epochs=10,
    validation_data=val_data,
    validation_steps=math.ceil(float(VALIDATION_SAMPLES)/BATCH_SIZE))

# Reference: https://www.tensorflow.org/tutorials/keras/save_and_load
mobilenet_model.save('/content/drive/MyDrive/NSP_Project/mobilenet_model_v2.h5')

# 5. Model Prediction
# loading the model
model = load_model('/content/drive/MyDrive/NSP_Project/mobilenet_model_v2.h5')

# Evaluating the model on test set
print(f'Accuracy on test set: {model.evaluate(test_data)[1]}')

# Classifying random images
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/motorbike/motorbike_0011.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/dog/dog_0200.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/person/person_0319.jpg')
image_prediction(model, '/content/drive/MyDrive/NSP_Project/natural_images/flower/flower_0475.jpg')

model_metrics(history_mobilenet_1)

