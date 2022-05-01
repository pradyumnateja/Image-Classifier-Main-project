from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# class labels
class_labels = {0: 'airplane',
                1: 'car',
                2: 'cat',
                3: 'dog',
                4: 'flower',
                5: 'fruit',
                6: 'motorbike',
                7: 'person'}

# helper function to display sample train images
def display_images(train_data):
  plt.figure(figsize=(10, 10))
  for i in range(25):
      ax = plt.subplot(5, 5, i + 1)
      plt.imshow(train_data[0][0][i])
      plt.title(class_labels[train_data[0][1][i]])
      plt.axis("off")

# helper function to predict the image class
def image_prediction(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = expanded_img_array/255.
    prediction = model.predict(preprocessed_img)
    prediction_class = prediction.argmax(axis=1)
    print(f'Predicted as a {class_labels[int(prediction_class)]}!')

# helper function to plot model metrics
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
