# -*- coding: utf-8 -*-
import tensorflow as tf
import os

path_file = os.getcwd()


model = tf.keras.models.load_model(path_file + "\my_model.h5")
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
