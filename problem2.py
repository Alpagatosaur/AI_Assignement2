# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow import lite
from problem1 import load_img

path_file = os.getcwd()

model_dir = os.path.join(path_file, 'model.h5')
output_dir = os.path.join(path_file, 'train_val_test')

# Load model
model = load_model(model_dir)

# Convert the model
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model TFLite
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)



"""Test the new model"""

# Load the TFLite model and allocate tensors.
interpreter = lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

for i in range(11):
    print(" Result TFLite :: class ", i, " : ", output_data[0][i])


# Load test dataset
test_dir = os.path.join(path_file, 'train_val_test', 'test')
list_train_img = glob.glob(test_dir + "/*/*.png")
test_img, test_labels = load_img(list_train_img)
print(test_img.shape)
print(test_labels.shape)
pred = model.predict(test_img)

for i in range(11):
    print(" Result MODEL  :: class ", i, " : ", pred[0][i])
    print(" Result TFLite :: class ", i, " : ", output_data[0][i])
    print("_______________________")
    print(" ")


"""
 Result MODEL  :: class  0  :  0.09775572
 Result TFLite :: class  0  :  0.09203777
_______________________
 
 Result MODEL  :: class  1  :  0.25075567
 Result TFLite :: class  1  :  0.009348555
_______________________
 
 Result MODEL  :: class  2  :  0.52726597
 Result TFLite :: class  2  :  0.0032387807
_______________________
 
 Result MODEL  :: class  3  :  4.4501896e-05
 Result TFLite :: class  3  :  0.00089329126
_______________________
 
 Result MODEL  :: class  4  :  0.012733475
 Result TFLite :: class  4  :  0.0121793905
_______________________
 
 Result MODEL  :: class  5  :  0.0036333448
 Result TFLite :: class  5  :  0.0076124906
_______________________
 
 Result MODEL  :: class  6  :  0.107321255
 Result TFLite :: class  6  :  0.70355636
_______________________
 
 Result MODEL  :: class  7  :  1.93319e-06
 Result TFLite :: class  7  :  0.00044466805
_______________________
 
 Result MODEL  :: class  8  :  3.5825316e-05
 Result TFLite :: class  8  :  0.14504051
_______________________
 
 Result MODEL  :: class  9  :  0.00045007924
 Result TFLite :: class  9  :  0.00561996
_______________________
 
 Result MODEL  :: class  10  :  2.2392553e-06
 Result TFLite :: class  10  :  0.020028291
_______________________
 
"""