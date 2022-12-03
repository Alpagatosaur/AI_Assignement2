# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow import lite
from problem1 import load_img
import time

import splitfolders

path_file = os.getcwd()

model_dir = os.path.join(path_file, 'model.h5')
output_dir = os.path.join(path_file, 'train_val_test')
img_dir = os.path.join(path_file, 'data_img')

# Load model
model = load_model(model_dir)

# Start time

start = time.time()

# Convert the model
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# End

end = time.time()

print("TIMER : ", int(end - start), "s\n")

# TIMER :  9 s

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

# Create folder for test validation
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio(img_dir, output=output_dir, seed=1337, ratio=(.8, .1, .1))
    
# Load test dataset
test_dir = os.path.join(path_file, 'train_val_test', 'test')
list_test_img = glob.glob(test_dir + "/*/*.png")
test_img, test_labels = load_img(list_test_img)

# Test with 10 img
nb_test_img = 10

for i in range(nb_test_img):
    img_8 = test_img[i]
    label = test_labels[i]
    img_32 = img_8.astype('float32')
    img_32 = np.expand_dims(img_32, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_32)

    interpreter.invoke()
    
    output_tflite = interpreter.get_tensor(output_details[0]['index'])
    output_model = model.predict(img_32)

    print(" OUTPUT TFLITE ", output_tflite[0].argmax(), max(output_tflite[0]))
    print(" OUTPUT MODEL  ", output_model[0].argmax(), max(output_model[0]))
    print(" LABEL  ", label)
    print("_______________________\n")

"""
1/1 [==============================] - 0s 281ms/step
 OUTPUT TFLITE  2 0.5272661
 OUTPUT MODEL   2 0.5272658
 LABEL   2
_______________________

1/1 [==============================] - 0s 45ms/step
 OUTPUT TFLITE  3 0.9516666
 OUTPUT MODEL   3 0.9516667
 LABEL   3
_______________________

1/1 [==============================] - 0s 45ms/step
 OUTPUT TFLITE  3 0.5781447
 OUTPUT MODEL   3 0.5781444
 LABEL   3
_______________________

1/1 [==============================] - 0s 44ms/step
 OUTPUT TFLITE  1 0.96721864
 OUTPUT MODEL   1 0.96721864
 LABEL   1
_______________________

1/1 [==============================] - 0s 45ms/step
 OUTPUT TFLITE  2 0.9204558
 OUTPUT MODEL   2 0.9204553
 LABEL   2
_______________________

1/1 [==============================] - 0s 46ms/step
 OUTPUT TFLITE  2 0.6046625
 OUTPUT MODEL   2 0.60466224
 LABEL   2
_______________________

1/1 [==============================] - 0s 59ms/step
 OUTPUT TFLITE  6 0.46202195
 OUTPUT MODEL   6 0.46202204
 LABEL   9
_______________________

1/1 [==============================] - 0s 54ms/step
 OUTPUT TFLITE  4 0.7411054
 OUTPUT MODEL   4 0.7411053
 LABEL   4
_______________________

1/1 [==============================] - 0s 51ms/step
 OUTPUT TFLITE  1 0.8826441
 OUTPUT MODEL   1 0.88264394
 LABEL   1
_______________________

1/1 [==============================] - 0s 64ms/step
 OUTPUT TFLITE  1 0.997813
 OUTPUT MODEL   1 0.997813
 LABEL   1
_______________________


"""
print("END COMPARAISON\n")

print("SIZE OF TFLite : 59 591 Ko")
print("SIZE OF MODEL  : 178 860 Ko")

print("TFLite represents 33% of MODEL's size")