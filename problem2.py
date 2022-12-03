# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow import lite
from problem1 import load_img
import time

path_file = os.getcwd()

model_dir = os.path.join(path_file, 'model.h5')
output_dir = os.path.join(path_file, 'train_val_test')


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

# TIMER :  8 s

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


# Load val dataset
val_dir = os.path.join(path_file, 'train_val_test', 'val')
list_val_img = glob.glob(val_dir + "/*/*.png")
val_img, val_labels = load_img(list_val_img)

# Test with 10 img
nb_test_img = 10

for i in range(nb_test_img):
    img_8 = val_img[i]
    label = val_labels[i]
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
1/1 [==============================] - 0s 196ms/step
 OUTPUT TFLITE  2 0.54476225
 OUTPUT MODEL   2 0.54476285
 LABEL   2
_______________________

1/1 [==============================] - 0s 47ms/step
 OUTPUT TFLITE  3 0.9597202
 OUTPUT MODEL   3 0.95972043
 LABEL   3
_______________________

1/1 [==============================] - 0s 48ms/step
 OUTPUT TFLITE  5 0.4582212
 OUTPUT MODEL   5 0.45822024
 LABEL   3
_______________________

1/1 [==============================] - 0s 51ms/step
 OUTPUT TFLITE  1 0.9450509
 OUTPUT MODEL   1 0.9450509
 LABEL   1
_______________________

1/1 [==============================] - 0s 47ms/step
 OUTPUT TFLITE  2 0.8629077
 OUTPUT MODEL   2 0.86290735
 LABEL   2
_______________________

1/1 [==============================] - 0s 47ms/step
 OUTPUT TFLITE  2 0.65619695
 OUTPUT MODEL   2 0.6561966
 LABEL   2
_______________________

1/1 [==============================] - 0s 46ms/step
 OUTPUT TFLITE  0 0.49432638
 OUTPUT MODEL   0 0.49432668
 LABEL   9
_______________________

1/1 [==============================] - 0s 62ms/step
 OUTPUT TFLITE  6 0.43688843
 OUTPUT MODEL   6 0.4368882
 LABEL   4
_______________________

1/1 [==============================] - 0s 51ms/step
 OUTPUT TFLITE  1 0.9473194
 OUTPUT MODEL   1 0.9473196
 LABEL   1
_______________________

1/1 [==============================] - 0s 48ms/step
 OUTPUT TFLITE  1 0.8291788
 OUTPUT MODEL   1 0.82917905
 LABEL   1
_______________________


"""
print("END COMPARAISON\n")

print("SIZE OF TFLite : 59 591 Ko")
print("SIZE OF MODEL  : 178 860 Ko")

print("TFLite represents 33% of MODEL's size")