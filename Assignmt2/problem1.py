# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.applications.vgg16 import VGG16


path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

train_dir = os.path.join(path_file, 'train')
img_path = os.path.join(path_file, 'test', '0000.png')



BATCH_SIZE = 32
IMG_SIZE = (160, 160)
"""
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)"""

train_dataset = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size = IMG_SIZE,
  batch_size = BATCH_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size = IMG_SIZE)


class_names = train_dataset.class_names


val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


base_model  =  VGG16(weights="imagenet", include_top=False)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


base_model.trainable = False


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


initial_epochs = 5
loss0, accuracy0 = model.evaluate(validation_dataset)

loss, accuracy = model.evaluate(test_dataset)
model.save('my_model.h5')

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(3):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")


"""
model.save('my_model.h5')

img = image.load_img(img_path, target_size=(160, 160))
pred = model.predict(img)
"""