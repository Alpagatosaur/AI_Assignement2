# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.applications.vgg16 import VGG16

# PATH
path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

img_dir = os.path.join(path_file, 'data_img')
model_dir = os.path.join(path_file, 'my_model.h5')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


def creat_model():

    # Create dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
      img_dir,
      validation_split=0.1,
      subset="training",
      seed=123,
      image_size = IMG_SIZE,
      batch_size = BATCH_SIZE)
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
      img_dir,
      validation_split=0.1,
      subset="validation",
      seed=123,
      image_size = IMG_SIZE)
    
    # Get all name taffic sign
    class_names = train_dataset.class_names
    
    # From validation dataset, get test dataset
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    
    # Config the dataset performaence
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    # Data augmentation (turn img in diff deg)
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])
    
    
    # Rescale pixel values
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    
    # load the pre model VGG16
    base_model  =  VGG16(weights="imagenet", include_top=False)
    
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    
    # Freeze the convolution base
    base_model.trainable = False
    
    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)
    
    "Convert features into a predict per img"
    prediction_layer = tf.keras.layers.Dense(10) # 10 == dimensionality of the output space
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)
    
    # Build a model
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model
    initial_epochs = 10
    loss0, accuracy0 = model.evaluate(validation_dataset)
    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

    # Un-freeze the top layer
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False

    # Compile model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

    # Train model
    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs
    
    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)

    loss, accuracy = model.evaluate(test_dataset)
    
    print("loss : ", loss)
    print("accuracy : ", accuracy)
    
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
      plt.imshow(image_batch[i].astype("uint8"))
      plt.title(class_names[predictions[i]])
      plt.axis("off")
    
    """
    model.save('my_model.h5')
    
    img = image.load_img(img_path, target_size=(160, 160))
    pred = model.predict(img)
    """

def test():
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
    img_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size = IMG_SIZE)
    
    class_names = validation_dataset.class_names
    
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    
    model = tf.keras.models.load_model(model_dir)

    loss, accuracy = model.evaluate(test_dataset)
    
    print("loss : ", loss)
    print("accuracy : ", accuracy)
    
    
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
      plt.imshow(image_batch[i].astype("uint8"))
      plt.title(class_names[predictions[i]])
      plt.axis("off")
      


if __name__ == "__main__":
    x = input("Do you want to re-creat the model? (yes or no) ->  ")
    if x == "yes":
        creat_model()
    x = input("Do you want to test the model (just test accuracy)? (yes or no) ->  ")
    if x == "yes":
        test()