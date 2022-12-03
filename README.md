# Embedded and distributed AI
 
use pipreqs to create requirements.txt

## Problem 1: Transfer Learning

    - Choose a pre-trained deep learning model that has been trained to classify images.
    - Use the German Traffic Sign Dataset below (that the pre-trained model in (1) wasn't trained on) to define a new "traffic sign" classification problem.
    - Use transfer learning to adapt the pre-trained model in (1) to your new image classification problem in (2).
    - Is your new model able to classify test images from (2) with high accuracy? 

## Problem 2: Model Compression

    - Compress your new model from problem 1 by converting it into a TF Lite model.
    - Use the TensorFlow Interpreter to load your compressed model in (1) and classify new images.
    - How big is your compressed model compared to the pre-compressed model in problem 1?
    - How quickly does your compressed model classify an image?
    - How much did compressing the model affect classification accuracy? 
