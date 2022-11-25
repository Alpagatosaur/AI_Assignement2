# Just run the file

import cv2
import numpy as np
import os

YOLOV3_PATH = "yolo"
name_test_image = "test.jpg"
name_res_img = "res.jpg"

value_confidence = 0.25
score_threshold, nms_threshold = 0.1, 0.5
requied_size = (416, 416)

path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))
path_project = '\\'.join(path_file.split('\\')[0:-2])

PATH_CCNAMES = os.path.join(path_project, "Assignmt1", "lib_req", YOLOV3_PATH, "data", "coco.names")
PATH_YOLOV3_TINY_CFG = os.path.join(path_project, "Assignmt1", "lib_req", YOLOV3_PATH, "cfg", "yolov3-tiny.cfg")
PATH_YOLOV3_TINY_WEIGHTS = os.path.join(path_project, "Assignmt1", "lib_req", YOLOV3_PATH, "yolov3-tiny.weights")
PATH_TEST_IMG = os.path.join(path_project, "Assignmt1", "Pb2", "img", name_test_image)
PATH_RESULT_IMG = os.path.join(path_project, "Assignmt1", "Pb2", "img", name_res_img)



# Read labels that are used on object
labels_all = open(PATH_CCNAMES).read().splitlines()

# Change the colors of the boxes
np.random.seed(2)
colors_for_labels = np.random.randint(0, 255, size=(len(labels_all), 3)).tolist()
# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet(PATH_YOLOV3_TINY_CFG, PATH_YOLOV3_TINY_WEIGHTS)
# Determine the layer
inputlayers = net.getLayerNames()
inputlayers = [inputlayers [i-1] for i in net.getUnconnectedOutLayers()]

image = cv2.imread(PATH_TEST_IMG)
# Get the shape
h, w, c = image.shape
# Load it as a blob and feed it to the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, requied_size, swapRB=True, crop=False)
net.setInput(blob)

# Get the output
layer_outputs = net.forward(inputlayers)


# Initialize the lists we need to interpret the results
boxes = []
confidences = []
class_idn = []
# Loop over the layers
for output in layer_outputs:
    # For the layer loop over all detections
    for detection in output:
        # The detection first 4 entries contains the object position and size
        scores = detection[5:]
        # Then it has detection scores - it takes the one with maximal score
        class_id = np.argmax(scores).item()
        # The maximal score is the confidence
        confidence = scores[class_id].item()
        # Ensure we have some reasonable confidence, else ignorre
        if confidence > value_confidence:
            # The first four entries have the location and size (center, size)
            # It needs to be scaled up as the result is given in relative size (0.0 to 1.0)
            box = detection[0:4] * np.array([w, h, w, h])
            center_x, center_y, width, height = box.astype(int).tolist()
            # Calculate the upper corner
            x = center_x - width//2
            y = center_y - height//2
            # Add our findings to the lists
            boxes.append([x, y, width, height])
            confidences.append(confidence)
            class_idn.append(class_id)
# Only keep the best boxes of the overlapping ones
idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
# Ensure at least one detection exists - needed otherwise flatten will fail
if len(idxs) > 0:
    # Loop over the indexes we are keeping
    for i in idxs.flatten():
        # Get the box information
        x, y, w, h = boxes[i]
        # Make a rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), colors_for_labels[class_idn[i]], 2)
        # Make and add text
        text = "{}: {:.4f}".format(labels_all[class_idn[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors_for_labels[class_idn[i]], 2)
        
cv2.imshow("Final", image)
# Save the result
cv2.imwrite(PATH_RESULT_IMG, image)

