import cv2
import numpy as np
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


# Load your trained model (update the model path accordingly)
model = VGG16()

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    image = preprocess_input(image)  # Preprocess using VGG16 specific preprocessing
    return image

def preprocess_image2(image):
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))

# Function to predict and annotate the image
def predict_and_annotate(image):
    #preprocess_image2(image)
    preprocessed_image = preprocess_image(image)
    yhat = model.predict(preprocessed_image)
    label = decode_predictions(yhat)
    label = label[0][0]
    label_text = f'{label[1]} ({label[2]*100:.2f}%)'
    annotated_image = image.copy()
    cv2.putText(annotated_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_image

# Initialize the video capture
cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('USB Camera')

# Variable to store the frame when clicked
captured_frame = None

# Mouse callback function to capture frame on click
def capture_frame(event, x, y, flags, param):
    global captured_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        captured_frame = frame.copy()

global frame
cv2.setMouseCallback('USB Camera', capture_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('USB Camera', frame)

    if captured_frame is not None:
        result_frame = predict_and_annotate(captured_frame)
        cv2.imshow('Detection Result', result_frame)
        captured_frame = None  # Reset captured_frame after processing

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
