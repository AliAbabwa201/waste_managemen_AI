import cv2
import numpy as np
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# Load the VGG16 model
model = VGG16()

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    image = preprocess_input(image)  # Preprocess using VGG16 specific preprocessing
    return image

# Function to predict and annotate the image
def predict_and_annotate(image):
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

# Variables to store the state of the rectangle
drawing = False
start_point = (0, 0)
end_point = (0, 0)
rectangle = None
captured_frame = None

# Mouse callback function to draw rectangle
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, rectangle, captured_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        rectangle = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rectangle = (start_point, end_point)
        if rectangle is not None:
            x1, y1 = rectangle[0]
            x2, y2 = rectangle[1]
            captured_frame = frame[y1:y2, x1:x2]

cv2.setMouseCallback('USB Camera', draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if drawing:
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
    elif rectangle is not None:
        cv2.rectangle(frame, rectangle[0], rectangle[1], (0, 255, 0), 2)

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

