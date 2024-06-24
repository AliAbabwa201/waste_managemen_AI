import joblib
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2

# Load the trained model
model = load_model('Ali_1.h5')

# Load the label encoder
labels = joblib.load('Ali_1.pkl')
class_names = labels.classes_

# Function to preprocess the image
def preprocess_image(image, target_size):
    image = cv2.resize(image, (160,90),interpolation = cv2.INTER_CUBIC)
    cv2.imshow('predict_and_annotate_1', image)
    image = cv2.resize(image, target_size,interpolation = cv2.INTER_CUBIC)  # Resize to model input size
    cv2.imshow('predict_and_annotate', image)
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    image = preprocess_input(image)  # Preprocess using VGG16 specific preprocessing
    return image

# Function to predict and annotate the image
def predict_and_annotate(image):
    preprocessed_image = preprocess_image(image, (70, 70))  # Adjust target size to (70, 70)
    
    yhat = model.predict(preprocessed_image)
    print(yhat)
    predicted_class_index = np.argmax(yhat, axis=1)
    label_text = f'{class_names[predicted_class_index][0]} ({np.max(yhat)*100:.2f}%)'
    annotated_image = image.copy()
    cv2.putText(annotated_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_image

# Initialize the video capture
cap = cv2.VideoCapture(2)

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
        cv2.imshow('Copy', captured_frame)
        result_frame = predict_and_annotate(captured_frame)
        cv2.imshow('Detection Result', result_frame)
        captured_frame = None  # Reset captured_frame after processing

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
