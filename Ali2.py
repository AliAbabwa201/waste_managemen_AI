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
    image = cv2.resize(image, target_size)  # Resize to model input size
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    image = preprocess_input(image)  # Preprocess using VGG16 specific preprocessing
    return image

# Function to predict and annotate the image
def predict_and_annotate(image):
    preprocessed_image = preprocess_image(image, (70, 70))  # Adjust target size to (70, 70)
    yhat = model.predict(preprocessed_image)
    
    # Debugging: Print raw prediction values
    print("Raw predictions:", yhat)
    
    predicted_class_index = np.argmax(yhat, axis=1)
    
    # Debugging: Print predicted class index and corresponding label
    print("Predicted class index:", predicted_class_index)
    print("Predicted label:", class_names[predicted_class_index][0])
    
    label_text = f'{class_names[predicted_class_index][0]} ({np.max(yhat)*100:.2f}%)'
    annotated_image = image.copy()
    cv2.putText(annotated_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_image

# Initialize the video capture with 1280x720 resolution
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

cv2.setMouseCallback('USB Camera', capture_frame)
global frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Stretch the image to 1280x720 resolution
    frame = cv2.resize(frame, (1280, 720))

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
