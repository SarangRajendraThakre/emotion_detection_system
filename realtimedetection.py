import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture from JSON
try:
    with open("emotiondetectionsrt.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    print("Model architecture loaded successfully.")
except FileNotFoundError:
    print("Error: Model JSON file not found. Ensure 'emotiondetectionsrt.json' exists in the current directory.")
    exit()

# Load the model weights
try:
    model.load_weights("emotiondetectionsrt.weights.h5")
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("Error: Model weights file not found. Ensure 'emotiondetectionsrt.weights.h5' exists in the current directory.")
    exit()

# Load the Haar cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error: Haar cascade file could not be loaded. Check Haar cascade file path.")
    exit()

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Webcam could not be accessed.")
    exit()

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create a named window and set it to fullscreen
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop to capture frames and perform emotion detection
while True:
    # Read a frame from the webcam
    ret, frame = webcam.read()
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = gray[y:y+h, x:x+w]
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Resize the face image to match model input size
        try:
            face_image = cv2.resize(face_image, (48, 48))
            
            # Extract features and normalize
            img = extract_features(face_image)
            
            # Predict emotion
            pred = model.predict(img)
            prediction_label = labels[np.argmax(pred)]
            
            # Display predicted emotion label
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error processing face image: {e}")
    
    # Display the output frame
    cv2.imshow("Output", frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' is pressed
        print("Exiting...")
        break

# Release the webcam and close OpenCV windows  
webcam.release()
cv2.destroyAllWindows()
