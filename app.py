import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json

# Load JSON and create model
json_file = open('realtime8_emotion_detection.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("realtime8_emotion_detection.h5")
print("Loaded model from disk")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the desired width and height for displaying the frame
desired_width = 840
desired_height = 680

# Streamlit app layout
st.title('Realtime Emotion Detection')

# Function to process and display the video feed
def display_video():
    cap = cv2.VideoCapture(0)  # Access the webcam
    while cap.isOpened():
        # Read frame from the webcam feed
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to fit the window
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract the region of interest (ROI) containing the face
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Preprocess the ROI for emotion prediction
            roi_color_resized = cv2.resize(roi_color, (48, 48))
            roi_color_resized = cv2.cvtColor(roi_color_resized, cv2.COLOR_BGR2RGB)
            roi_color_resized = np.expand_dims(roi_color_resized, axis=0)

            # Predict the emotion
            emotion_prediction = emotion_model.predict(roi_color_resized)
            maxindex = int(np.argmax(emotion_prediction))

            # Display the predicted emotion text above the bounding box
            cv2.putText(frame, emotion_dict[maxindex], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame in the Streamlit app
        st.image(frame, channels='BGR')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

# Call the function to display the video feed
display_video()
