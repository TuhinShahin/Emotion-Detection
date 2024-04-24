
import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json
import threading
import tempfile
import os

# Load JSON and create model
json_file = open('realtime8_emotion_detection.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("realtime8_emotion_detection.weights.h5")
st.write("Loaded model from disk")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion dictionary
emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

# Function to detect emotion from image
def detect_emotion_from_image(image):
    # Convert image to grayscale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Preprocess the ROI for emotion prediction
        roi_color_resized = cv2.resize(roi_color, (48, 48))
        roi_color_resized = cv2.cvtColor(roi_color_resized, cv2.COLOR_BGR2RGB)
        roi_color_resized = np.expand_dims(roi_color_resized, axis=0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(roi_color_resized)
        maxindex = int(np.argmax(emotion_prediction))

        # Display the predicted emotion text above the bounding box
        cv2.putText(image, emotion_dict[maxindex], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image


def display_webcam():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam Feed', 840, 600)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_emotion_from_image(frame)
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Streamlit app layout
st.title('Emotion Detection with Streamlit')

# Sidebar options
option = st.sidebar.selectbox(
    'Choose detection method:',
    ('Detect from Image/Video', 'Real-time Detection')
)

# Function to process and display the video feed
# Function to process and display the video feed
def display_video():
    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    
    # Infinite loop to continuously capture frames
    while run_webcam:
        # Read frame from the webcam feed
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
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
        st.image(frame, channels='BGR', use_column_width=True)

# Start button
if option == 'Real-time Detection':
    run_webcam = False
    if st.button('Start'):
        if not run_webcam:
            run_webcam = True
            # Start a new thread to run the webcam feed
            webcam_thread = threading.Thread(target=display_video)
            webcam_thread.start()


# Upload file for image/video detection
elif option == 'Detect from Image/Video':
    st.write('Upload an image or video file:')
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        # Temporary directory to save the uploaded file
        temp_dir = tempfile.mkdtemp()

        # Save the uploaded file to the temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the uploaded file
        if file_path.endswith('.mp4'):
            # Read video file frame by frame
            video_cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break

                # Detect emotion from each frame
                frame = detect_emotion_from_image(frame)

                # Display the frame in the Streamlit app
                st.image(frame, channels='BGR', use_column_width=True)

            video_cap.release()
        else:
            # Read image file
            image = cv2.imread(file_path)
            image = detect_emotion_from_image(image)

            # Display the image in the Streamlit app
            st.image(image, channels='BGR', use_column_width=True)
