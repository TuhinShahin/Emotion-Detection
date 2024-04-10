
import cv2
import os
import numpy as np
from keras.models import load_model


emotion_names = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


# Function to preprocess the input image
def preprocess_image(image, target_size=(48, 48)):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to the target size
    resized_image = cv2.resize(gray_image, target_size)
    # Convert grayscale image back to RGB format (3 channels)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    # Normalize pixel values
    normalized_image = rgb_image / 255.0
    return normalized_image

# Function to capture frames from webcam and label them with predicted emotions
# Function to capture frames from webcam and label them with predicted emotions
# Define a dictionary to map emotion numbers to their names
emotion_names = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Function to capture frames from webcam and label them with predicted emotions
def capture_and_label_frames(output_folder, model_path, crop_size=(48, 48)):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load pre-trained emotion detection model
    model = load_model(model_path)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Initialize frame count
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if any faces are detected
        if len(faces) == 0:
            # No faces detected, continue to next iteration
            continue

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the face region
            face_crop = frame[y:y+h, x:x+w]

            # Preprocess the cropped face image
            preprocessed_image = preprocess_image(face_crop)

            # Reshape the preprocessed image to match model input shape
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

            # Predict emotion label using the model
            emotion_prediction = model.predict(preprocessed_image)
            predicted_emotion = np.argmax(emotion_prediction)

            # Get the emotion name from the dictionary
            emotion_name = emotion_names.get(predicted_emotion, "Unknown")

            # Display predicted emotion label on the frame
            cv2.putText(frame, f"Emotion: {emotion_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Create subfolder for the predicted emotion if it doesn't exist
            emotion_folder = os.path.join(output_folder, f"emotion_{emotion_name}")
            os.makedirs(emotion_folder, exist_ok=True)

            # Save the cropped face with predicted emotion label
            face_filename = os.path.join(emotion_folder, f"frame_{frame_count}.png")
            cv2.imwrite(face_filename, face_crop)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Increment frame count
        frame_count += 1

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()



# Specify the output folder where labeled frames will be saved
output_folder = "New_training_image"

# Specify the path to the pre-trained emotion detection model
model_path = 'realtime8_emotion_detection.h5'

# Capture frames from webcam and label them with predicted emotions
capture_and_label_frames(output_folder, model_path)


