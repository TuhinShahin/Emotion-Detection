
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import pickle
from keras.models import load_model

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Preprocess all test images
train_data_gen = train_data_gen.flow_from_directory(
        'C:\\Users\\Md Shahin\\Desktop\\Emotion detection\\Dataset\\train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_data_gen = validation_data_gen.flow_from_directory(
        'C:\\Users\\Md Shahin\\Desktop\\Emotion detection\\Dataset\\test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create CNN model structure
emotion8_model = Sequential()

emotion8_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion8_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion8_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion8_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion8_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion8_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion8_model.add(MaxPooling2D(pool_size=(2, 2)))

# emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

emotion8_model.add(Flatten())
emotion8_model.add(Dense(512, activation='relu'))
emotion8_model.add(Dropout(0.4))
emotion8_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion8_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the neural network/model
emotion_model_info = emotion8_model.fit(
        train_data_gen,
        steps_per_epoch=59725 // 64,
        epochs=50,
        validation_data=validation_data_gen,
        validation_steps=7178 // 64)

# save trained model weight in .h5 file
emotion8_model.save_weights('emotion8_model.h5')
emotion8_model.save('emotion8_model.h5')
# save model structure in jason file
model_json = emotion8_model.to_json()
with open("emotion8_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion8_model.save_weights('emotion8_model.weights.h5')
# emotion_model.save_weights('emotion_model.h5')
# Evaluate the model
evaluation = emotion8_model.evaluate(validation_data_gen)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])
model8 = load_model('model8.h5')  #  model file
history_file = {}  # actual history object
image_paths = ['C:\\Users\\Md Shahin\\Desktop\\Emotion detection\\Dataset\\Plot\\Angry_training.png', 'C:\\Users\\Md Shahin\\Desktop\\Emotion detection\\Dataset\\Plot\\Fearful_training.png', 'C:\\Users\\Md Shahin\\Desktop\\Emotion detection\\Dataset\\Plot\Happy_training.png', 'C:\\Users\\Md Shahin\\Desktop\\Emotion detection\\Dataset\\Plot\sad_training.png']  #image paths
titles = ['angry', 'disgusted','fearful', 'happy','neutral', 'sad' ]  # Titles for the images
model_dir = 'emotion8_detection.h5'  # Directory to save model-related files

# Saving the best model
emotion8_model.save_weights('best_emotion_detection_model_weights.h5')
emotion8_model.save('best_emotion_detection_model.h5')
history_file = 'training8_history.json'
with open(history_file, 'w') as f:
    json.dump(history_file.history, f)

# Function to save the model, its architecture, weights, training history, and plotted images
def save_everything(emotion8_model, history, image_paths, titles, model_dir, plots_dir):
    # Save the model architecture to a JSON file
    with open(f'{model_dir}/model8_architecture.json', 'w') as json_file:
        json_file.write(emotion8_model.to_json())

    # Save the model weights to an HDF5 file
    emotion8_model.save_weights(f'{model_dir}/model8_weights.h5')

    # Save the model history to a pickle file
    with open(f'{model_dir}/model8_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)