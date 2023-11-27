import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
from PIL import ImageOps

def preprocess_image_object(img, target_height=48, target_width=48):
    # Check if the image is not grayscale (assuming grayscale images have 1 channel)
    #if img.mode != 'L':
    #    img = ImageOps.grayscale(img)

    # Resize the image to target size
    img = img.resize((target_width, target_height))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array

# Function to classify emotion from image object
def classify_emotion(img_object):
    model_path = '/Users/janbode/computer-vision/GenFacialExpressions/Classifier/model.h5'
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    processed_image = preprocess_image_object(img_object, 48, 48)  # Resizing image to 48x48

    # Predict the emotion
    predictions = model.predict(processed_image)
    emotion_index = np.argmax(predictions, axis=1)
    class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
    emotion = class_labels[emotion_index[0]]

    print("Predicted Emotion:", emotion)

    return class_labels[emotion_index[0]]

