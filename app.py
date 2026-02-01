# import streamlit as st
# import numpy as np
# import cv2
# #from tensorflow.keras.models import load_model
# from PIL import Image

# import tensorflow as tf
# f

# # Load model
# model = tf.load_model("expression_model.h5")

# # Emotion labels (change according to your model)
# classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# st.title("Facial Emotion Recognition")
# st.write("Upload an image and the model will predict the emotion.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
    
#     img = np.array(image.convert('RGB'))
#     img = cv2.resize(img, (48, 48))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     prediction = model.predict(img)
#     emotion = classes[np.argmax(prediction)]
    
#     st.write(f"**Predicted Emotion:** {emotion}")


#import necessary libraries
from flask import Flask, render_template, request,jsonify

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO

from tensorflow.keras.models import load_model

model = load_model("facialemotionmodel.h5")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# take image input from user and predict the emotion
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', emotion="No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', emotion="No selected file")

# Preprocess the image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))
    # img_array = np.array(image).reshape(1, 48, 48, 1) / 255.0

    img_array = image.reshape(1, 48, 48, 1)  # Model expects 4D input
    img_array = img_array / 255.0    # Normalize the image


    # Make a prediction
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    return render_template('index.html', emotion=emotion)

# Route to handle webcam image input
@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    # Get the base64 image from the request
    data = request.get_json()
    image_data = data['image']
    
    # Decode the base64 image
    image_data = image_data.split(',')[1]  # Remove the base64 header part
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    
    # Preprocess the image
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    img_array = image.reshape(1, 48, 48, 1)
    img_array = img_array / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]
    
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)

