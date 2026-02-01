from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("model/facialemotionmodel.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


@app.route('/')
def home():
    return render_template('index.html')


# Image Upload Prediction
@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return render_template('index.html', emotion="No image uploaded")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', emotion="No file selected")

    # Preprocess
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (48, 48))
    img_array = image.reshape(1, 48, 48, 1) / 255.0

    # Predict
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    return render_template('index.html', emotion=emotion)


# Webcam API Prediction
@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():

    data = request.get_json()
    image_data = data['image']

    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    img_array = image.reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    return jsonify({'emotion': emotion})


if __name__ == '__main__':
    app.run()
