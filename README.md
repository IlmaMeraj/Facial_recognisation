# EmoVision: Facial Emotion Recognition System

EmoVision is a Deep Learning based Facial Emotion Recognition Web Application that detects human emotions from facial images using Computer Vision and Machine Learning.

The system supports:

- Image Upload Emotion Detection

- Real-time Webcam Emotion Prediction

- Flask-based Web Application Interface

This project demonstrates end-to-end ML integration from model training → backend integration → web deployment ready architecture.

Repository: https://github.com/IlmaMeraj/Facial_recognisation

## Features

- Deep learning–based emotion detection
- Real-time webcam emotion prediction
- Image upload emotion prediction
- Flask backend integration
- REST API for webcam / image prediction
- Bootstrap-based responsive frontend UI

## Dataset

The model was trained using the FER-2013 (Facial Expression Recognition 2013) dataset.
Dataset Details:

- Contains ~35,000 grayscale facial images
- Image resolution: 48 × 48 pixels

Covers 7 emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Preprocessing Steps Applied:

- Image resizing to 48 × 48
- Grayscale normalization
- Label encoding for emotion classes
- Data augmentation (rotation, zoom, shifts, horizontal flip) to improve model generalization

The dataset helps the model learn facial expression patterns such as muscle movement, eye shape changes, and mouth curvature associated with different emotions.

## Tech Stack

### Programming Language

- Python

### Machine Learning / AI

- TensorFlow
- Keras
- OpenCV
- NumPy

### Web Development

- Flask
- HTML
- Bootstrap
- JavaScript

## Project Structure

```
facial_expression_recognition/
│
├ model/
│   └ facialemotionmodel.h5
│
├ templates/
│   └ index.html
│
│
├ app.py
├ requirements.txt
├ Procfile
├ trainmodel.ipynb
└ README.md
```

## Installation and Setup

### Clone Repository

```
git clone https://github.com/IlmaMeraj/Facial_recognisation.git
cd Facial_recognisation
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run Application

```
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000
```

## Model Architecture

The system uses a Convolutional Neural Network (CNN) designed for image-based emotion classification.
Custom CNN Architecture:

- Conv Layer 1: 32 filters, 3×3 kernel, ReLU activation
- Max Pooling Layer
- Conv Layer 2: 64 filters, 3×3 kernel, ReLU activation
- Max Pooling Layer
- Conv Layer 3: 128 filters, 3×3 kernel, ReLU activation
- Max Pooling Layer
- Flatten Layer
- Dense Layer: 128 neurons, ReLU activation
- Dropout Layer: 0.5 (prevents overfitting)
- Output Layer: 7 neurons, Softmax activation

Training Configuration:

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Evaluation Metric: Accuracy

## Transfer Learning Enhancement (VGG16)

To improve performance, transfer learning was implemented using VGG16 pretrained on ImageNet.

Approach:

- Used VGG16 convolutional base as feature extractor
- Removed top classification layers
- Added:
  - Global Average Pooling
  - Dense (256 neurons, ReLU)
  - Dropout (0.5)
  - Dense (7 neurons, Softmax)

Benefits:

- Faster training convergence
- Better feature extraction
- Improved generalization
- Achieved ~92% accuracy

## Application Workflow

1. User uploads an image or captures a webcam frame
2. Flask backend receives the image
3. Image preprocessing (resize + normalize + grayscale)
4. Deep learning model prediction
5. Emotion displayed on the web interface

---

## API Endpoints

- Image upload prediction: POST /predict
- Webcam prediction: POST /predict-webcam

## Future Improvements

- Cloud deployment (Render / AWS / GCP)
- Display emotion confidence scores
- Face detection before emotion classification
- React or Next.js frontend integration
- Model accuracy optimization

## Author

Ilma Meraj  
Computer Engineering Student
