# ASL-Detection
Real-time sign language detection using CNN, TensorFlow, and OpenCV

# ASL Sign Language Detection Using TensorFlow

Sistem deteksi bahasa isyarat berbasis Computer Vision menggunakan
Convolutional Neural Network (CNN), TensorFlow, dan OpenCV.

## Features

- Custom gesture dataset collection using webcam
- Image preprocessing
- CNN-based gesture classification
- Real-time ASL recognition
- Accuracy and loss visualization
- Confusion matrix evaluation

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Workflow

Dataset Collection
→ Preprocessing
→ CNN Training
→ Model Evaluation
→ Real-Time Recognition

## Dataset

Each gesture class contains approximately 1200 images.
The dataset is divided into:

- 80% Training
- 20% Validation

## Model Evaluation

The model is evaluated using:

- Accuracy
- Loss
- Confusion Matrix
- Real-time testing

## How to Run

### Create Dataset

```bash
python create_gestures.py

python cnn_model_train.py

python final.py

```bash
python create_gestures.py
