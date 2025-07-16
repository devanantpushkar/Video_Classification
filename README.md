Overview
This project is a deep learning-based system that classifies videos into three categories:

General

Obscene

Violent

It uses a Convolutional Neural Network (CNN) with optional Spatial Attention to analyze keyframes extracted from a video. It supports both offline video uploads and real-time screen monitoring, built using a Streamlit interface.

- Features
  
 1)Keyframe extraction using I-frames via PyAV

2) Frame-by-frame classification using a CNN model

3) Streamlit web app for video upload and real-time display

4) Real-time screen capture & classification using MSS

5) Aggregates predictions using average probabilities and threshold voting

- How It Works
Extract frames from uploaded video (or screen).

Resize and normalize frames to [0, 1].

Predict class for each frame using trained CNN.

Combine per-frame predictions to determine overall label.

Show result in a user-friendly web UI.

- Model Architecture
scss
Copy
Edit
Conv2D(32) -> MaxPooling2D ->
Conv2D(64) -> MaxPooling2D ->
[Optional: Spatial Attention] ->
Flatten -> Dense(128) -> Dense(3, softmax)
Input: 256x256x3 RGB frame

Output: Class probabilities for General, Obscene, and Violent

-Training Summary
Dataset: ~2,500 labeled keyframes

Train/Validation: 80/20 split

Optimizer: Adam

Loss: Categorical Crossentropy

Regularization: Dropout(0.5), EarlyStopping

- File Structure
File	Description
app1.py	Streamlit app for video upload
screen_reader_app.py	Real-time screen monitoring app
keyframe.py	Extracts keyframes using PyAV
model_utils.py	Prediction & preprocessing logic
model.h5	Trained Keras CNN model

-Technologies Used
TensorFlow / Keras

Streamlit

OpenCV

PyAV

NumPy

MSS (for screen capture)

-Sample Output
yaml
Copy
Edit
General: 0.63
Obscene: 0.23
Violent: 0.14
Final Classification: General
- Future Enhancements
1)Integrate pre-trained ViT (Vision Transformer)
2)Add 3D CNN or LSTM for temporal understanding
3)Incorporate audio classification








