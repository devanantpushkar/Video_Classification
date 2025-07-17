 Video Classification using CNN with Attention
A deep learning-based system for classifying video content into three categories: General, Obscene, and Violent, using Convolutional Neural Networks (CNN) with optional Spatial Attention. Built with TensorFlow + Keras and deployed with a user-friendly Streamlit interface. Supports both video uploads and real-time screen monitoring.

📌 Table of Contents
Features

Dataset

Approach

Model Training

Model Architecture

Accuracy

Technologies Used

How to Run

Future Improvements

🚀 Features
🎥 Keyframe Extraction using PyAV (I-frames)

🧠 Frame-by-Frame Classification using CNN

💡 Optional Spatial Attention Layer for focused learning

🌐 Streamlit Web Interface for:

Video Upload

Real-time Screen Monitoring

📊 Aggregated Prediction using Averaging + Threshold Voting

📂 Dataset
Total Labeled Keyframes: ~2,500 images

Classes: General, Obscene, Violent

Split:

2,000 for training (80%)

500 for validation (20%)

Source: Frames manually labeled from various public videos

🧭 Approach
Frame Extraction:

Use either fixed rate (1 frame/sec) or keyframe extraction (I-frames using PyAV).

Preprocessing:

Resize to 256x256

Normalize pixel values to range [0, 1]

Model Prediction:

CNN model processes each frame and predicts class probabilities.

Optional Spatial Attention layer enhances focus on important frame regions.

Prediction Aggregation:

Average all frame predictions.

OR count frames with >80% confidence for a class (threshold voting).

Final Classification:

If majority frames show high probability for Obscene or Violent, raise alert.

Otherwise, label as General.

🧠 Model Training
Frameworks: TensorFlow + Keras

Preprocessing:

Resize to 256×256

Normalize pixel values to [0, 1]

Training Strategy:

Optimizer: Adam

Loss: Categorical Crossentropy

Batch Size: 32

Epochs: 10 with early stopping

Regularization: Dropout(0.5)

🏗️ Model Architecture
python
Copy
Edit
Conv2D(32, kernel_size=(3, 3), activation='relu')
MaxPooling2D(pool_size=(2, 2))
Conv2D(64, kernel_size=(3, 3), activation='relu')
MaxPooling2D(pool_size=(2, 2))
[Optional: Spatial Attention Layer]
Flatten()
Dense(128, activation='relu')
Dropout(0.5)
Dense(3, activation='softmax')
Output: Class probabilities for General, Obscene, and Violent

✅ Accuracy
Training Accuracy: ~85%

Validation Accuracy: ~80%

Metrics Used: Accuracy, Precision, Recall, F1-Score

🛠️ Technologies Used
TensorFlow/Keras: Deep learning model training

OpenCV: Frame extraction and image processing

PyAV: Keyframe extraction (I-frames)

MSS: Screen capture for real-time input

Streamlit: Web-based frontend for user interaction

