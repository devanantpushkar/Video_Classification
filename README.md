# Video Classification

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/devanantpushkar/Video_Classification?style=social)](https://github.com/devanantpushkar/Video_Classification/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/devanantpushkar/Video_Classification?style=social)](https://github.com/devanantpushkar/Video_Classification/network/members)
[![License: Unlicensed](https://img.shields.io/badge/License-Unlicensed-lightgrey.svg)](https://choosealicense.com/no-permission/)

A deep learning-based system designed to classify video content into three distinct categories: **General**, **Obscene**, and **Violent**. This project leverages a Convolutional Neural Network (CNN) with optional Spatial Attention to analyze keyframes extracted from videos. It provides versatile functionality, supporting both offline video uploads and real-time screen monitoring, all accessible through an intuitive Streamlit web interface.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Usage](#usage)
  - [Using the Streamlit Web App](#using-the-streamlit-web-app)
  - [Programmatic Usage](#programmatic-usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

This system offers a robust set of features for comprehensive video classification:

1.  **Keyframe Extraction**: Efficiently extracts keyframes (I-frames) from videos using `PyAV` to focus analysis on the most informative frames.
2.  **Frame-by-Frame Classification**: Utilizes a pre-trained Convolutional Neural Network (CNN) to classify individual keyframes.
3.  **Streamlit Web Application**: Provides a user-friendly, interactive web interface for:
    *   **Offline Video Upload**: Upload local video files for classification.
    *   **Real-time Screen Monitoring**: Capture and classify content directly from your screen in real-time using `MSS`.
4.  **Prediction Aggregation**: Combines per-frame predictions to determine an overall video label, employing strategies like average probabilities and threshold-based voting for robust results.
5.  **Pre-trained Model**: Includes a pre-trained `.h5` model for immediate use.

## How It Works

The video classification process follows these main steps:

1.  **Input Acquisition**: Videos can be provided either by uploading a local file or by capturing a live screen stream.
2.  **Keyframe Extraction**: For the given video or stream, intelligent keyframe extraction identifies and isolates the most significant frames.
3.  **Preprocessing**: Each extracted keyframe is resized, normalized (e.g., to a `[0, 1]` range), and prepared for model input.
4.  **Frame-level Prediction**: The preprocessed keyframes are fed into the trained CNN model, which predicts the class (General, Obscene, Violent) for each individual frame.
5.  **Prediction Aggregation**: The individual frame predictions are aggregated using statistical methods (e.g., averaging probabilities, majority voting based on thresholds) to determine the final classification label for the entire video.
6.  **User Interface Display**: The classification result, along with any relevant metrics, is displayed in a clear and user-friendly manner via the Streamlit web application.

## Model Architecture

The core of the classification system is a Convolutional Neural Network (CNN). While the exact architecture can vary, a typical structure might involve several convolutional and pooling layers to extract spatial features, followed by fully connected layers for classification. The model is trained to distinguish between the three defined categories.

A simplified representation of the model's architecture could be:

```
Input Layer (e.g., 224x224x3 - RGB image)
├── Conv2D(32 filters, kernel_size=(3,3), activation='relu')
├── MaxPooling2D(pool_size=(2,2))
├── Conv2D(64 filters, kernel_size=(3,3), activation='relu')
├── MaxPooling2D(pool_size=(2,2))
├── Flatten()
├── Dense(128 units, activation='relu')
├── Dropout(0.5)  # For regularization
└── Dense(3 units, activation='softmax') # Output layer for 3 classes
```

The `model.h5` file contains the pre-trained weights for this CNN model, allowing for immediate inference.

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

*   **Python**: Version 3.9 or higher. You can check your Python version using `python --version`. If not installed, download it from [python.org](https://www.python.org/downloads/).
*   **Git**: For cloning the repository.

### Setup Steps

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/devanantpushkar/Video_Classification.git
    cd Video_Classification
    ```

2.  **Create a virtual environment**:
    It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:

    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies**:
    All required Python packages are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    The `model.h5` file, which contains the pre-trained model, is included in the repository's root directory, so no separate download is required.

## Usage

You can use the system via its Streamlit web interface or by integrating its core functionalities into your Python scripts.

### Using the Streamlit Web App

To launch the web application for video classification:

1.  **Activate your virtual environment** (if not already active):
    *   macOS/Linux: `source venv/bin/activate`
    *   Windows: `.\venv\Scripts\activate`

2.  **Run the Streamlit app**:

    *   For **offline video uploads**:
        ```bash
        streamlit run app.py
        ```
    *   For **real-time screen monitoring and classification**:
        ```bash
        streamlit run screen_reader_app.py
        ```

    Once the command is executed, a new tab will open in your web browser displaying the Streamlit application. Follow the on-screen instructions to upload a video or start screen monitoring.

### Programmatic Usage

The core classification logic can also be used directly in your Python scripts. Here's an example demonstrating how you might use the `prediction.py` module to classify a video file:

```python
import os
from prediction import VideoClassifier

if __name__ == "__main__":
    # Ensure the model path is correct
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure 'model.h5' is in the root directory of the project.")
    else:
        # Initialize the VideoClassifier
        # (This assumes VideoClassifier takes a model path and possibly other configs)
        classifier = VideoClassifier(model_path=model_path)

        # Example: Classify an offline video
        video_file_path = 'path/to/your/video.mp4' # Replace with your video file
        if os.path.exists(video_file_path):
            print(f"Classifying video: {video_file_path}")
            # The exact method might vary (e.g., classify_file, process_video)
            # This is a hypothetical example based on common patterns.
            try:
                results = classifier.classify_video_from_path(video_file_path)
                print("\n--- Classification Results ---")
                print(f"Video: {video_file_path}")
                print(f"Predicted Label: {results['label']}")
                print(f"Confidence Scores: {results['scores']}")
                # You might get more detailed results depending on implementation
            except Exception as e:
                print(f"An error occurred during classification: {e}")
        else:
            print(f"Error: Video file not found at {video_file_path}")
            print("Please provide a valid path to a video file for programmatic classification.")

        # Example for real-time (more complex, involves capturing frames)
        # For real-time screen capture, you would typically use `screen_reader_app.py`
        # as it handles the MSS integration and continuous processing.
        # Programmatic real-time usage would involve directly using MSS and feeding frames.
        # This is typically handled by the Streamlit app for user convenience.
```

*Note: The exact method names (`classify_video_from_path`, `results['label']`, etc.) are illustrative. Please refer to the source code (`prediction.py`, `main.py`) for precise API details.*

## Project Structure

The repository is organized as follows:

```
Video_Classification/
├── .gitignore               # Specifies intentionally untracked files to ignore
├── .python-version          # Specifies the Python version used (e.g., pyenv)
├── README.md                # This README file
├── app.py                   # Streamlit application for offline video uploads
├── framerate.py             # Utility for handling frame rate related operations (e.g., frame skipping)
├── keyframe.py              # Module for keyframe extraction logic (e.g., I-frame detection)
├── main.py                  # Main entry point or core logic for video processing/classification
├── model.h5                 # Pre-trained deep learning model (CNN weights)
├── model_utils.py           # Utilities for loading, preprocessing, and managing the DL model
├── prediction.py            # Contains the core logic for running predictions on frames/videos
├── requirements.txt         # List of Python dependencies
├── screen_reader_app.py     # Streamlit application for real-time screen monitoring and classification
└── uv.lock                  # Lock file for uv dependency manager (if used)
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to good coding practices and includes relevant documentation.

## License

This project is currently **unlicensed**. This means that by default, all rights are reserved by the copyright holder (devanantpushkar), and you may not use, distribute, or modify this software without explicit permission.

If you intend to use this
