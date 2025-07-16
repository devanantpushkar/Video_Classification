import os
import logging
import time
import cv2
import numpy as np
import tensorflow as tf

# Mute Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Load model
model = tf.keras.models.load_model('model.h5')
class_names = ['general', 'obscene', 'violent']

# Read frames from video file
def extract_frames_fast(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_interval = int(fps // frame_rate)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return frames

# Predict average class probabilities for a video
def average_predictions(video_path, batch_size=32):
    frames = extract_frames_fast(video_path)
    if not frames:
        return {"error": "No frames extracted"}

    frames_array = np.array(frames, dtype=np.float32)

    predictions = model.predict(frames_array, batch_size=batch_size, verbose=0)
    avg_prediction = np.mean(predictions, axis=0)

    results = {
        label: float(score)
        for label, score in zip(class_names, avg_prediction)
    }

    return results
