# American Sign Language Prediction Model 🤟💬

This project leverages YOLOv8 (PyTorch) to detect sign language gestures from images and videos. It predicts the corresponding words or phrases from the gestures, and the model is specifically designed to only work between 6 PM and 10 PM.

## Features ✨

Detects and predicts sign language gestures from images and videos 📸🎥.

Recognizes sign language phrases like "How are you?" or "What is your name?" 👋🤖.

Time-based prediction: The model only predicts between 6 PM and 10 PM 🕕.

JSON responses for output prediction data 📄.

## Technologies Used 🔧

YOLOv8 (PyTorch): For gesture detection in images and videos.

TensorFlow: For building the sign language recognition model.

OpenCV: For image and video processing.

Streamlit: For creating the web interface.

JSON: For providing structured response data.

## Installation ⚙️
### Clone the repository:

git clone https://github.com/Hariarul/ameRican-singLanguage

### Install dependencies:

pip install -r requirements.txt

Download pre-trained YOLOv8 and custom-trained sign language models.

### Run the application:

streamlit run ASL.py

## How It Works 🎬

Image Upload: Upload an image of a sign language gesture, and the model predicts the word or phrase.

Video Upload: Upload a video, and the model predicts the corresponding sign language phrase.

Time-based Prediction: The model will only predict between 6 PM and 10 PM. If the upload occurs outside this time, the model will not process the image or video.

## Example Use Case 🧑‍🤝‍🧑

Input Video/Image: Sign language for "How are you?"

Predicted Output: JSON response { "prediction": "What is your name?"}

## Results 📊
Accuracy: The model achieves 60%+ accuracy on a diverse dataset of sign language images and videos.

Speed: The model can process 15-20 frames per second in real-time video streams.

Time-based Prediction: The time-based functionality works correctly, blocking any uploads before or after 6 PM - 10 PM.

Example JSON responses include:

json
{
  "prediction": "What is your name?",
}
