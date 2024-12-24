import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import json
import time
import cvzone
from PIL import Image

# Load YOLO model
model = YOLO(r"D:\Dataset for internships\YOLO pre trained weights\Video call ASL.pt")

# Class names for ASL detection
classNames = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Load phrase mappings from JSON file
phrase_file_path = r"C:\Users\user\Downloads\phrase_mappings.json"
try:
    with open(phrase_file_path, 'r') as file:
        phrase_mappings = json.load(file)
except FileNotFoundError:
    st.error(f"The file {phrase_file_path} was not found.")
    phrase_mappings = {}

# Function to check time-based restrictions (allowing until 2 AM)
def is_prediction_allowed():
    current_time = datetime.now().time()
    start_time = datetime.strptime("08:00:00", "%H:%M:%S").time()
    end_time = datetime.strptime("02:00:00", "%H:%M:%S").time()

    # Check if current time is between 8:00 AM and 2:00 AM the next day
    if start_time <= current_time or current_time <= end_time:
        return True
    else:
        return False

# Streamlit App Layout
st.set_page_config(page_title="ASL Detection System", layout="wide")
st.title("📹 ASL Detection System")

# Sidebar Configuration for Input Source
st.sidebar.header("Input Source Options")
# Using buttons for webcam and file upload selection
input_option = st.sidebar.radio("Select Input Source", ("Webcam", "Upload Video"))

cap = None  # Initialize cap variable

if input_option == "Webcam":
    start_button = st.sidebar.button("Start Camera")
    if start_button:
        st.sidebar.write("📹 Camera started!")
        cap = cv2.VideoCapture(0)  # Access webcam
    else:
        st.sidebar.write("Click 'Start Camera' to begin using the webcam.")

elif input_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video (Drag or Select)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        cap = cv2.VideoCapture(uploaded_file.name)  # Load the uploaded video file
        st.sidebar.write(f"Video selected: {uploaded_file.name}")
    else:
        st.sidebar.write("Drag or select a video file to upload.")

# Initialize placeholders
frame_placeholder = st.empty()
current_sign_placeholder = st.empty()
detected_words_placeholder = st.empty()
response_message_placeholder = st.empty()

# Detection Loop
if cap:
    word_buffer = ""
    valid_words = []
    current_sign = None
    frame_count = 0
    confirmation_threshold = 15
    confidence_threshold = 0.6
    response_display_time = 5
    last_response_time = None
    response_message = ""
    last_detected_time = time.time()
    letter_spacing_threshold = 5.0
    word_confirmation_timeout = 15.0

    # Add Reset Button
    reset_button = st.sidebar.button("Reset Detection")
    if reset_button:
        word_buffer = ""
        valid_words = []
        current_sign = None
        frame_count = 0
        last_detected_time = time.time()
        response_message = ""
        last_response_time = None
        st.sidebar.write("Detection has been reset.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.warning("Video source ended or not available.")
            break

        imgOutput = frame.copy()
        detected_letter = None

        # Check if predictions are allowed based on time
        if is_prediction_allowed():
            frame_resized = cv2.resize(frame, (1280, 720))
            results = model(frame_resized, stream=True)

            max_confidence_box = None
            max_confidence = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > max_confidence:
                        max_confidence = conf
                        max_confidence_box = box

            if max_confidence_box and max_confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, max_confidence_box.xyxy[0])
                cls = int(max_confidence_box.cls[0])
                detected_letter = classNames[cls]

                if time.time() - last_detected_time > letter_spacing_threshold:
                    word_buffer += " "

                last_detected_time = time.time()

                if detected_letter == current_sign:
                    frame_count += 1
                else:
                    frame_count = 0
                    current_sign = detected_letter

                if frame_count >= confirmation_threshold:
                    word_buffer += detected_letter
                    frame_count = 0

                cvzone.cornerRect(imgOutput, (x1, y1, x2 - x1, y2 - y1), l=25, colorR=(255, 0, 255), colorC=(255, 255, 255), t=5, rt=0)
                cv2.putText(
                    imgOutput,
                    f'{detected_letter} {max_confidence:.2f}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

        # If predictions are not allowed, show message on screen
        if not is_prediction_allowed():
            current_sign_placeholder.markdown("### 🚫 Predictions are not allowed at this time. Allowed from 8 AM to 2 AM.")

        if time.time() - last_detected_time > word_confirmation_timeout and word_buffer:
            valid_words.append(word_buffer.strip())
            full_phrase = " ".join(valid_words).lower()
            response_message = phrase_mappings.get(full_phrase, "No predefined response found.")
            last_response_time = time.time()
            word_buffer = ""

        # Update UI
        current_sign_placeholder.markdown(f"### Current Sign: `{current_sign or '...'}`")
        detected_words_placeholder.markdown(f"### Detected Words: `{word_buffer.strip()}`")
        if response_message and last_response_time:
            if time.time() - last_response_time <= response_display_time:
                response_message_placeholder.markdown(f"### Response: `{response_message}`")
            else:
                response_message = ""

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_pil, caption="Live ASL Detection", use_column_width=True)

    cap.release()
else:
    st.error("No video source selected or available.")
