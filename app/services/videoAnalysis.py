import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import os

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    confidence_scores = []

    if len(faces) == 0:
        return 0.4, 0.5  # Default values for no face detected

    eye_contact_score = track_eye_contact(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            # Check if model exists in cache
            cache_dir = os.path.expanduser("~/.deepface/weights/")
            model_path = os.path.join(cache_dir, "facial_expression_model_weights.h5")

            if not os.path.exists(model_path):
                print("Downloading the model...")
                DeepFace.download_face_model("Emotion")

            # Proceed with your analysis
            analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
            emotion = analysis["dominant_emotion"]
            confidence_scores.append(emotion_confidence(emotion))
        except:
            continue

    avg_emotion_conf = np.mean(confidence_scores) if confidence_scores else 0.6
    final_confidence = (avg_emotion_conf * 0.7 + eye_contact_score * 0.3)
    return final_confidence, eye_contact_score

def track_eye_contact(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        return 1.0
    return 0.5

def emotion_confidence(emotion):
    confidence_mapping = {
        "happy": 1.0, "smile": 1.0, "neutral": 0.85, "calm": 0.9, "surprise": 0.7,
        "sad": 0.5, "fear": 0.4, "angry": 0.3, "disgust": 0.2
    }
    return confidence_mapping.get(emotion, 0.6)

def generate_feedback(avg_conf, avg_eye_contact):
    if avg_conf < 0.6 and avg_eye_contact < 0.6:
        return "You had low confidence and poor eye contact. Try to be more expressive and look into the camera."
    elif avg_conf < 0.6:
        return "You had low confidence. Try to express more positive emotions like smiling."
    elif avg_eye_contact < 0.6:
        return "You lacked eye contact. Try to look directly into the camera more often."
    else:
        return "Great job! You maintained good confidence and eye contact."

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_confidences = []
    all_eye_scores = []

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Start from first frame (index 0), then analyze every 5th frame
        if frame_index % 5 == 0:
            confidence, eye_contact_score = analyze_frame(frame)
            all_confidences.append(confidence)
            all_eye_scores.append(eye_contact_score)

        frame_index += 1

    cap.release()

    avg_conf = np.mean(all_confidences) if all_confidences else 0
    avg_eye = np.mean(all_eye_scores) if all_eye_scores else 0
    feedback = generate_feedback(avg_conf, avg_eye)

    return avg_conf, avg_eye, feedback
