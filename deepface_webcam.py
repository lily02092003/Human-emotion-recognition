import cv2
from deepface import DeepFace
import os
import uuid


def detect_emotions_with_deepface(face_roi):
    # Save the face region as an image temporarily
    temp_image_path = f'temp_faces/{str(uuid.uuid4())}.jpg'
    cv2.imwrite(temp_image_path, face_roi)

    # Load the DeepFace emotion recognition model
    emotion_model = DeepFace.build_model('VGG-Face')

    # Perform emotion detection using DeepFace
    result = DeepFace.analyze(img_path=temp_image_path, actions=['emotion'],enforce_detection=False)

    # Delete the temporary image
    os.remove(temp_image_path)

    # Access the detected emotions from the result
    detected_emotions = result[0]['emotion']

    # Find the dominant emotion (the one with the highest confidence)
    dominant_emotion = max(detected_emotions, key=detected_emotions.get)
    confidence = detected_emotions[dominant_emotion]
    return dominant_emotion, confidence
