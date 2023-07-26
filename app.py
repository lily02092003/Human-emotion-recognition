from flask import Flask, render_template, Response
import cv2
from deepface_webcam import detect_emotions_with_deepface
import os


app = Flask(__name__)


def detect_video():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]  # Extract the face region

            # Perform emotion detection using DeepFace
            detected_emotion, confidence = detect_emotions_with_deepface(face_roi)

            # Draw bounding box and emotion label on the original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            emotion_label = f' {detected_emotion}:  {confidence:.2f}'
            cv2.putText(img, emotion_label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/webcam", methods=['GET','POST'])

def webcam():
    return render_template('custom.html')
@app.route('/video_feed')
def video_feed():
    return Response(detect_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Create a temporary folder to store the face images
    if not os.path.exists('temp_faces'):
        os.makedirs('temp_faces')

    app.run(debug=True)
