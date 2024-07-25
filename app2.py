import os
import cv2
import numpy as np
import torch
import timm
from mtcnn import MTCNN
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, Response, redirect, url_for, jsonify, request, session
import threading
import yagmail
import torch.nn.functional as F
from datetime import datetime
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'supersecretkey'  

detector = MTCNN()

model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 2)  
pretrained_model_path = 'vit_teacher_inc_reduced_lr-7.pth'
if not os.path.exists(pretrained_model_path):
    raise FileNotFoundError(f"Pretrained model file not found: {pretrained_model_path}")

model = tf.keras.models.load_model('model4.h5')

cascade_path = 'Haarcascades/haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")

face_cascade = cv2.CascadeClassifier(cascade_path)

SENDER_EMAIL = 'sender_email@example.com'
APP_PASSWORD = 'sender_passcode'  
RECEIVER_EMAIL = 'receiver_email@example.com'

def send_email(subject, body):
    yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
    try:
        yag.send(to=RECEIVER_EMAIL, subject=subject, contents=body)
    except Exception as e:
        print(f"Error sending email: {e}")

camera = None
processing = False
result_data = {'result': None, 'color': None}
recorded_frames = []
email_sent = False
frame_count = 0

def start_video_capture():
    global camera
    camera_index = 2  
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not start camera at index {camera_index}. Please check if the camera is connected and accessible.")

def stop_video_capture():
    global camera
    if camera:
        camera.release()

def capture_frames(account_number):
    global camera, processing, result_data, recorded_frames, email_sent, frame_count

    frame_count = 0
    real_count = 0
    fake_count = 0
    recorded_frames = []  

    while not processing:
        if camera and camera.isOpened():
            success, frame = camera.read()
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) == 1:
                    for (x, y, w, h) in faces:
                        img_resized = cv2.resize(frame, (64, 64))
                        img_normalized = img_resized / 255.0
                        img_expanded = np.expand_dims(img_normalized, axis=0)
                        
                        predicted = model.predict(img_expanded)
                        
                        recorded_frames.append(frame.copy())
                        frame_count += 1
                        if predicted == 1:
                            real_count += 1
                        else:
                            fake_count += 1

                        if frame_count >= 10:
                            evaluate_result(real_count, frame_count, account_number)
                            processing = True
                            break

        if not processing:
            result_data['result'] = None
            result_data['color'] = None

    stop_video_capture()

def process_face(frame, x, y, w, h):
    face = frame[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    face_tensor = transform(face_pil).unsqueeze(0)
    return face_tensor

def predict_face(face_tensor):
    with torch.no_grad():
        outputs = model(face_tensor)
        print("Output (logits)", outputs)
        probabilities = F.softmax(outputs, dim=1)
        print("Output (probabilities)", probabilities)
    return probabilities.argmax(dim=1).item()

def evaluate_result(real_count, frame_count, account_number):
    global result_data, email_sent, recorded_frames

    if (real_count / frame_count) >= 0.75:
        result_data['result'] = 'ACCESS GRANTED'
        result_data['color'] = 'green'
    else:
        result_data['result'] = 'ACCESS DENIED'
        result_data['color'] = 'black'
        
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{account_number}_{current_time}.avi"
        
        if not email_sent:
            
            print("Email SENT")
            email_sent = True
        
        if recorded_frames:
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  
            height, width, layers = recorded_frames[0].shape
            
            video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

            for frame in recorded_frames:
                video_writer.write(frame)

            video_writer.release()
            print(f"Video saved as '{video_filename}'")

def gen_frames():
    global camera, result_data, processing

    if camera is None or not camera.isOpened():
        start_video_capture()

    try:
        while not processing:
            if camera and camera.isOpened():
                success, frame = camera.read()
                if not success:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        stop_video_capture()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global camera, email_sent, processing, result_data
    account_number = request.form['account_number']
    session['account_number'] = account_number  
    session['original_url'] = request.referrer  
    email_sent = False
    processing = False
    result_data = {'result': None, 'color': None}
    
    if camera is None or not camera.isOpened():
        start_video_capture()
        threading.Thread(target=capture_frames, args=(account_number,), daemon=True).start()
    
    return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    global result_data
    if result_data['result'] in ['ACCESS GRANTED', 'ACCESS DENIED']:
        return redirect(url_for('final_result'))
    else:
        return render_template('index2.html')

@app.route('/status')
def get_status():
    global result_data
    return jsonify(result_data)

@app.route('/video_capture')
def video_capture():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/final_result')
def final_result():
    result = result_data.get('result', 'Error')
    color = result_data.get('color', 'red')
    original_url = session.get('original_url', url_for('index'))  
    return render_template('stop.html', result=result, color=color, original_url=original_url)

@app.route('/redirect_back', methods=['POST'])
def redirect_back():
    original_url = session.get('original_url', url_for('index'))
    return redirect(original_url)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
