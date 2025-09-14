from flask import Flask, render_template, Response, jsonify, request
import threading
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time

app = Flask(__name__)

# Global variables for data storage
data_file = r'D:\peer\kvcet_vehicle\data_log\detected_plates.csv'
plate_data = pd.DataFrame(columns=['Timestamp', 'License_Plate', 'Detection_Confidence', 'OCR_Confidence'])
detected_plates = set()  # To track unique plates
lock = threading.Lock()

# Initialize the YOLO model and PaddleOCR
MODEL_PATH = r"D:\peer\kvcet_vehicle\model\yolov11_trained_model.pt"
VIDEO_PATH = r"D:\peer\kvcet_vehicle\sample_video\a.mp4"
model = YOLO(MODEL_PATH)
reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False, rec_algorithm='SVTR_LCNet')

# Control flags
video_active = False
video_paused = False
detection_threshold = 0.50  # Default detection confidence threshold
ocr_threshold = 0.40  # Default OCR confidence threshold

def process_plate_image(plate_image):
    """Extract text from license plate image with PaddleOCR"""
    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, searchWindowSize=21, templateWindowSize=7)

        results = reader.ocr(gray, cls=False)

        if results is not None and len(results) > 0:
            all_results = []
            for result in results[0]:
                if len(result) >= 2:
                    text = result[1][0]
                    confidence = round(float(result[1][1]), 2)  # Round to 2 decimal places

                    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()

                    if len(cleaned_text) >= 4 and len(cleaned_text) <= 8:
                        has_letter = any(c.isalpha() for c in cleaned_text)
                        has_number = any(c.isdigit() for c in cleaned_text)

                        if has_letter and has_number:
                            all_results.append((cleaned_text, confidence))

            if all_results:
                best_text, best_conf = max(all_results, key=lambda x: x[1])
                if best_conf >= ocr_threshold:
                    return best_text, best_conf

        return "", 0.0
    except Exception as e:
        print(f"Error in plate processing: {str(e)}")
        return "", 0.0

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    global detection_threshold, ocr_threshold
    try:
        detection_threshold = round(float(request.form['detection_threshold']), 2)
        ocr_threshold = round(float(request.form['ocr_threshold']), 2)
        return jsonify(success=True)
    except ValueError:
        return jsonify(success=False, error="Invalid threshold values"), 400

def generate_video_feed():
    """Video feed generator for the Flask app using OpenCV."""
    global video_active, video_paused
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    while True:
        if not video_active:
            break
            
        if video_paused:
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run YOLO detection
        results = model(frame, conf=detection_threshold)
        
        for result in results[0]:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Get coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence >= detection_threshold:
                    # Extract plate region
                    plate_region = frame[y1:y2, x1:x2]
                    
                    if plate_region.size > 0:
                        # Process the plate with OCR
                        plate_text, ocr_confidence = process_plate_image(plate_region)
                        
                        if plate_text and ocr_confidence >= ocr_threshold:
                            # Only add if it's a new plate or hasn't been seen in the last minute
                            current_time = datetime.now()
                            plate_key = f"{plate_text}_{current_time.strftime('%Y%m%d%H%M')}"
                            
                            if plate_key not in detected_plates:
                                detected_plates.add(plate_key)
                                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                
                                with lock:
                                    plate_data.loc[len(plate_data)] = [
                                        timestamp,
                                        plate_text,
                                        confidence,
                                        ocr_confidence
                                    ]
                                
                                # Clean up old plates (older than 1 minute)
                                old_time = (current_time.minute - 1) % 60
                                old_plates = {p for p in detected_plates 
                                            if int(p.split('_')[1][10:12]) == old_time}
                                detected_plates.difference_update(old_plates)

        # Encode the frame as JPEG and yield it
        ret, jpeg = cv2.imencode('.jpg', frame)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index_old.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video', methods=['POST'])
def start_video():
    global video_active
    video_active = True
    return jsonify(success=True)

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_active
    video_active = False
    return jsonify(success=True)

@app.route('/pause_video', methods=['POST'])
def pause_video():
    global video_paused
    video_paused = not video_paused
    return jsonify(success=True)

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global plate_data, detected_plates
    with lock:
        plate_data = pd.DataFrame(columns=['Timestamp', 'License_Plate', 'Detection_Confidence', 'OCR_Confidence'])
        detected_plates.clear()
    return jsonify({'message': 'Data reset successfully'})

@app.route('/get_latest_data')
def get_latest_data():
    with lock:
        return jsonify(plate_data.to_dict('records'))

import socket
from flask import Flask

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

"""if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
"""

if __name__ == '__main__':
    host_ip = get_ip()
    print(f"Running on http://{host_ip}:5000")
    app.run(host=host_ip, port=5000, debug=True)

