from flask import Flask, render_template, Response, jsonify, request
import threading
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time
import os
import json
from pathlib import Path
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from queue import Queue
import concurrent.futures
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ROISettings:
    enabled: bool = False
    x1: int = 100
    y1: int = 100
    x2: int = 800
    y2: int = 600
    detection_threshold: float = 0.50
    ocr_threshold: float = 0.90

class PlateDetectionSystem:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Initialize paths
        self.base_path = Path('D:/peer/kvcet_vehicle')
        self.data_log_path = self.base_path / 'data_log'
        self.data_log_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.model = YOLO(str(self.base_path / 'model/last.pt'))
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
        except Exception:
            pass
        try:
            import paddle
            use_paddle_gpu = hasattr(paddle, 'device') and paddle.device.is_compiled_with_cuda()
        except Exception:
            use_paddle_gpu = False
        self.reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_paddle_gpu, show_log=False, rec_algorithm='SVTR_LCNet')
        
        # Initialize data storage
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.data_file = self.data_log_path / f'detected_plates_{self.current_date}.csv'
        self.roi_file = self.data_log_path / 'roi_settings.json'
        
        # Initialize dataframes and control variables
        self.plate_data = pd.DataFrame(columns=['Timestamp', 'License_Plate', 'Detection_Confidence', 'OCR_Confidence'])
        self.frame_queue = Queue(maxsize=30)
        self.results_queue = Queue(maxsize=30)

        # Initialize threading components
        self.lock = threading.Lock()
        self.video_active = False
        self.video_paused = False
        self.processing_thread = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Load settings
        self.roi_settings = self.load_roi_settings()
        
        # Video source
        self.video_source = str(self.base_path / 'sample_video/a.mp4')

    def update_plate_data(self, license_plate: str, ocr_confidence: float):
        with self.lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_data = pd.DataFrame([{
                'Timestamp': timestamp,
                'License_Plate': license_plate,
                'Detection_Confidence': self.roi_settings.detection_threshold,
                'OCR_Confidence': ocr_confidence
            }])
            self.plate_data = pd.concat([self.plate_data, new_data], ignore_index=True)
            self.save_daily_data()

    def load_roi_settings(self) -> ROISettings:
        if self.roi_file.exists():
            try:
                with open(self.roi_file, 'r') as f:
                    settings = json.load(f)
                return ROISettings(**settings)
            except Exception as e:
                logger.error(f"Error loading ROI settings: {e}")
        return ROISettings()

    def save_roi_settings(self):
        try:
            with open(self.roi_file, 'w') as f:
                json.dump(vars(self.roi_settings), f)
        except Exception as e:
            logger.error(f"Error saving ROI settings: {e}")

    def process_plate_image(self, plate_img: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            gray_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            ocr_results = self.reader.ocr(gray_img, cls=True)
            if ocr_results and len(ocr_results[0]) > 0:
                text, confidence = ocr_results[0][0][1][0], ocr_results[0][0][1][1]
                return text.strip(), confidence
            return None, 0.0
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return None, 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], float]:
        try:
            if self.roi_settings.enabled:
                roi = frame[self.roi_settings.y1:self.roi_settings.y2, self.roi_settings.x1:self.roi_settings.x2]
                cv2.rectangle(frame, (self.roi_settings.x1, self.roi_settings.y1), (self.roi_settings.x2, self.roi_settings.y2), (0, 255, 0), 2)
                plate_img = roi
            else:
                plate_img = frame

            results = self.model(plate_img, conf=self.roi_settings.detection_threshold)
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                confidence = float(box.conf)
                if confidence >= self.roi_settings.detection_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_roi = plate_img[y1:y2, x1:x2]
                    license_plate, ocr_conf = self.process_plate_image(plate_roi)
                    if license_plate and ocr_conf >= self.roi_settings.ocr_threshold:
                        self.update_plate_data(license_plate, ocr_conf)
                        return frame, license_plate, ocr_conf
            return frame, None, 0.0
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            return frame, None, 0.0

    def save_daily_data(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.current_date:
            with self.lock:
                if not self.plate_data.empty:
                    self.plate_data.to_csv(self.data_file, index=False)
                self.current_date = today
                self.data_file = self.data_log_path / f'detected_plates_{self.current_date}.csv'
                self.plate_data = pd.DataFrame(columns=['Timestamp', 'License_Plate', 'Detection_Confidence', 'OCR_Confidence'])
        else:
            if not self.plate_data.empty:
                self.plate_data.to_csv(self.data_file, index=False)

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/start_video', methods=['POST'])
        def start_video():
            self.video_active = True
            logger.info("Video Started")
            return jsonify(success=True)

        @self.app.route('/stop_video', methods=['POST'])
        def stop_video():
            self.video_active = False
            logger.info("Video Stopped")
            return jsonify(success=True)

        @self.app.route('/pause_video', methods=['POST'])
        def pause_video():
            self.video_paused = not self.video_paused
            return jsonify(success=True)

        @self.app.route('/reset_data', methods=['POST'])
        def reset_data():
            with self.lock:
                self.plate_data = pd.DataFrame(columns=['Timestamp', 'License_Plate', 'Detection_Confidence', 'OCR_Confidence'])
            return jsonify({'message': 'Data reset successfully'})

        @self.app.route('/get_latest_data')
        def get_latest_data():
            with self.lock:
                return jsonify(self.plate_data.to_dict('records'))

        @self.app.route('/set_roi', methods=['POST'])
        def set_roi():
            try:
                self.roi_settings = ROISettings(
                    enabled=request.form.get('enabled') == 'true',
                    x1=int(request.form.get('x1', self.roi_settings.x1)),
                    y1=int(request.form.get('y1', self.roi_settings.y1)),
                    x2=int(request.form.get('x2', self.roi_settings.x2)),
                    y2=int(request.form.get('y2', self.roi_settings.y2)),
                    detection_threshold=float(request.form.get('detection_threshold', self.roi_settings.detection_threshold)),
                    ocr_threshold=float(request.form.get('ocr_threshold', self.roi_settings.ocr_threshold))
                )
                self.save_roi_settings()
                return jsonify(success=True)
            except Exception as e:
                logger.error(f"Error setting ROI: {e}")
                return jsonify(success=False, error=str(e))

    def generate_video_feed(self):
        cap = cv2.VideoCapture(self.video_source)
        while self.video_active:
            if self.video_paused:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            processed_frame, license_plate, ocr_confidence = self.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    def cleanup(self):
        logger.info("Shutting down, saving data...")
        self.save_daily_data()

    def run(self, host='0.0.0.0', port=5000, debug=False):
        atexit.register(self.cleanup)
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    system = PlateDetectionSystem()
    system.run(debug=True)
