import sys, os, json, csv, glob, zipfile, cv2, numpy as np, pandas as pd
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QHBoxLayout, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QDialog,
    QGroupBox, QFormLayout, QDialogButtonBox, QMenuBar, QMenu, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QLineEdit, QHeaderView, QSplitter, QDateEdit, QGridLayout, QComboBox,
    QSizePolicy
)
from PySide6.QtCore import QTimer, Qt, QDate, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen

from ultralytics import YOLO
from paddleocr import PaddleOCR

# Set the data log directory and ensure it exists.
DATA_LOG_DIR = r"D:\peer\kvcet_vehicle\data_log"
os.makedirs(DATA_LOG_DIR, exist_ok=True)

# File paths for settings and ROI saved in the data log folder.
ROI_SETTINGS_FILE = os.path.join(DATA_LOG_DIR, "roi_settings.json")
SETTINGS_FILE = os.path.join(DATA_LOG_DIR, "settings.json")

########################################################################
# VideoLabel: Supports freeform polyline ROI selection.
# Left-click to add a point; double-click to finish the polygon.
########################################################################
class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_points = []      # List of QPoint
        self.poly_finished = False
        self.editable = False     # When True, user can add/edit ROI
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.editable and event.button() == Qt.LeftButton:
            self.roi_points.append(event.pos())
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.editable:
            if len(self.roi_points) >= 3:  # Need at least 3 points to form a polygon.
                self.poly_finished = True
                self.editable = False  # Stop further editing.
                self.update()
        else:
            super().mouseDoubleClickEvent(event)

    def reset_roi(self):
        self.roi_points = []
        self.poly_finished = False
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.roi_points:
            painter = QPainter(self)
            pen_color = Qt.red if not self.poly_finished else Qt.blue
            painter.setPen(QPen(pen_color, 2, Qt.SolidLine))
            for i in range(len(self.roi_points) - 1):
                painter.drawLine(self.roi_points[i], self.roi_points[i+1])
            if self.poly_finished and len(self.roi_points) > 2:
                painter.drawLine(self.roi_points[-1], self.roi_points[0])
            painter.end()

########################################################################
# SettingsDialog: Allows threshold settings and ROI adjustments.
########################################################################
class SettingsDialog(QDialog):
    def __init__(self, parent=None, plate_conf_threshold=0.4, ocr_conf_threshold=0.4, video_label=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.video_label = video_label

        thresh_group = QGroupBox("Threshold Settings")
        form_layout = QFormLayout()
        self.spin_plate = QDoubleSpinBox()
        self.spin_plate.setRange(0.0, 1.0)
        self.spin_plate.setSingleStep(0.01)
        self.spin_plate.setValue(plate_conf_threshold)
        self.spin_ocr = QDoubleSpinBox()
        self.spin_ocr.setRange(0.0, 1.0)
        self.spin_ocr.setSingleStep(0.01)
        self.spin_ocr.setValue(ocr_conf_threshold)
        form_layout.addRow("Plate Conf. Threshold:", self.spin_plate)
        form_layout.addRow("OCR Conf. Threshold:", self.spin_ocr)
        thresh_group.setLayout(form_layout)

        roi_group = QGroupBox("ROI Settings")
        roi_layout = QVBoxLayout()
        roi_explanation = QLabel("Click to add ROI points; double-click to finish.")
        roi_explanation.setWordWrap(True)
        roi_explanation.setStyleSheet("font-size: 9pt;")
        roi_layout.addWidget(roi_explanation)
        button_layout = QHBoxLayout()
        self.btn_adjust_roi = QPushButton("Adjust ROI")
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_save_roi = QPushButton("Save ROI")
        button_layout.addWidget(self.btn_adjust_roi)
        button_layout.addWidget(self.btn_apply_roi)
        button_layout.addWidget(self.btn_save_roi)
        roi_layout.addLayout(button_layout)
        roi_group.setLayout(roi_layout)

        self.btn_adjust_roi.clicked.connect(self.enable_roi_adjustment)
        self.btn_apply_roi.clicked.connect(self.apply_roi)
        self.btn_save_roi.clicked.connect(self.save_roi)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(thresh_group)
        main_layout.addWidget(roi_group)
        main_layout.addWidget(buttons)
        self.setLayout(main_layout)

    def enable_roi_adjustment(self):
        if self.video_label:
            self.video_label.reset_roi()
            self.video_label.editable = True

    def apply_roi(self):
        if self.video_label:
            self.video_label.editable = False

    def save_roi(self):
        if self.video_label and self.video_label.roi_points and self.video_label.poly_finished:
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            norm_points = [{"x": pt.x() / label_width, "y": pt.y() / label_height} for pt in self.video_label.roi_points]
            try:
                with open(ROI_SETTINGS_FILE, "w") as f:
                    json.dump(norm_points, f)
                QMessageBox.information(self, "ROI", "ROI saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "ROI", f"Error saving ROI: {e}")
        else:
            QMessageBox.warning(self, "ROI", "No finished ROI defined to save.")

    def get_thresholds(self):
        return self.spin_plate.value(), self.spin_ocr.value()

########################################################################
# DataViewDialog: Displays detection logs and supports CSV export.
########################################################################
class DataViewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data View")
        self.resize(900, 600)
        
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItems(["Today", "Entire", "Date Range"])
        self.filter_mode_combo.currentTextChanged.connect(self.filter_mode_changed)
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search by Plate Number...")
        self.search_edit.textChanged.connect(self.filter_table)
        
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate())
        self.start_date_edit.setEnabled(False)
        self.end_date_edit.setEnabled(False)
        
        self.export_selected_button = QPushButton("Export Selected CSV")
        self.export_selected_button.clicked.connect(self.export_to_excel)
        self.export_all_button = QPushButton("Export Merged CSV (Range)")
        self.export_all_button.clicked.connect(self.export_all_csv)
        self.export_zip_button = QPushButton("Export All as ZIP")
        self.export_zip_button.clicked.connect(self.export_all_as_zip)
        
        controls_layout = QGridLayout()
        controls_layout.addWidget(QLabel("Filter Mode:"), 0, 0)
        controls_layout.addWidget(self.filter_mode_combo, 0, 1)
        controls_layout.addWidget(QLabel("Plate Search:"), 1, 0)
        controls_layout.addWidget(self.search_edit, 1, 1)
        controls_layout.addWidget(QLabel("Start Date:"), 2, 0)
        controls_layout.addWidget(self.start_date_edit, 2, 1)
        controls_layout.addWidget(QLabel("End Date:"), 3, 0)
        controls_layout.addWidget(self.end_date_edit, 3, 1)
        controls_layout.addWidget(self.export_selected_button, 4, 0)
        controls_layout.addWidget(self.export_all_button, 4, 1)
        controls_layout.addWidget(self.export_zip_button, 5, 0, 1, 2)
        
        self.table_data = QTableWidget()
        self.table_data.setColumnCount(5)
        self.table_data.setHorizontalHeaderLabels(["Timestamp", "Plate ID", "Plate Number", "OCR Conf", "Plate Conf"])
        self.table_data.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        right_layout = QVBoxLayout()
        right_layout.addLayout(controls_layout)
        right_layout.addWidget(self.table_data)
        
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        self.list_files = QListWidget()
        self.list_files.itemClicked.connect(self.load_csv_data)
        left_layout.addWidget(self.list_files)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 700])
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        self.data = pd.DataFrame()
        self.load_csv_list()

    def filter_mode_changed(self, text):
        if text == "Today":
            today = QDate.currentDate()
            self.start_date_edit.setDate(today)
            self.end_date_edit.setDate(today)
            self.start_date_edit.setEnabled(False)
            self.end_date_edit.setEnabled(False)
        elif text == "Entire":
            self.start_date_edit.setEnabled(False)
            self.end_date_edit.setEnabled(False)
        elif text == "Date Range":
            self.start_date_edit.setEnabled(True)
            self.end_date_edit.setEnabled(True)
        self.load_csv_list()

    def load_csv_list(self):
        self.list_files.clear()
        files = glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv"))
        mode = self.filter_mode_combo.currentText()
        if mode == "Today":
            today_str = datetime.now().strftime("%Y-%m-%d")
            files = [f for f in files if today_str in f]
        files.sort(reverse=True)
        if files:
            for f in files:
                filename = os.path.basename(f)
                date_str = filename.replace("detections_", "").replace(".csv", "")
                item = QListWidgetItem(date_str)
                item.setData(Qt.UserRole, f)
                self.list_files.addItem(item)
        else:
            self.list_files.addItem("No CSV files found")

    def load_csv_data(self, item):
        file_path = item.data(Qt.UserRole)
        if not file_path:
            file_path = item.text()
        try:
            self.data = pd.read_csv(file_path)
            self.populate_table(self.data)
        except Exception as e:
            QMessageBox.warning(self, "Data View", f"Error loading CSV: {e}")

    def populate_table(self, df):
        self.table_data.clearContents()
        self.table_data.setRowCount(len(df))
        for row in range(len(df)):
            for col in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iat[row, col]))
                self.table_data.setItem(row, col, item)

    def filter_table(self, text):
        if not self.data.empty:
            filtered = self.data[self.data["Plate Number"].str.contains(text, case=False, na=False)]
            self.populate_table(filtered)

    def export_to_excel(self):
        if not self.data.empty:
            save_path, _ = QFileDialog.getSaveFileName(self, "Export Selected CSV to Excel", "", "Excel Files (*.xlsx)")
            if save_path:
                try:
                    self.data.to_excel(save_path, index=False)
                    QMessageBox.information(self, "Export", "Data exported successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Export", f"Error exporting data: {e}")
        else:
            QMessageBox.warning(self, "Export", "No data to export.")

    def export_all_csv(self):
        mode = self.filter_mode_combo.currentText()
        if mode == "Entire":
            selected_files = glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv"))
        elif mode == "Today":
            today_str = datetime.now().strftime("%Y-%m-%d")
            selected_files = [f for f in glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv")) if today_str in f]
        elif mode == "Date Range":
            start_date = self.start_date_edit.date().toPython()
            end_date = self.end_date_edit.date().toPython()
            selected_files = []
            for f in glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv")):
                try:
                    basename = os.path.basename(f)
                    date_str = basename.replace("detections_", "").replace(".csv", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if start_date <= file_date <= end_date:
                        selected_files.append(f)
                except Exception as e:
                    print(f"Error processing file {f}: {e}")
        merged_df = pd.DataFrame()
        for f in selected_files:
            try:
                df = pd.read_csv(f)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error processing file {f}: {e}")
        if not merged_df.empty:
            save_path, _ = QFileDialog.getSaveFileName(self, "Export Merged Data to Excel", "", "Excel Files (*.xlsx)")
            if save_path:
                try:
                    merged_df.to_excel(save_path, index=False)
                    QMessageBox.information(self, "Export", "Merged data exported successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Export", f"Error exporting merged data: {e}")
        else:
            QMessageBox.warning(self, "Export", "No data found for the selected filter.")

    def export_all_as_zip(self):
        mode = self.filter_mode_combo.currentText()
        if mode == "Entire":
            selected_files = glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv"))
        elif mode == "Today":
            today_str = datetime.now().strftime("%Y-%m-%d")
            selected_files = [f for f in glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv")) if today_str in f]
        elif mode == "Date Range":
            start_date = self.start_date_edit.date().toPython()
            end_date = self.end_date_edit.date().toPython()
            selected_files = []
            for f in glob.glob(os.path.join(DATA_LOG_DIR, "detections_*.csv")):
                try:
                    basename = os.path.basename(f)
                    date_str = basename.replace("detections_", "").replace(".csv", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if start_date <= file_date <= end_date:
                        selected_files.append(f)
                except Exception as e:
                    print(f"Error processing file {f}: {e}")
        if selected_files:
            save_path, _ = QFileDialog.getSaveFileName(self, "Export CSV Files as ZIP", "", "Zip Files (*.zip)")
            if save_path:
                try:
                    with zipfile.ZipFile(save_path, 'w') as zf:
                        for file in selected_files:
                            zf.write(file, os.path.basename(file))
                    QMessageBox.information(self, "Export", "CSV files exported as ZIP successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Export", f"Error exporting ZIP: {e}")
        else:
            QMessageBox.warning(self, "Export", "No CSV files found for the selected filter.")

########################################################################
# MainWindow: Main application window for video detection.
# It features the video feed, detection log, and data loading.
# This version integrates a fixed polygon area check for OCR processing.
########################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KCvet-ANPR")
        
        # Define a fixed area (from your reference) if no ROI is set.
        self.fixed_area = [(27, 417), (16, 456), (1015, 451), (992, 417)]
        
        # Top header with app name and Reload button.
        heading_label = QLabel("KCvet-ANPR")
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setStyleSheet("font-size: 24pt; font-weight: bold;")
        self.reload_button = QPushButton("Reload")
        self.reload_button.setFixedWidth(100)
        self.reload_button.clicked.connect(self.reload_app)
        header_layout = QHBoxLayout()
        header_layout.addWidget(heading_label)
        header_layout.addStretch()
        header_layout.addWidget(self.reload_button)
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        
        # Left Panel: Video Feed.
        self.btn_web = QPushButton("Web")
        self.btn_upload = QPushButton("Upload Video")
        self.btn_web.clicked.connect(self.start_webcam)
        self.btn_upload.clicked.connect(self.upload_video)
        video_btn_layout = QHBoxLayout()
        video_btn_layout.addWidget(self.btn_web)
        video_btn_layout.addWidget(self.btn_upload)
        self.video_label = VideoLabel("Video Feed")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout = QVBoxLayout()
        left_layout.addLayout(video_btn_layout)
        left_layout.addWidget(self.video_label)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        # Right Panel: Detection Log and Reset Data button.
        self.table_detections = QTableWidget()
        self.table_detections.setColumnCount(5)
        self.table_detections.setHorizontalHeaderLabels(
            ["Plate ID", "Plate Number", "OCR Confidence", "Plate Confidence", "Timestamp"]
        )
        self.table_detections.setMinimumSize(500, 600)
        self.reset_button = QPushButton("Reset Data")
        self.reset_button.clicked.connect(self.reset_data)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.table_detections)
        right_layout.addWidget(self.reset_button)
        about_label = QLabel("Created by durgaram, ADS Department")
        about_label.setAlignment(Qt.AlignCenter)
        about_label.setStyleSheet("font-style: italic; font-size: 10pt; margin-top: 10px;")
        right_layout.addWidget(about_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([800, 600])
        
        central_layout = QVBoxLayout()
        central_layout.addWidget(header_widget)
        central_layout.addWidget(main_splitter)
        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        
        # Menu Bar.
        menu_bar = QMenuBar(self)
        settings_menu = QMenu("Settings", self)
        settings_action = settings_menu.addAction("Show Settings")
        settings_action.triggered.connect(self.open_settings_dialog)
        menu_bar.addMenu(settings_menu)
        data_menu = QMenu("Data", self)
        data_view_action = data_menu.addAction("Data View")
        data_view_action.triggered.connect(self.open_data_view)
        menu_bar.addMenu(data_menu)
        self.setMenuBar(menu_bar)
        
        # Initialize Video Capture, Timer, Models, and Settings.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        
        # Load YOLO model and PaddleOCR.
        self.model = YOLO("model/best.pt")
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
        self.ocr_reader = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_paddle_gpu, show_log=False, rec_algorithm="SVTR_LCNet")
        self.load_app_settings()
        
        self.last_detection_times = {}  # plate_text -> datetime
        self.last_detection_ids = {}      # plate_text -> detection id
        self.detection_interval = 60      # seconds
        self.plate_id_counter = 0
        
        self.apply_saved_roi()
        self.load_existing_detection_data()  # Load today's CSV data into the detection log

    def load_existing_detection_data(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DATA_LOG_DIR, f"detections_{date_str}.csv")
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                for index, row in df.iterrows():
                    row_position = self.table_detections.rowCount()
                    self.table_detections.insertRow(row_position)
                    self.table_detections.setItem(row_position, 0, QTableWidgetItem(str(row["Plate ID"])))
                    self.table_detections.setItem(row_position, 1, QTableWidgetItem(row["Plate Number"]))
                    self.table_detections.setItem(row_position, 2, QTableWidgetItem(str(row["OCR Confidence"])))
                    self.table_detections.setItem(row_position, 3, QTableWidgetItem(str(row["Plate Confidence"])))
                    self.table_detections.setItem(row_position, 4, QTableWidgetItem(row["Timestamp"]))
            except Exception as e:
                print("Error loading detection data:", e)

    def reload_app(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.apply_saved_roi()
        self.timer.start(30)
        
    def open_settings_dialog(self):
        dlg = SettingsDialog(
            parent=self,
            plate_conf_threshold=self.plate_conf_threshold,
            ocr_conf_threshold=self.ocr_conf_threshold,
            video_label=self.video_label
        )
        if dlg.exec():
            new_plate, new_ocr = dlg.get_thresholds()
            self.plate_conf_threshold = new_plate
            self.ocr_conf_threshold = new_ocr
            settings = {
                "plate_confidence_threshold": self.plate_conf_threshold,
                "ocr_confidence_threshold": self.ocr_conf_threshold
            }
            try:
                with open(SETTINGS_FILE, "w") as f:
                    json.dump(settings, f)
            except Exception as e:
                QMessageBox.warning(self, "Settings", f"Error saving settings: {e}")
        
    def open_data_view(self):
        self.data_view_dialog = DataViewDialog(self)
        self.data_view_dialog.show()
        
    def load_app_settings(self):
        default_settings = {"plate_confidence_threshold": 0.4, "ocr_confidence_threshold": 0.4}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
            except Exception as e:
                print(f"Error loading settings: {e}")
                settings = default_settings
        else:
            settings = default_settings
        self.plate_conf_threshold = settings.get("plate_confidence_threshold", 0.4)
        self.ocr_conf_threshold = settings.get("ocr_confidence_threshold", 0.4)
        
    def apply_saved_roi(self):
        if os.path.exists(ROI_SETTINGS_FILE):
            try:
                with open(ROI_SETTINGS_FILE, "r") as f:
                    roi_data = json.load(f)
                if isinstance(roi_data, list):
                    label_width = self.video_label.width()
                    label_height = self.video_label.height()
                    self.video_label.roi_points = [QPoint(int(pt["x"] * label_width), int(pt["y"] * label_height)) for pt in roi_data]
                    self.video_label.poly_finished = True
                self.video_label.editable = False
            except Exception as e:
                print(f"Error applying saved ROI: {e}")
                
    def start_webcam(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.apply_saved_roi()
        self.timer.start(30)
        
    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_name)
            self.apply_saved_roi()
            self.timer.start(30)
            
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return
            processed_frame = self.process_frame(frame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            bytes_per_line = channels * width
            q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            # Scale the pixmap to fit the video label while keeping its aspect ratio.
            pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
            
    def process_plate_image(self, plate_image):
        try:
            # Convert to grayscale.
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            # Apply brightness normalization (CLAHE) for bright areas.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            # Resize for better OCR accuracy.
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            # Denoise the image.
            gray = cv2.fastNlMeansDenoising(gray, None, h=15, searchWindowSize=21, templateWindowSize=7)
            results = self.ocr_reader.ocr(gray, cls=False)
            if results is not None and len(results) > 0:
                all_results = []
                for result in results[0]:
                    if len(result) >= 2:
                        text = result[1][0]
                        confidence = round(float(result[1][1]), 2)
                        # Clean the text.
                        cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                        if 4 <= len(cleaned_text) <= 8:
                            has_letter = any(c.isalpha() for c in cleaned_text)
                            has_number = any(c.isdigit() for c in cleaned_text)
                            if has_letter and has_number:
                                all_results.append((cleaned_text, confidence))
                if all_results:
                    best_text, best_conf = max(all_results, key=lambda x: x[1])
                    if best_conf >= self.ocr_conf_threshold:
                        return best_text, best_conf
            return "", 0.0
        except Exception as e:
            print(f"Error in plate processing: {e}")
            return "", 0.0
        
    def process_frame(self, frame):
        padding = 5  # Padding to expand the detected bounding box for OCR.
        # If an ROI is defined by the user, use that ROI.
        if self.video_label.roi_points and self.video_label.poly_finished:
            points = [(pt.x(), pt.y()) for pt in self.video_label.roi_points]
            pts = np.array(points)
            x, y, w, h = cv2.boundingRect(pts)
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            frame_height, frame_width, _ = frame.shape
            scale_x = frame_width / label_width
            scale_y = frame_height / label_height
            x_frame = int(x * scale_x)
            y_frame = int(y * scale_y)
            w_frame = int(w * scale_x)
            h_frame = int(h * scale_y)
            if w_frame <= 0 or h_frame <= 0:
                return frame
            if x_frame + w_frame > frame_width:
                w_frame = frame_width - x_frame
            if y_frame + h_frame > frame_height:
                h_frame = frame_height - y_frame
            roi = frame[y_frame:y_frame+h_frame, x_frame:x_frame+w_frame].copy()
            results = self.model(roi)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                detections = results[0].boxes.data.cpu().numpy()
                for det in detections:
                    conf_val = det[4]
                    if conf_val >= self.plate_conf_threshold:
                        bx1, by1, bx2, by2, conf_val, cls = det
                        # Map detection coordinates back to original frame.
                        x_det1 = max(x_frame + int(bx1) - padding, 0)
                        y_det1 = max(y_frame + int(by1) - padding, 0)
                        x_det2 = min(x_frame + int(bx2) + padding, frame.shape[1])
                        y_det2 = min(y_frame + int(by2) + padding, frame.shape[0])
                        cv2.rectangle(frame, (x_det1, y_det1), (x_det2, y_det2), (0, 255, 0), 2)
                        roi_plate = frame[y_det1:y_det2, x_det1:x_det2]
                        if roi_plate.size > 0:
                            plate_text, ocr_conf = self.process_plate_image(roi_plate)
                        else:
                            plate_text, ocr_conf = "", 0.0
                        if plate_text:
                            current_time = datetime.now()
                            if (plate_text in self.last_detection_times and 
                                (current_time - self.last_detection_times[plate_text]).total_seconds() < self.detection_interval):
                                detection_id = self.last_detection_ids[plate_text]
                            else:
                                self.plate_id_counter += 1
                                detection_id = self.plate_id_counter
                                self.last_detection_times[plate_text] = current_time
                                self.last_detection_ids[plate_text] = detection_id
                                self.append_detection_info(detection_id, plate_text, ocr_conf, conf_val)
                                self.log_detection(detection_id, plate_text, ocr_conf, conf_val)
                            cv2.putText(frame, f"ID: {detection_id}", (x_det1, y_det1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # If no ROI is defined, use the fixed polygon area.
            cv2.polylines(frame, [np.array(self.fixed_area, np.int32)], True, (255, 0, 0), 2)
            results = self.model(frame)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                detections = results[0].boxes.data.cpu().numpy()
                for det in detections:
                    conf_val = det[4]
                    if conf_val >= self.plate_conf_threshold:
                        bx1, by1, bx2, by2, conf_val, cls = det
                        # Compute the detection center.
                        cx = (int(bx1) + int(bx2)) // 2
                        cy = (int(by1) + int(by2)) // 2
                        # Check if the center lies inside the fixed area.
                        pt_result = cv2.pointPolygonTest(np.array(self.fixed_area, np.int32), (cx, cy), False)
                        if pt_result < 0:
                            continue
                        x_det1 = max(int(bx1) - padding, 0)
                        y_det1 = max(int(by1) - padding, 0)
                        x_det2 = min(int(bx2) + padding, frame.shape[1])
                        y_det2 = min(int(by2) + padding, frame.shape[0])
                        cv2.rectangle(frame, (x_det1, y_det1), (x_det2, y_det2), (0, 255, 0), 2)
                        roi_plate = frame[y_det1:y_det2, x_det1:x_det2]
                        if roi_plate.size > 0:
                            plate_text, ocr_conf = self.process_plate_image(roi_plate)
                        else:
                            plate_text, ocr_conf = "", 0.0
                        if plate_text:
                            current_time = datetime.now()
                            if (plate_text in self.last_detection_times and 
                                (current_time - self.last_detection_times[plate_text]).total_seconds() < self.detection_interval):
                                detection_id = self.last_detection_ids[plate_text]
                            else:
                                self.plate_id_counter += 1
                                detection_id = self.plate_id_counter
                                self.last_detection_times[plate_text] = current_time
                                self.last_detection_ids[plate_text] = detection_id
                                self.append_detection_info(detection_id, plate_text, ocr_conf, conf_val)
                                self.log_detection(detection_id, plate_text, ocr_conf, conf_val)
                            cv2.putText(frame, f"ID: {detection_id}", (x_det1, y_det1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def append_detection_info(self, plate_id, plate_text, ocr_conf, plate_conf):
        row_position = self.table_detections.rowCount()
        self.table_detections.insertRow(row_position)
        self.table_detections.setItem(row_position, 0, QTableWidgetItem(str(plate_id)))
        self.table_detections.setItem(row_position, 1, QTableWidgetItem(plate_text))
        self.table_detections.setItem(row_position, 2, QTableWidgetItem(f"{ocr_conf:.2f}"))
        self.table_detections.setItem(row_position, 3, QTableWidgetItem(f"{plate_conf:.2f}"))
        self.table_detections.setItem(row_position, 4, QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def log_detection(self, plate_id, plate_text, ocr_conf, plate_conf):
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DATA_LOG_DIR, f"detections_{date_str}.csv")
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Timestamp", "Plate ID", "Plate Number", "OCR Confidence", "Plate Confidence"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, plate_id, plate_text, f"{ocr_conf:.2f}", f"{plate_conf:.2f}"])

    def reset_data(self):
        self.last_detection_times.clear()
        self.last_detection_ids.clear()
        self.plate_id_counter = 0
        self.table_detections.setRowCount(0)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DATA_LOG_DIR, f"detections_{date_str}.csv")
        if os.path.isfile(filename):
            try:
                os.remove(filename)
                QMessageBox.information(self, "Reset Data", "Current day CSV data has been reset.")
            except Exception as e:
                QMessageBox.warning(self, "Reset Data", f"Error resetting CSV data: {e}")
        else:
            QMessageBox.information(self, "Reset Data", "No CSV data to reset.")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

########################################################################
# Main Application Entry Point
########################################################################
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
