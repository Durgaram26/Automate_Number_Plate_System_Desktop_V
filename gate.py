import sys, os, json, csv, glob, zipfile, cv2, numpy as np, pandas as pd, re, time
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QHBoxLayout, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QDialog,
    QGroupBox, QFormLayout, QDialogButtonBox, QMenuBar, QMenu, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QLineEdit, QHeaderView, QSplitter, QDateEdit, QGridLayout, QComboBox,
    QSizePolicy, QToolBar, QStatusBar, QStyle, QAbstractItemView, QCheckBox, QStyleFactory, QFrame, QProgressBar, QGraphicsDropShadowEffect
)
from PySide6.QtCore import QTimer, Qt, QDate, QPoint, QSettings, QCoreApplication
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QAction, QPalette, QColor, QFont, QIcon, QLinearGradient, QBrush

from ultralytics import YOLO
from paddleocr import PaddleOCR
import serial
import serial.tools.list_ports
from typing import Optional

# Set the data log directory and ensure it exists.
DATA_LOG_DIR = r"D:\peer\kvcet_vehicle\data_log"
os.makedirs(DATA_LOG_DIR, exist_ok=True)

# File paths for settings and ROI saved in the data log folder.
ROI_SETTINGS_FILE = os.path.join(DATA_LOG_DIR, "roi_settings.json")
SETTINGS_FILE = os.path.join(DATA_LOG_DIR, "settings.json")

# ------------------------------ Gate/ESP32 Serial Settings ------------------------------
TARGET_PLATE = "HR26CQ6869"
# Optional preferred COM port name hint. Leave empty to auto-detect by USB VID/PID matching typical CP210x/CH340/FTDI
PREFERRED_COM = "COM4"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.2

def find_serial_port(preferred: str = PREFERRED_COM) -> Optional[str]:
    try:
        if preferred:
            return preferred
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Heuristic: pick first USB serial
            if ("USB" in p.description) or ("UART" in p.description) or ("CP210" in p.description) or ("CH340" in p.description) or ("FTDI" in p.description):
                return p.device
        # fallback to first available
        if ports:
            return ports[0].device
    except Exception:
        pass
    return None

class GateController:
    def __init__(self):
        self.ser = None
        self.last_open_time = 0.0
        self.min_interval_seconds = 5.0  # debounce open commands

    def connect(self):
        if self.ser and self.ser.is_open:
            return True
        port = find_serial_port()
        if not port:
            return False
        try:
            self.ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
            # give ESP32 time to reset on serial open
            time.sleep(1.0)
            return True
        except Exception:
            self.ser = None
            return False

    def send_command(self, cmd: str) -> bool:
        try:
            if not self.connect():
                return False
            payload = (cmd.strip() + "\n").encode("utf-8")
            self.ser.write(payload)
            self.ser.flush()
            # Optional: read a short response
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                # We ignore content but could log if needed
            except Exception:
                pass
            return True
        except Exception:
            return False

    def open_gate(self):
        now = time.time()
        if now - self.last_open_time < self.min_interval_seconds:
            return
        print("[Gate] Attempting OPEN command...")
        if self.send_command("OPEN"):
            self.last_open_time = now
            print("[Gate] OPEN sent")
        else:
            print("[Gate] Failed to send OPEN (no serial?)")

    def close(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass

# ------------------------------ UI Theming Helpers ------------------------------
ACCENT_DARK = "#1976d2"
ACCENT_LIGHT = "#1565c0"

DARK_STYLESHEET = f"""
QWidget {{
    background-color: #0f1115;
    color: #e0e6ef;
}}
QMenuBar, QMenu {{
    background-color: #121521;
    color: #e0e6ef;
}}
QMenu::item:selected {{
    background-color: #1c2130;
}}
QToolBar {{
    background: #121521;
    spacing: 6px;
    border: none;
}}
QStatusBar {{
    background: #121521;
    color: #cfd8dc;
}}
QPushButton {{
    background-color: #1f2937;
    color: #e5e7eb;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
}}
QPushButton:hover {{
    background-color: #273449;
    border-color: #3b82f6;
    color: #ffffff;
}}
QPushButton:pressed {{
    background-color: #0f172a;
}}
QLineEdit, QComboBox, QDateEdit {{
    background-color: #121826;
    border: 1px solid #2a3246;
    border-radius: 6px;
    padding: 6px;
    color: #e0e6ef;
}}
QTableWidget {{
    background-color: #0f1117;
    gridline-color: #2a2f3a;
    selection-background-color: #2f4f84;
    selection-color: #ffffff;
}}
QTableWidget::item {{
    padding: 6px;
}}
QHeaderView::section {{
    background-color: #141a26;
    color: #cfd8dc;
    border: 0px;
    border-bottom: 1px solid #232b3d;
    padding: 8px;
}}
QSplitter::handle {{
    background: #1a2131;
    width: 6px;
}}
QGroupBox {{
    border: 1px solid #242b3e;
    border-radius: 8px;
    margin-top: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QScrollBar:vertical {{
    background: #121826;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #3b4252;
    border-radius: 5px;
    min-height: 20px;
}}
QLabel#HeaderTitle {{
    color: #ffffff;
    background: transparent;
    padding: 0px;
    margin: 0px;
    border: none;
}}
QWidget#Card {{
    background: #111723;
    border: 1px solid #1f2636;
    border-radius: 12px;
}}
QWidget#Header {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f172a, stop:1 #1e3a8a);
    border: 1px solid #1f2636;
    border-radius: 12px;
}}
QLabel#Badge {{
    background: #3b82f6;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeSuccess {{
    background: #10b981;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeInfo {{
    background: #64748b;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeWarning {{
    background: #f59e0b;
    color: #111827;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeError {{
    background: #ef4444;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#VideoArea {{
    background: #0e1420;
    border: 1px solid #1f2636;
    border-radius: 12px;
}}
QProgressBar {{
    height: 16px;
    border: 1px solid #2a3246;
    border-radius: 6px;
    background: #121826;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: #3b82f6;
    border-radius: 6px;
}}
"""

LIGHT_STYLESHEET = f"""
QWidget {{
    background-color: #f5f7fb;
    color: #1e293b;
}}
QMenuBar, QMenu {{
    background-color: #ffffff;
    color: #1e293b;
}}
QMenu::item:selected {{
    background-color: #e8eef9;
}}
QToolBar {{
    background: #ffffff;
    spacing: 6px;
    border: none;
}}
QStatusBar {{
    background: #ffffff;
    color: #334155;
}}
QPushButton {{
    background-color: #e8eefc;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px 14px;
}}
QPushButton:hover {{
    background-color: #dbeafe;
    border-color: #3b82f6;
}}
QPushButton:pressed {{
    background-color: #cfe1ff;
}}
QLineEdit, QComboBox, QDateEdit {{
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px;
    color: #0f172a;
}}
QTableWidget {{
    background-color: #ffffff;
    gridline-color: #cbd5e1;
    selection-background-color: #bfdbfe;
    selection-color: #0f172a;
}}
QTableWidget::item {{
    padding: 6px;
}}
QHeaderView::section {{
    background-color: #eaf0fb;
    color: #0f172a;
    border: 0px;
    border-bottom: 1px solid #cbd5e1;
    padding: 8px;
}}
QSplitter::handle {{
    background: #e2e8f0;
    width: 6px;
}}
QGroupBox {{
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    margin-top: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QScrollBar:vertical {{
    background: #f1f5f9;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #94a3b8;
    border-radius: 5px;
    min-height: 20px;
}}
QLabel#HeaderTitle {{
    color: #ffffff;
    background: transparent;
    padding: 0px;
    margin: 0px;
    border: none;
}}
QWidget#Card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}}
QWidget#Header {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e3a8a, stop:1 #3b82f6);
    border: 1px solid #cbd5e1;
    border-radius: 12px;
}}
QLabel#Badge {{
    background: #3b82f6;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeSuccess {{
    background: #10b981;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeInfo {{
    background: #64748b;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeWarning {{
    background: #f59e0b;
    color: #111827;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#BadgeError {{
    background: #ef4444;
    color: white;
    padding: 2px 8px;
    border-radius: 9px;
}}
QLabel#VideoArea {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}}
QProgressBar {{
    height: 16px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    background: #f8fafc;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: #3b82f6;
    border-radius: 6px;
}}
"""

def build_palette(theme: str) -> QPalette:
    palette = QPalette()
    if theme == "dark":
        palette.setColor(QPalette.Window, QColor("#0f1115"))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor("#0f1115"))
        palette.setColor(QPalette.AlternateBase, QColor("#121521"))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor("#121521"))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor("#2f4f84"))
        palette.setColor(QPalette.HighlightedText, Qt.white)
    else:
        palette.setColor(QPalette.Window, QColor("#f5f7fb"))
        palette.setColor(QPalette.WindowText, QColor("#0f172a"))
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.AlternateBase, QColor("#f1f5f9"))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, QColor("#0f172a"))
        palette.setColor(QPalette.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ButtonText, QColor("#0f172a"))
        palette.setColor(QPalette.Highlight, QColor("#93c5fd"))
        palette.setColor(QPalette.HighlightedText, QColor("#0f172a"))
    return palette

def apply_modern_style(theme: str = "dark") -> None:
    app = QApplication.instance()
    if app is None:
        return
    try:
        app.setStyle(QStyleFactory.create("Fusion"))
    except Exception:
        pass
    app.setPalette(build_palette(theme))
    app.setStyleSheet(DARK_STYLESHEET if theme == "dark" else LIGHT_STYLESHEET)

def apply_drop_shadow(widget: QWidget, radius: int = 18, dx: int = 0, dy: int = 6, color: QColor = QColor(0, 0, 0, 140)) -> None:
    try:
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(radius)
        effect.setOffset(dx, dy)
        effect.setColor(color)
        widget.setGraphicsEffect(effect)
    except Exception:
        pass

def create_app_icon() -> QIcon:
    pm = QPixmap(128, 128)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    grad = QLinearGradient(0, 0, 128, 128)
    grad.setColorAt(0.0, QColor("#1e3c72"))
    grad.setColorAt(1.0, QColor("#2a5298"))
    painter.fillRect(pm.rect(), QBrush(grad))
    painter.setPen(Qt.white)
    painter.setFont(QFont("Segoe UI", 36, QFont.Bold))
    painter.drawText(pm.rect(), Qt.AlignCenter, "ANPR")
    painter.end()
    return QIcon(pm)

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
        self.setStyleSheet("border-radius: 12px;")

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
            # ROI instruction overlay when editing
            if self.editable:
                painter.setRenderHint(QPainter.Antialiasing, True)
                overlay_height = 28
                painter.fillRect(0, self.height() - overlay_height, self.width(), overlay_height, QColor(0, 0, 0, 140))
                painter.setPen(Qt.white)
                painter.drawText(10, self.height() - 8, "ROI edit mode: Click to add points, double-click to finish")
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
        # Table UX improvements
        self.table_data.setAlternatingRowColors(True)
        self.table_data.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_data.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_data.setSortingEnabled(True)
        self.table_data.setWordWrap(False)
        
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
# LoginDialog: Shown before main window to authenticate user
########################################################################
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setModal(True)
        form = QFormLayout()

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Username")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.remember_check = QCheckBox("Remember me")

        # Prefill from settings
        settings = QSettings("KVCET", "ANPR")
        saved_user = settings.value("auth/username", "")
        remember = settings.value("auth/remember", False, type=bool)
        self.username_edit.setText(saved_user if remember else "")
        self.remember_check.setChecked(remember)

        form.addRow("Username", self.username_edit)
        form.addRow("Password", self.password_edit)
        form.addRow("", self.remember_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.try_login)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        title = QLabel("KVCET - ANPR")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18pt; font-weight: 700;")
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._username = None

    def try_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        # Simple credential check; replace with secure auth if needed
        if (username.lower() == "admin" and password == "admin") or (username and password == "admin"):
            self._username = username if username else "admin"
            # Save remember preference
            settings = QSettings("KVCET", "ANPR")
            if self.remember_check.isChecked():
                settings.setValue("auth/username", self._username)
                settings.setValue("auth/remember", True)
            else:
                settings.setValue("auth/username", "")
                settings.setValue("auth/remember", False)
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def get_username(self):
        return self._username or "User"

########################################################################
# MainWindow: Main application window for video detection.
# It features the video feed, detection log, and data loading.
# This version integrates a fixed polygon area check for OCR processing.
########################################################################
class MainWindow(QMainWindow):
    def __init__(self, username: str = "User"):
        super().__init__()
        self.setWindowTitle("ANPR_DR")
        self.setWindowIcon(create_app_icon())
        self.setMinimumSize(1200, 800)
        
        # Define a fixed area (from your reference) if no ROI is set.
        self.fixed_area = [(27, 417), (16, 456), (1015, 451), (992, 417)]
        self.current_user = username
        
        # Top header with app name and Reload button.
        heading_label = QLabel("KVCET - ANPR")
        heading_label.setObjectName("HeaderTitle")
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setStyleSheet("font-size: 22pt; font-weight: 800; letter-spacing: 0.8px;")
        heading_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        apply_drop_shadow(heading_label, radius=12, dy=2, color=QColor(0, 0, 0, 160))
        self.reload_button = QPushButton("Reload")
        self.reload_button.setFixedWidth(100)
        self.reload_button.clicked.connect(self.reload_app)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(8)
        header_layout.addWidget(heading_label, 1, Qt.AlignCenter)
        header_layout.addWidget(self.reload_button, 0, Qt.AlignRight)
        header_widget = QWidget()
        header_widget.setObjectName("Header")
        header_widget.setLayout(header_layout)
        apply_drop_shadow(header_widget)
        
        # Left Panel: Video Feed.
        self.btn_web = QPushButton("Web")
        self.btn_upload = QPushButton("Upload Video")
        self.btn_web.clicked.connect(self.start_webcam)
        self.btn_upload.clicked.connect(self.upload_video)
        video_btn_layout = QHBoxLayout()
        video_btn_layout.addWidget(self.btn_web)
        video_btn_layout.addWidget(self.btn_upload)
        self.video_label = VideoLabel("Video Feed")
        self.video_label.setObjectName("VideoArea")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout = QVBoxLayout()
        left_layout.addLayout(video_btn_layout)
        left_layout.addWidget(self.video_label)
        left_widget = QWidget()
        left_widget.setObjectName("Card")
        left_widget.setLayout(left_layout)
        apply_drop_shadow(left_widget)
        
        # Right Panel: Detection Log and Reset Data button.
        self.table_detections = QTableWidget()
        self.table_detections.setColumnCount(5)
        self.table_detections.setHorizontalHeaderLabels(
            ["Plate ID", "Plate Number", "OCR Confidence", "Plate Confidence", "Timestamp"]
        )
        self.table_detections.setMinimumSize(500, 600)
        # Table UX improvements
        self.table_detections.setAlternatingRowColors(True)
        self.table_detections.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_detections.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_detections.setSortingEnabled(True)
        self.table_detections.setWordWrap(False)
        self.table_detections.setStyleSheet("QTableWidget::item{padding:6px;}")
        self.table_detections.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_detections.verticalHeader().setDefaultSectionSize(28)
        
        self.reset_button = QPushButton("Reset Data")
        self.reset_button.clicked.connect(self.reset_data)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.table_detections)
        right_layout.addWidget(self.reset_button)
        about_label = QLabel("Created by Durgaram R & Keerthana R, ADS Department")
        about_label.setAlignment(Qt.AlignCenter)
        about_label.setStyleSheet("font-style: italic; font-size: 10pt; margin-top: 10px;")
        right_layout.addWidget(about_label)
        right_widget = QWidget()
        right_widget.setObjectName("Card")
        right_widget.setLayout(right_layout)
        apply_drop_shadow(right_widget)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([800, 600])
        self.main_splitter = main_splitter
        
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(8)
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
        view_menu = QMenu("View", self)
        self.act_theme = QAction("Toggle Theme", self)
        self.act_theme.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.act_theme)
        menu_bar.addMenu(view_menu)
        help_menu = QMenu("Help", self)
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(lambda: QMessageBox.information(self, "About", "KVCET ANPR - Desktop Client\nVersion 1.0"))
        menu_bar.addMenu(help_menu)
        self.setMenuBar(menu_bar)
        
        # Toolbar with shortcuts and standard icons
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
        act_webcam = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Start Webcam", self)
        act_webcam.setShortcut("Ctrl+Shift+W")
        act_webcam.setStatusTip("Start webcam input")
        act_webcam.triggered.connect(self.start_webcam)
        
        act_open_video = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Open Video", self)
        act_open_video.setShortcut("Ctrl+O")
        act_open_video.setStatusTip("Open a video file")
        act_open_video.triggered.connect(self.upload_video)
        
        act_settings = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), "Settings", self)
        act_settings.setShortcut("Ctrl+,")
        act_settings.setStatusTip("Open settings dialog")
        act_settings.triggered.connect(self.open_settings_dialog)
        
        act_data = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Data View", self)
        act_data.setShortcut("Ctrl+D")
        act_data.setStatusTip("Open data viewer")
        act_data.triggered.connect(self.open_data_view)
        
        act_reset = QAction(self.style().standardIcon(QStyle.SP_DialogResetButton), "Reset Data", self)
        act_reset.setShortcut("Ctrl+Shift+R")
        act_reset.setStatusTip("Clear today's detections")
        act_reset.triggered.connect(self.reset_data)
        
        act_reload = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "Reload", self)
        act_reload.setShortcut("F5")
        act_reload.setStatusTip("Reload video source")
        act_reload.triggered.connect(self.reload_app)
        
        toolbar.addAction(act_webcam)
        toolbar.addAction(act_open_video)
        toolbar.addSeparator()
        toolbar.addAction(act_settings)
        toolbar.addAction(act_data)
        toolbar.addSeparator()
        toolbar.addAction(act_reset)
        toolbar.addAction(act_reload)
        toolbar.addSeparator()
        toolbar.addAction(self.act_theme)
        
        # Status bar with FPS and detection count
        status = QStatusBar()
        self.setStatusBar(status)
        self.user_label = QLabel(f"User: {self.current_user}")
        self.user_label.setObjectName("BadgeInfo")
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setObjectName("Badge")
        self.count_label = QLabel("Detections: 0")
        self.count_label.setObjectName("BadgeSuccess")
        self.mode_label = QLabel("Mode: Idle")
        self.mode_label.setObjectName("BadgeInfo")
        status.addPermanentWidget(self.user_label)
        status.addPermanentWidget(self.mode_label)
        status.addPermanentWidget(self.fps_label)
        status.addPermanentWidget(self.count_label)
        
        # Pause/Resume button for video control
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        # Add pause button alongside existing video controls
        video_btn_layout.addWidget(self.btn_pause)
        
        # Track current mode and pause state
        self.is_paused = False
        self.current_mode = "Idle"
        
        # Initialize Video Capture, Timer, Models, and Settings.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        
        # Load YOLO model and PaddleOCR.
        self.model = YOLO(r"D:\peer\kvcet_vehicle\model\best.pt")
        try:
            import torch
            if not torch.cuda.is_available():
                QMessageBox.critical(self, "GPU Required", "CUDA GPU not available. Plate detection (YOLO) requires a CUDA GPU. PaddleOCR will run on CPU.")
                self.gpu_ready = False
            else:
                self.model.to('cuda:0')
                self.gpu_ready = True
        except Exception as e:
            QMessageBox.critical(self, "GPU Error", f"Failed to initialize CUDA: {e}")
            self.gpu_ready = False
        
        # Force PaddleOCR to CPU only
        self.ocr_reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, rec_algorithm="SVTR_LCNet", use_gpu=False)
        self.load_app_settings()
        
        self.last_detection_times = {}  # plate_text -> datetime
        self.last_detection_ids = {}      # plate_text -> detection id
        self.detection_interval = 60      # seconds
        self.plate_id_counter = 0
        
        # FPS tracking
        self._last_time = None
        self._fps = 0.0
        
        # Initialize gate controller (ESP32 over serial)
        self.gate_controller = GateController()
        
        self.apply_saved_roi()
        self.load_existing_detection_data()  # Load today's CSV data into the detection log
        
        # Apply persisted theme
        try:
            ui_settings = QSettings("KVCET", "ANPR")
            theme = ui_settings.value("ui/theme", "dark")
            apply_modern_style(theme)
        except Exception:
            pass
        
        # Restore persisted UI state
        self.restore_ui_state()

    def toggle_theme(self):
        try:
            settings = QSettings("KVCET", "ANPR")
            current = settings.value("ui/theme", "dark")
            new_theme = "light" if str(current).lower() == "dark" else "dark"
            settings.setValue("ui/theme", new_theme)
            apply_modern_style(new_theme)
        except Exception as e:
            print("Error toggling theme:", e)

    def restore_ui_state(self):
        try:
            settings = QSettings("KVCET", "ANPR")
            geo = settings.value("geometry")
            if geo is not None:
                self.restoreGeometry(geo)
            state = settings.value("windowState")
            if state is not None:
                self.restoreState(state)
            sizes = settings.value("splitterSizes")
            if sizes is not None:
                try:
                    int_sizes = [int(s) for s in sizes]
                    if len(int_sizes) == 2:
                        self.main_splitter.setSizes(int_sizes)
                except Exception:
                    pass
        except Exception:
            pass

    def save_ui_state(self):
        try:
            settings = QSettings("KVCET", "ANPR")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
            settings.setValue("splitterSizes", self.main_splitter.sizes())
        except Exception:
            pass

    def load_existing_detection_data(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DATA_LOG_DIR, f"detections_{date_str}.csv")
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                for _, r in df.iterrows():
                    row = self.table_detections.rowCount()
                    self.table_detections.insertRow(row)
                    self.table_detections.setItem(row, 0, QTableWidgetItem(str(r["Plate ID"])))
                    self.table_detections.setItem(row, 1, QTableWidgetItem(str(r["Plate Number"])) )
                    # OCR
                    try:
                        ocr_val = float(r["OCR Confidence"]) if pd.notna(r["OCR Confidence"]) else 0.0
                    except Exception:
                        ocr_val = 0.0
                    ocr_bar = QProgressBar()
                    ocr_bar.setRange(0, 100)
                    ocr_bar.setValue(int(round(ocr_val * 100)))
                    ocr_bar.setFormat(f"{ocr_val:.2f}")
                    ocr_bar.setTextVisible(True)
                    ocr_bar.setStyleSheet("QProgressBar::chunk{background-color:#3b82f6;border-radius:6px;} QProgressBar{border-radius:6px;}")
                    self.table_detections.setCellWidget(row, 2, ocr_bar)
                    # Plate
                    try:
                        plate_val = float(r["Plate Confidence"]) if pd.notna(r["Plate Confidence"]) else 0.0
                    except Exception:
                        plate_val = 0.0
                    plate_bar = QProgressBar()
                    plate_bar.setRange(0, 100)
                    plate_bar.setValue(int(round(plate_val * 100)))
                    plate_bar.setFormat(f"{plate_val:.2f}")
                    plate_bar.setTextVisible(True)
                    plate_bar.setStyleSheet("QProgressBar::chunk{background-color:#10b981;border-radius:6px;} QProgressBar{border-radius:6px;}")
                    self.table_detections.setCellWidget(row, 3, plate_bar)
                    self.table_detections.setItem(row, 4, QTableWidgetItem(str(r["Timestamp"])))
                self.count_label.setText(f"Detections: {self.table_detections.rowCount()}")
            except Exception as e:
                print("Error loading detection data:", e)

    def reload_app(self):
        if self.cap is not None:
            self.cap.release()
        if not getattr(self, "gpu_ready", False):
            QMessageBox.critical(self, "GPU Required", "CUDA GPU not available. Cannot start detection.")
            return
        self.cap = cv2.VideoCapture(0)
        self.apply_saved_roi()
        self._last_time = None
        self._fps = 0.0
        self.mode_label.setText("Mode: Webcam")
        self.current_mode = "Webcam"
        self.is_paused = False
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
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
        if not getattr(self, "gpu_ready", False):
            QMessageBox.critical(self, "GPU Required", "CUDA GPU not available. Cannot start detection.")
            return
        self.cap = cv2.VideoCapture(0)
        self.apply_saved_roi()
        self._last_time = None
        self._fps = 0.0
        self.mode_label.setText("Mode: Webcam")
        self.current_mode = "Webcam"
        self.is_paused = False
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.timer.start(30)
        
    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            if self.cap is not None:
                self.cap.release()
            if not getattr(self, "gpu_ready", False):
                QMessageBox.critical(self, "GPU Required", "CUDA GPU not available. Cannot start detection.")
                return
            self.cap = cv2.VideoCapture(file_name)
            self.apply_saved_roi()
            self._last_time = None
            self._fps = 0.0
            self.mode_label.setText("Mode: File")
            self.current_mode = "File"
            self.is_paused = False
            self.btn_pause.setEnabled(True)
            self.btn_pause.setText("Pause")
            self.timer.start(30)
            
    def toggle_pause(self):
        if self.cap is None or not self.cap.isOpened():
            return
        if not self.is_paused:
            # Pause
            try:
                self.timer.stop()
            except Exception:
                pass
            self.is_paused = True
            self.btn_pause.setText("Resume")
            self.mode_label.setText("Mode: Paused")
            self._last_time = None
        else:
            # Resume
            try:
                self.timer.start(30)
            except Exception:
                pass
            self.is_paused = False
            self.btn_pause.setText("Pause")
            self.mode_label.setText(f"Mode: {self.current_mode}")
            self._last_time = None
            
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
            # FPS calculation (EMA)
            now = time.time()
            if self._last_time is not None:
                inst_fps = 1.0 / max(1e-6, (now - self._last_time))
                self._fps = 0.9 * self._fps + 0.1 * inst_fps if self._fps > 0 else inst_fps
                self.fps_label.setText(f"FPS: {self._fps:.1f}")
            self._last_time = now
            
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
                best_text = ""
                best_conf = 0.0
                # Simply select the result with the highest confidence, no regex or length filtering.
                for result in results[0]:
                    if len(result) >= 2:
                        text = result[1][0]
                        confidence = round(float(result[1][1]), 2)
                        if confidence > best_conf:
                            best_conf = confidence
                            best_text = text.replace(" ","").strip()
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
            results = self.model(roi, device=0)
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
                            # Open gate for target plate (always check, debounced in controller)
                            try:
                                if plate_text.upper() == TARGET_PLATE.upper():
                                    self.gate_controller.open_gate()
                            except Exception:
                                pass
                            cv2.putText(frame, f"ID: {detection_id}", (x_det1, y_det1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # If no ROI is defined, use the fixed polygon area.
            cv2.polylines(frame, [np.array(self.fixed_area, np.int32)], True, (255, 0, 0), 2)
            results = self.model(frame, device=0)
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
                            # Open gate for target plate (always check, debounced in controller)
                            try:
                                if plate_text.upper() == TARGET_PLATE.upper():
                                    self.gate_controller.open_gate()
                            except Exception:
                                pass
                            cv2.putText(frame, f"ID: {detection_id}", (x_det1, y_det1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def append_detection_info(self, plate_id, plate_text, ocr_conf, plate_conf):
        row = self.table_detections.rowCount()
        self.table_detections.insertRow(row)
        self.table_detections.setItem(row, 0, QTableWidgetItem(str(plate_id)))
        self.table_detections.setItem(row, 1, QTableWidgetItem(plate_text))
        # OCR confidence (blue)
        ocr_bar = QProgressBar()
        ocr_bar.setRange(0, 100)
        ocr_bar.setValue(int(round(ocr_conf * 100)))
        ocr_bar.setFormat(f"{ocr_conf:.2f}")
        ocr_bar.setTextVisible(True)
        ocr_bar.setStyleSheet("QProgressBar::chunk{background-color:#3b82f6;border-radius:6px;} QProgressBar{border-radius:6px;}")
        self.table_detections.setCellWidget(row, 2, ocr_bar)
        # Plate confidence (green)
        plate_bar = QProgressBar()
        plate_bar.setRange(0, 100)
        plate_bar.setValue(int(round(plate_conf * 100)))
        plate_bar.setFormat(f"{plate_conf:.2f}")
        plate_bar.setTextVisible(True)
        plate_bar.setStyleSheet("QProgressBar::chunk{background-color:#10b981;border-radius:6px;} QProgressBar{border-radius:6px;}")
        self.table_detections.setCellWidget(row, 3, plate_bar)
        self.table_detections.setItem(row, 4, QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.count_label.setText(f"Detections: {self.table_detections.rowCount()}")

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
        self.count_label.setText("Detections: 0")
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
        self.save_ui_state()
        try:
            self.gate_controller.close()
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()
        event.accept()

########################################################################
# Main Application Entry Point
########################################################################
if __name__ == '__main__':
    # Enable High DPI and set professional light theme
    try:
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    try:
        app.setStyle(QStyleFactory.create("Fusion"))
        app.setFont(QFont("Segoe UI", 10))
        # Use default light palette (no dark overrides)
    except Exception:
        pass

    # Apply saved theme at startup
    try:
        _settings = QSettings("KVCET", "ANPR")
        _theme = _settings.value("ui/theme", "dark")
        apply_modern_style(_theme)
    except Exception:
        pass

    # Show login before main window
    login = LoginDialog()
    if login.exec():
        user = login.get_username()
        window = MainWindow(username=user)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
