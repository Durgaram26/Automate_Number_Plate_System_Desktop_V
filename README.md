# ANPR Desktop App

An offline Automatic Number Plate Recognition (ANPR) desktop application with region-of-interest (ROI) support. This repository contains scripts for running ANPR on video files or camera input, a trained model, recorded detection logs, and example videos.

- **Repository root files**: `app_old.py`, `gate.py`, `real7.py`, `real*.py` test variants
- **Model**: `model/best.pt` (trained model weights)
- **Data logs**: `data_log/` (CSV detection exports and ROI/settings JSON)
- **Sample videos**: `sample_video/` (example input MP4s)

## Features

- Offline ANPR (no cloud required)
- ROI editing and configuration for focused detection areas
- Save detection events to CSV logs in `data_log/`
- Simple gate control example (`gate.py` / `gate1.ino`) for integration with hardware
- Training and experiment scripts in `TESTING/`

## Requirements

- Python 3.8+ (3.10 recommended)
- Common packages: `torch`, `opencv-python`, `numpy`, `pandas` (see usage below for install)

## Quick desktop setup

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install common dependencies:

```bash
pip install torch torchvision opencv-python numpy pandas
```

## Running the ANPR desktop app

Pick an entry-point script suited for desktop use:

- Run a testing script (example):

```bash
python TESTING/app.py
```

- Run the offline detector (example):

```bash
python real7.py
# or
python app_old.py
```

Notes:
- The detector expects a model file at `model/best.pt`.
- Detection CSV files are stored under `data_log/` (e.g., `detections_YYYY-MM-DD.csv`).
- Edit `data_log/roi_settings.json` to change the ROI used by the detector.

## ROI configuration

ROI settings are stored in `data_log/roi_settings.json`. Edit the polygon or rectangle coordinates in that file to restrict detection to a specific area of the frame.

## Testing with sample videos

Use the sample videos in `sample_video/` to test the detector without a camera:

```bash
python real7.py --source sample_video/a.mp4
```

## Development notes

- Experimental and training scripts are in `TESTING/` (e.g., `onnx_trainer.py`).
- Model weights are included in `model/best.pt` — replace with your own trained weights if desired.

## Contributing

Contributions are welcome. Please open issues or pull requests with a clear description of changes.

## License

This project does not currently include a license file. Add a `LICENSE` in the repository if you want to apply an open-source license (e.g., MIT). 