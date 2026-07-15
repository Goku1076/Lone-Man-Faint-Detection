# Lone-Man-Faint-Detection

This project is a deep learning computer vision pipeline designed to monitor isolated rooms and detect sudden faint or collapse events and alert safety and medical personnel in emergency situations.

## Logic
Instead of utilizing standard MediaPipe skeleton tracking which is prone to frame drops or inaccurate joint localization in sideways/falling positions, this codebase relies on:
1. YOLOv8-Pose Estimation - to track human existence using bounding boxes and body keypoints during sudden movements.
2. LSTM Sequence Classifier - to analyze temporal movements (with a 10-frame buffer) and distinguish between everyday movements (such as sitting down, bending over) and actual collapse patterns.
3. Continuous Fall Alerting - prints alert notifications if a faint state is held for 10 consecutive seconds, saving timestamps and logs in using a CSV file.
## Repository Structure
* requirements.txt - Python module dependency requirements.
* extract_for_yolopose.py - Preprocesses raw video training datasets and converts them into normalized spatial coordinate keypoint arrays.
* train.py - Standard PyTorch training script for building and optimizing the LSTM classification model.
* fall_detector.py - The live inference script which renders skeletons, predictions, logs frame actions, and outputs console warnings.

## Setup and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
