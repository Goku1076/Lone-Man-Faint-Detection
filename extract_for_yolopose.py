import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

DATASET_DIR = "2july_dataset"
OUTPUT_FILE = "data_train.npy"
SEQ_LENGTH = 10
INPUT_SIZE = 132
RESIZE_WIDTH, RESIZE_HEIGHT = 640, 480
CONF_THRES = 0.3

model = YOLO("yolov8x-pose.pt")

def get_frame_keypoints(frame):
    results = model.predict(frame, conf=CONF_THRES, verbose=False)[0]
   
    if results.keypoints is None or results.keypoints.data.shape[0] == 0:
        return None

    try:
        raw_kp = results.keypoints.data[0].cpu().numpy()  
    except Exception:
        return None

    formatted_kp = []
    for pt in raw_kp:
        if len(pt) == 3:
            x, y, conf = pt
            if conf > CONF_THRES:
                formatted_kp.extend([x / frame.shape[1], y / frame.shape[0], 0.0, conf])
            else:
                formatted_kp.extend([0.0, 0.0, 0.0, 0.0])
        else:
            formatted_kp.extend([0.0, 0.0, 0.0, 0.0])

    formatted_kp = (formatted_kp + [0.0] * INPUT_SIZE)[:INPUT_SIZE]
    return formatted_kp

def slice_into_sequences(frames, label):
    sequences = []
    for i in range(len(frames) - SEQ_LENGTH + 1):
        chunk = frames[i : i + SEQ_LENGTH]
        sequences.append((np.array(chunk), label))
    return sequences

def main():
    dataset_records = []
    class_map = {"FAINT": 1, "NORMAL": 0}

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Directory '{DATASET_DIR}' was not found.")
        return

    for class_name in ["FAINT", "NORMAL"]:
        class_folder = os.path.join(DATASET_DIR, class_name)
        target_label = class_map[class_name]
        
        if not os.path.exists(class_folder):
            continue

        for video_file in tqdm(os.listdir(class_folder), desc=f"Videos ({class_name})"):
            path = os.path.join(class_folder, video_file)
            cap = cv2.VideoCapture(path)
            
            if not cap.isOpened():
                continue

            extracted_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                kps = get_frame_keypoints(resized)
                if kps:
                    extracted_frames.append(kps)
            
            cap.release()

            if len(extracted_frames) < SEQ_LENGTH:
                continue

            sequences = slice_into_sequences(extracted_frames, target_label)
            for seq, lbl in sequences:
                if seq.shape == (SEQ_LENGTH, INPUT_SIZE):
                    dataset_records.append((seq, lbl))

    faint_total = sum(1 for _, l in dataset_records if l == 1)
    normal_total = sum(1 for _, l in dataset_records if l == 0)
    print(f"Stats -> FAINT: {faint_total} | NORMAL: {normal_total}")

    np.save(OUTPUT_FILE, np.array(dataset_records, dtype=object))
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
