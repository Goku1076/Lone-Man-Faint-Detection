import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


# Parameters
DATASET_DIR = "dataset/test"  # or "dataset/test"
OUTPUT_FILE = "data_test.npy"  # or "data_test.npy"
SEQ_LENGTH = 30


# Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def extract_pose_keypoints(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return keypoints
    return None


def generate_sequences(frames, label):
    sequences = []
    for i in range(len(frames) - SEQ_LENGTH + 1):
        clip = frames[i:i + SEQ_LENGTH]
        sequences.append((np.array(clip), label))
    return sequences


def main():
    all_data = []
    label_map = {"FAINT": 1, "NORMAL": 0}


    for cls in ["FAINT", "NORMAL"]:
        folder = os.path.join(DATASET_DIR, cls)
        label = label_map[cls]


        for file in tqdm(os.listdir(folder), desc=f"📼 Processing {cls}"):
            video_path = os.path.join(folder, file)
            cap = cv2.VideoCapture(video_path)


            if not cap.isOpened():
                print(f"❌ Could not open {video_path}")
                continue


            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                keypoints = extract_pose_keypoints(frame)
                if keypoints:
                    frames.append(keypoints)
            cap.release()


            if len(frames) < SEQ_LENGTH:
                print(f"⚠️ Skipping {file}: only {len(frames)} pose frames found.")
                continue


            sequences = generate_sequences(frames, label)
            for seq, lbl in sequences:
                if seq.shape == (SEQ_LENGTH, 132):
                    all_data.append((seq, lbl))
                else:
                    print(f"⚠️ Skipping sequence with shape {seq.shape} from {file}")


    # ✅ Save after filtering
    print(f"✅ Total sequences collected: {len(all_data)}")
    np.save(OUTPUT_FILE, np.array(all_data, dtype=object))  # Use dtype=object for tuples


if __name__ == "__main__":
    main()
