import os
import cv2
import numpy as np
import torch
import csv
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from train import LSTMClassifier, INPUT_SIZE, device
import screeninfo

SEQ_LENGTH = 10
FAINT_CLASS = 1
CONF_THRES = 0.3
SMOOTH_WINDOW = 3
ALERT_TIME_LIMIT = 10 

lstm_model = LSTMClassifier().to(device)
lstm_model.load_state_dict(torch.load("mp_lstm_faint_model.pth", map_location=device))
lstm_model.eval()

yolo_model = YOLO("yolov8x-pose.pt")

SKELETON_BONES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (1, 2), (2, 3), (3, 4)
]

cap = cv2.VideoCapture("final_test_dataset/test_2 .mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
required_consec_frames = int(ALERT_TIME_LIMIT * fps // 5) 

monitor = screeninfo.get_monitors()[0]
SCR_W, SCR_H = monitor.width, monitor.height

seq_buffer = deque(maxlen=SEQ_LENGTH)
frame_count = 0
faint_streak = 0
is_alerting = False

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"faint_log_yolopose_{session_id}.csv"

cv2.namedWindow("Real-Time Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Detection", SCR_W, SCR_H)

def extract_and_draw_skeleton(frame):
    pred_results = yolo_model.predict(frame, conf=CONF_THRES, verbose=False)[0]
    flat_keypoints = [0.0] * INPUT_SIZE
    drawn_coords = []

    if pred_results.keypoints is not None and pred_results.keypoints.data.shape[0] > 0:
        kp_data = pred_results.keypoints.data[0].cpu().numpy()
        temp_list = []
        
        for idx, point in enumerate(kp_data):
            if len(point) == 3:
                x, y, conf = point
                if conf > CONF_THRES:
                    temp_list.extend([x / frame.shape[1], y / frame.shape[0], 0.0, conf])
                    drawn_coords.append((int(x), int(y)))
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                else:
                    temp_list.extend([0.0, 0.0, 0.0, 0.0])
                    drawn_coords.append(None)
            else:
                temp_list.extend([0.0, 0.0, 0.0, 0.0])
                drawn_coords.append(None)

        for bone_start, bone_end in SKELETON_BONES:
            if bone_start < len(drawn_coords) and bone_end < len(drawn_coords):
                pt1, pt2 = drawn_coords[bone_start], drawn_coords[bone_end]
                if pt1 and pt2:
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        flat_keypoints = (temp_list + [0.0] * INPUT_SIZE)[:INPUT_SIZE]
        num_visible = sum(1 for pt in drawn_coords if pt is not None)
        return flat_keypoints, num_visible, drawn_coords

    return flat_keypoints, 0, []

with open(log_file_path, mode='w', newline='') as csvfile:
    log_writer = csv.writer(csvfile)
    log_writer.writerow(['frame', 'timestamp', 'visible_keypoints', 'prediction'])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        timestamp_seconds = frame_count / fps
        posture_label = "NO_POSE"
        
        kps, visible_points, _ = extract_and_draw_skeleton(frame)
        seq_buffer.append(kps)

        if len(seq_buffer) == SEQ_LENGTH:
            smoothed_sequence = []
            for i in range(len(seq_buffer)):
                start_index = max(0, i - SMOOTH_WINDOW + 1)
                averaged_vals = np.mean(list(seq_buffer)[start_index : i + 1], axis=0)
                smoothed_sequence.append(averaged_vals)

            input_tensor = torch.tensor([smoothed_sequence], dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = lstm_model(input_tensor)
                predicted_idx = torch.argmax(outputs, dim=1).item()
                posture_label = "FAINT" if predicted_idx == FAINT_CLASS else "NORMAL"

                if predicted_idx == FAINT_CLASS:
                    faint_streak += 1
                    if faint_streak >= required_consec_frames and not is_alerting:
                        print("🚨 SYSTEM ALERT: Worker fainting detected!")
                        is_alerting = True
                else:
                    faint_streak = 0
                    is_alerting = False

        log_writer.writerow([frame_count, f"{timestamp_seconds:.2f}", visible_points, posture_label])

        output_frame = cv2.resize(frame, (SCR_W, SCR_H))
        
        if posture_label == "NO_POSE":
            hud_color = (128, 128, 128)
        elif posture_label == "FAINT":
            hud_color = (0, 0, 255)
        else:
            hud_color = (0, 255, 0)

        cv2.putText(output_frame, posture_label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, hud_color, 4)
        cv2.imshow("Real-Time Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Video processing finished. Logs written to {log_file_path}")
