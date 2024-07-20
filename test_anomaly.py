import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque
import shutil

def frame_analysis(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(prev_gray, curr_gray)
    mean_diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))

    return ssim_score, mean_diff

def is_anomaly(ssim_score, mean_diff, ssim_history, mean_diff_history):
    if len(ssim_history) < 5:
        return False

    ssim_avg = np.mean(ssim_history)
    mean_diff_avg = np.mean(mean_diff_history)

    ssim_threshold = 0.8
    mean_diff_threshold = 7.0

    if (ssim_score < ssim_avg * 0.75 and mean_diff > mean_diff_avg * 1.25) or \
       (ssim_score < ssim_threshold and mean_diff > mean_diff_threshold):
        return True

    return False

def process_frames(input_folder, output_folder):
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')])

    normal_folder = os.path.join(output_folder, 'normal')
    anomaly_folder = os.path.join(output_folder, 'anomaly')

    os.makedirs(normal_folder, exist_ok=True)
    os.makedirs(anomaly_folder, exist_ok=True)

    prev_frame = None
    ssim_history = deque(maxlen=5)
    mean_diff_history = deque(maxlen=5)

    for idx, frame_name in enumerate(frames):
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)

        if prev_frame is not None:
            ssim_score, mean_diff = frame_analysis(prev_frame, frame)
            ssim_history.append(ssim_score)
            mean_diff_history.append(mean_diff)

            if is_anomaly(ssim_score, mean_diff, ssim_history, mean_diff_history):
                print(f"Anomaly detected in frame {frame_name}")
                shutil.copy(frame_path, os.path.join(anomaly_folder, frame_name))
            else:
                shutil.copy(frame_path, os.path.join(normal_folder, frame_name))
        else:
            shutil.copy(frame_path, os.path.join(normal_folder, frame_name))

        prev_frame = frame

if __name__ == "__main__":
    input_folder = "anomaly_test"
    output_folder = "test_anomaly_detect"

    process_frames(input_folder, output_folder)
    print("Processing completed.")
