import os
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from media_pipe import FaceMeshDetector, FaceMeshAlign

def process_video_with_align(video_path, save_dir):
    face_detector = FaceMeshDetector()
    face_aligner = FaceMeshAlign()

    frames = imageio.get_reader(video_path)
    face_results = []
    motions = []

    # Process first frame to get reference
    first_frame = next(iter(frames))
    first_frame_rgb = np.array(first_frame)
    motion, ref_result = face_detector(first_frame_rgb)
    if ref_result is None:
        print("No face detected in the first frame. Exiting.")
        return

    face_results.append(ref_result)
    motions.append(motion)

    # Process remaining frames
    for frame in tqdm(frames):
        frame_rgb = np.array(frame)
        motion, face_result = face_detector(frame_rgb)
        if face_result is None:
            continue
        face_results.append(face_result)
        motions.append(motion)

    # Perform alignment
    aligned_motions = face_aligner(ref_result, face_results)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Save original results
    npy_path = os.path.join(save_dir, f"{base_name}_mppose.npy")
    np.save(npy_path, face_results)

    gif_path = os.path.join(save_dir, f"{base_name}_mppose.gif")
    imageio.mimsave(gif_path, motions, 'GIF', duration=0.2, loop=0)

    # Save aligned results
    aligned_npy_path = os.path.join(save_dir, f"{base_name}_mppose_aligned.npy")
    np.save(aligned_npy_path, aligned_motions)

    aligned_gif_path = os.path.join(save_dir, f"{base_name}_mppose_aligned.gif")
    imageio.mimsave(aligned_gif_path, aligned_motions, 'GIF', duration=0.2, loop=0)

    return npy_path, aligned_npy_path, len(face_results)

# Пример использования
video_path = "./test/test.mp4"
save_dir = "./test"

original_npy, aligned_npy, frame_count = process_video_with_align(video_path, save_dir)
print(f"Processed {frame_count} frames.")
print(f"Original NPY saved at: {original_npy}")
print(f"Aligned NPY saved at: {aligned_npy}")
print(f"GIFs saved in: {save_dir}")
