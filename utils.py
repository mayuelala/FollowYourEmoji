import os
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from media_pipe.mp_utils import LMKExtractor
from media_pipe.draw_util import FaceMeshVisualizer
from media_pipe import FaceMeshAlign

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return int(fps)

def process_video(video_path, save_dir, save_gif=True):
    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer(forehead_edge=False)
    face_aligner = FaceMeshAlign()

    frames = imageio.get_reader(video_path)
    face_results = []
    motions = []

    # Process first frame to get reference
    first_frame = next(iter(frames))
    first_frame_bgr = cv2.cvtColor(np.array(first_frame), cv2.COLOR_RGB2BGR)
    ref_result = lmk_extractor(first_frame_bgr)
    if ref_result is None:
        print("No face detected in the first frame. Exiting.")
        return None, 0

    ref_result['width'] = first_frame_bgr.shape[1]
    ref_result['height'] = first_frame_bgr.shape[0]
    face_results.append(ref_result)

    # Process remaining frames
    for frame in tqdm(frames):
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        face_result = lmk_extractor(frame_bgr)
        if face_result is None:
            continue
        face_result['width'] = frame_bgr.shape[1]
        face_result['height'] = frame_bgr.shape[0]
        face_results.append(face_result)
        lmks = face_result['lmks'].astype(np.float32)
        motion = vis.draw_landmarks((frame_bgr.shape[1], frame_bgr.shape[0]), lmks, normed=True)
        motions.append(motion)

    # Perform alignment
    aligned_motions = face_aligner(ref_result, face_results)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    npy_path = os.path.join(save_dir, f"{base_name}_mppose.npy")
    np.save(npy_path, face_results)

    if save_gif:
        # Save regular GIF
        gif_path = os.path.join(save_dir, f"{base_name}_mppose.gif")
        imageio.mimsave(gif_path, motions, 'GIF', duration=0.2, loop=0)

        # Save aligned GIF
        aligned_gif_path = os.path.join(save_dir, f"{base_name}_mppose_aligned.gif")
        imageio.mimsave(aligned_gif_path, aligned_motions, 'GIF', duration=0.2, loop=0)

    return npy_path, len(face_results)

def get_npy_files(root_dir):
    npy_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def get_frame_count(npy_path):
    data = np.load(npy_path, allow_pickle=True)
    return len(data) - 1

def show_gif(npy_path):
    aligned_gif_path = npy_path.replace('.npy', '_aligned.gif')
    if os.path.exists(aligned_gif_path):
        return aligned_gif_path, "Aligned GIF found and displayed"
    return None, "No aligned GIF found for this NPY file"

def process_image(image_path, npy_path, save_dir, expand_x=1.0, expand_y=1.0, offset_x=0.0, offset_y=0.0):
    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer(forehead_edge=False)

    # Load data from npy file
    face_results = np.load(npy_path, allow_pickle=True)
    if len(face_results) == 0:
        print("No face data in the NPY file. Exiting.")
        return None

    # Get dimensions from first frame in npy
    target_width = face_results[0]['width']
    target_height = face_results[0]['height']

    # Process image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_result = lmk_extractor(image)

    if face_result is None:
        print("No face detected in the image. Exiting.")
        return None

    # Crop image
    landmarks = face_result['lmks']
    min_x, min_y = np.min(landmarks, axis=0)[:2]
    max_x, max_y = np.max(landmarks, axis=0)[:2]

    center_x = (min_x + max_x) / 2 * image.shape[1]
    center_y = (min_y + max_y) / 2 * image.shape[0]

    # Apply expansion and offset
    crop_width = target_width * expand_x
    crop_height = target_height * expand_y
    offset_x_pixels = offset_x * target_width
    offset_y_pixels = offset_y * target_height

    left = int(max(center_x - crop_width / 2 + offset_x_pixels, 0))
    top = int(max(center_y - crop_height / 2 + offset_y_pixels, 0))
    right = int(min(left + crop_width, image.shape[1]))
    bottom = int(min(top + crop_height, image.shape[0]))

    cropped_image = image_rgb[top:bottom, left:right]

    # If cropped image is smaller than target size, add padding
    if cropped_image.shape[0] < target_height or cropped_image.shape[1] < target_width:
        pad_top = max(0, (target_height - cropped_image.shape[0]) // 2)
        pad_bottom = max(0, target_height - cropped_image.shape[0] - pad_top)
        pad_left = max(0, (target_width - cropped_image.shape[1]) // 2)
        pad_right = max(0, target_width - cropped_image.shape[1] - pad_left)
        cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    cropped_image = cv2.resize(cropped_image, (target_width, target_height))

    # Save cropped image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cropped_image_path = os.path.join(save_dir, f"{base_name}_cropped.png")
    cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    # Process cropped image
    cropped_face_result = lmk_extractor(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    if cropped_face_result is None:
        print("No face detected in the cropped image. Exiting.")
        return None

    cropped_face_result['width'] = target_width
    cropped_face_result['height'] = target_height

    # Visualize facial landmarks
    lmks = cropped_face_result['lmks'].astype(np.float32)
    motion = vis.draw_landmarks((target_width, target_height), lmks, normed=True)

    # Save visualization
    motion_path = os.path.join(save_dir, f"{base_name}_motion.png")
    cv2.imwrite(motion_path, cv2.cvtColor(motion, cv2.COLOR_RGB2BGR))

    return cropped_image_path, motion_path
