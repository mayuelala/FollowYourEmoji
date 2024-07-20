import os
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from media_pipe import FaceMeshDetector, FaceMeshAlign
from datetime import datetime

def crop_image_to_face(image, face_result, target_size):
    height, width = image.shape[:2]
    target_height, target_width = target_size

    # Получаем координаты лица
    landmarks = face_result['lmks']
    min_x, min_y = np.min(landmarks, axis=0)[:2]
    max_x, max_y = np.max(landmarks, axis=0)[:2]

    # Вычисляем центр лица
    center_x = (min_x + max_x) / 2 * width
    center_y = (min_y + max_y) / 2 * height

    # Вычисляем координаты для обрезки
    left = int(max(center_x - target_width / 2, 0))
    top = int(max(center_y - target_height / 2, 0))
    right = int(min(left + target_width, width))
    bottom = int(min(top + target_height, height))

    # Обрезаем изображение
    cropped_image = image[top:bottom, left:right]

    # Если обрезанное изображение меньше целевого размера, добавляем отступы
    if cropped_image.shape[0] < target_height or cropped_image.shape[1] < target_width:
        pad_top = max(0, (target_height - cropped_image.shape[0]) // 2)
        pad_bottom = max(0, target_height - cropped_image.shape[0] - pad_top)
        pad_left = max(0, (target_width - cropped_image.shape[1]) // 2)
        pad_right = max(0, target_width - cropped_image.shape[1] - pad_left)
        cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return cv2.resize(cropped_image, (target_width, target_height))

def process_image(image_path, face_detector, target_size):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    motion, face_result = face_detector(image_rgb)

    if face_result is not None:
        cropped_image = crop_image_to_face(image_rgb, face_result, target_size)
        cropped_motion, cropped_face_result = face_detector(cropped_image)
    else:
        cropped_image = None
        cropped_motion = None
        cropped_face_result = None

    return motion, face_result, cropped_image, cropped_motion, cropped_face_result

def process_video_and_image(video_path, image_path, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"test_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    face_detector = FaceMeshDetector()
    face_aligner = FaceMeshAlign()

    # Получаем размер кадра видео
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    # Process image
    _, _, cropped_image, cropped_motion, cropped_face_result = process_image(image_path, face_detector, (frame_height, frame_width))
    if cropped_face_result is None:
        print("No face detected in the image. Exiting.")
        return

    # Save processed and cropped image
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_image_path = os.path.join(save_dir, f"{image_base_name}_processed.png")
    cv2.imwrite(processed_image_path, cv2.cvtColor(cropped_motion, cv2.COLOR_RGB2BGR))

    cropped_image_path = os.path.join(save_dir, f"{image_base_name}_cropped.png")
    cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    # Process video
    frames = imageio.get_reader(video_path)
    face_results = []
    motions = []

    for frame in tqdm(frames, desc="Processing video"):
        frame_rgb = np.array(frame)
        motion, face_result = face_detector(frame_rgb)
        if face_result is None:
            continue
        face_results.append(face_result)
        motions.append(motion)

    # Perform alignment
    aligned_motions = face_aligner(cropped_face_result, face_results)

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

    print(f"Processed image saved at: {processed_image_path}")
    print(f"Cropped image saved at: {cropped_image_path}")
    print(f"Original NPY saved at: {npy_path}")
    print(f"Aligned NPY saved at: {aligned_npy_path}")
    print(f"GIFs saved in: {save_dir}")

# Пример использования
video_path = "./test/test.mp4"
image_path = "./test/123.png"
output_dir = "./test"

process_video_and_image(video_path, image_path, output_dir)