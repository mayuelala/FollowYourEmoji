import gradio as gr
import os
import numpy as np
import yaml
import cv2
import zipfile
from utils import process_video, get_npy_files, get_frame_count, process_image
from infer_script import run_inference

import time
import datetime
import shutil

import imageio
from media_pipe.draw_util import FaceMeshVisualizer

PROCESSED_VIDEO_DIR = './processed_videos'
TEMP_DIR = './temp'
INFER_CONFIG_PATH = './configs/infer.yaml'
MODEL_PATH = './ckpt_models/ckpts'
OUTPUT_PATH = './output'

def load_config():
    with open(INFER_CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

def save_config(config):
    with open(INFER_CONFIG_PATH, 'w') as file:
        yaml.dump(config, file)

config = load_config()

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return int(fps)

def update_npy_choices():
    npy_files = get_npy_files(PROCESSED_VIDEO_DIR)
    return gr.update(choices=["None"] + npy_files)

def create_gif_from_npy(npy_path, gif_path):
    face_results = np.load(npy_path, allow_pickle=True)
    vis = FaceMeshVisualizer(forehead_edge=False)

    frames = []
    for face_result in face_results:
        width = face_result['width']
        height = face_result['height']
        lmks = face_result['lmks'].astype(np.float32)
        frame = vis.draw_landmarks((width, height), lmks, normed=True)
        frames.append(frame)

    imageio.mimsave(gif_path, frames, 'GIF', duration=0.2, loop=0)
    return gif_path

def show_gif_for_npy(npy_file, video_path):
    if npy_file and npy_file != "None":
        npy_path = npy_file
    elif video_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        npy_path = os.path.join(PROCESSED_VIDEO_DIR, video_name, f"{video_name}_mppose.npy")
        if not os.path.exists(npy_path):
            npy_path = os.path.join(TEMP_DIR, video_name, f"{video_name}_mppose.npy")
    else:
        return None, "No NPY file or video selected"

    if not os.path.exists(npy_path):
        return None, "NPY file not found"

    try:
        gif_path = os.path.join(TEMP_DIR, f"{os.path.splitext(os.path.basename(npy_path))[0]}_preview.gif")
        create_gif_from_npy(npy_path, gif_path)
        return gif_path, "GIF created and displayed"
    except Exception as e:
        return None, f"Failed to create GIF: {str(e)}"

def process_input_video(video, save_to_processed):
    if video is None:
        return "No video uploaded", None, gr.update(), gr.update()

    video_name = os.path.splitext(os.path.basename(video))[0]

    if save_to_processed:
        save_dir = os.path.join(PROCESSED_VIDEO_DIR, video_name)
    else:
        save_dir = os.path.join(TEMP_DIR, video_name)

    os.makedirs(save_dir, exist_ok=True)

    npy_path, frame_count = process_video(video, save_dir)
    frame_count = frame_count - 1
    fps = get_video_fps(video)

    return (f"Video processed. NPY file saved at {npy_path}. Original FPS: {fps}",
            npy_path,
            gr.update(maximum=frame_count, value=frame_count),
            gr.update(value=fps))


def update_frame_count(npy_file):
    if npy_file is None or npy_file == "None":
        return gr.update()
    frame_count = get_frame_count(npy_file)
    return gr.update(maximum=frame_count, value=frame_count)

def update_gif_on_video_change(video):
    if video:
        gif_path, status = show_gif_for_npy(None, video)
        return gif_path, status
    return None, "No video selected"

def toggle_fps_slider(use_custom):
    return gr.update(interactive=use_custom)

def crop_face(image_path, should_crop_face, npy_file, video_path, expand_x, expand_y, offset_x, offset_y):
    if not should_crop_face:
        return image_path, "Face cropping not requested"

    if npy_file and npy_file != "None":
        npy_path = npy_file
    elif video_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        npy_path = os.path.join(PROCESSED_VIDEO_DIR, video_name, f"{video_name}_mppose.npy")
        if not os.path.exists(npy_path):
            npy_path = os.path.join(TEMP_DIR, video_name, f"{video_name}_mppose.npy")
    else:
        return image_path, "No NPY file or video selected for face cropping"

    if not os.path.exists(npy_path):
        return image_path, "NPY file not found for face cropping"

    save_dir = os.path.dirname(npy_path)
    cropped_image_path, motion_path = process_image(image_path, npy_path, save_dir, expand_x, expand_y, offset_x, offset_y)

    if cropped_image_path:
        return cropped_image_path, "Face cropped successfully"
    else:
        return image_path, "Face cropping failed"

def preview_crop(image_path, npy_file, video_path, expand_x, expand_y, offset_x, offset_y):
    if not image_path:
        return None, "No image uploaded"

    if npy_file and npy_file != "None":
        npy_path = npy_file
    elif video_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        npy_path = os.path.join(PROCESSED_VIDEO_DIR, video_name, f"{video_name}_mppose.npy")
        if not os.path.exists(npy_path):
            npy_path = os.path.join(TEMP_DIR, video_name, f"{video_name}_mppose.npy")
    else:
        return None, "No NPY file or video selected for face cropping"

    if not os.path.exists(npy_path):
        return None, "NPY file not found for face cropping"

    save_dir = TEMP_DIR
    cropped_image_path, _ = process_image(image_path, npy_path, save_dir, expand_x, expand_y, offset_x, offset_y)

    if cropped_image_path:
        return cropped_image_path, "Crop preview generated"
    else:
        return None, "Failed to generate crop preview"

def generate_video(input_img, should_crop_face, expand_x, expand_y, offset_x, offset_y, input_video_type, input_video, input_npy_select, input_npy, input_video_frames,
                   settings_steps, settings_cfg_scale, settings_seed, resolution_w, resolution_h,
                   model_step, custom_output_path, use_custom_fps, output_fps, save_frames, remove_anomaly_frames, anomaly_detection_mode):
    config['resolution_w'] = resolution_w
    config['resolution_h'] = resolution_h
    config['video_length'] = input_video_frames
    save_config(config)

    if input_video_type == "video":
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        lmk_path = os.path.join(PROCESSED_VIDEO_DIR if input_video_save.value else TEMP_DIR, video_name, f"{video_name}_mppose.npy")
        if not use_custom_fps:
            output_fps = 7
    else:
        if input_npy_select != "None":
            lmk_path = input_npy_select
        else:
            lmk_path = input_npy
        video_name = os.path.splitext(os.path.basename(lmk_path))[0]
        if not use_custom_fps:
            output_fps = 7  # default FPS


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{video_name}_{timestamp}"

    if custom_output_path:
        output_path = os.path.join(custom_output_path, output_folder)
    else:
        output_path = os.path.join(OUTPUT_PATH, output_folder)

    os.makedirs(output_path, exist_ok=True)

    if should_crop_face:
        cropped_image_path, crop_status = crop_face(input_img, should_crop_face, input_npy_select if input_video_type == "npy" else None, input_video if input_video_type == "video" else None, expand_x, expand_y, offset_x, offset_y)
        print(crop_status)

        if cropped_image_path and os.path.exists(cropped_image_path):
            cropped_face_in_result = os.path.join(output_path, "cropped_face.png")
            shutil.copy(cropped_image_path, cropped_face_in_result)
            print(f"Cropped face saved in result folder: {cropped_face_in_result}")

        input_img = cropped_image_path

    status, oo_video_path, all_video_path = run_inference(
        config_path=INFER_CONFIG_PATH,
        model_path=MODEL_PATH,
        input_path=input_img,
        lmk_path=lmk_path,
        output_path=output_path,
        model_step=model_step,
        seed=settings_seed,
        resolution_w=resolution_w,
        resolution_h=resolution_h,
        video_length=input_video_frames,
        num_inference_steps=settings_steps,
        guidance_scale=settings_cfg_scale,
        output_fps=output_fps,
        save_frames=save_frames,
        remove_anomaly_frames=remove_anomaly_frames,
        anomaly_detection_mode=anomaly_detection_mode
    )

    frames_archive = None
    if save_frames:
        frames_dir = os.path.join(output_path, f"frames")
        if os.path.exists(frames_dir):
            archive_path = os.path.join(output_path, f"frames.zip")
            with zipfile.ZipFile(archive_path, 'w') as zipf:
                for root, dirs, files in os.walk(frames_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file),
                                   os.path.relpath(os.path.join(root, file),
                                                   os.path.join(frames_dir, '..')))
            frames_archive = archive_path
            print(f"The archive has been created: {archive_path}")
        else:
            print(f"Directory with frames not found: {frames_dir}")

    return status, oo_video_path, all_video_path, frames_archive

with gr.Blocks() as demo:
    gr.Markdown("# FollowYourEmoji Webui")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload reference image", type="filepath", height=500)

            crop_face_checkbox = gr.Checkbox(label="Crop face according to video",info="If your picture is too far away or the face doesn't fit you can use cropping, you can see a preview in the tab below", value=False)
            with gr.Accordion("Face Cropping", open=False):
                expand_x = gr.Slider(label="Expand X", minimum=0.5, maximum=5.0, value=1.2, step=0.01)
                expand_y = gr.Slider(label="Expand Y", minimum=0.5, maximum=5.0, value=1.2, step=0.01)
                offset_x = gr.Slider(label="Offset X", minimum=-1, maximum=1, value=0.0, step=0.01)
                offset_y = gr.Slider(label="Offset Y", minimum=-1, maximum=1, value=0.0, step=0.01)

                with gr.Row():
                    preview_crop_btn = gr.Button(value="Preview Crop")
                    crop_preview = gr.Image(label="Crop Preview", height=300)

            with gr.Accordion("Input Video", open=True):
                input_video_type = gr.Radio(label="Input reference video type",info="You can either upload the video through the interface or use an already compiled npy file", choices=["video","npy"], value="video")

                with gr.Group() as video_group:
                    input_video = gr.Video(label="Upload reference video", height=500)
                    input_video_save = gr.Checkbox(label="Save video to processed video folder", value=True)

                with gr.Group(visible=False) as npy_group:
                    input_npy_select = gr.Dropdown(label="Select from processed video folder", choices=["None"], value="None")
                    input_npy_refresh = gr.Button(value="Update NPY list")
                    input_npy = gr.File(file_types=[".npy"], label="Upload preprocessed video in .npy")
            with gr.Accordion("Animation Preview",open=False):
                with gr.Row():
                    show_gif_btn = gr.Button(value="Show Animation preview")
                    gif_output = gr.Image(label="GIF Preview", height=300)

            with gr.Accordion("Animation Settings", open=True):
                input_video_frames = gr.Slider(label="Video frames", minimum=1, maximum=30, value=30, step=1)
                settings_steps = gr.Slider(label="Steps", minimum=1, maximum=200, value=30)
                settings_cfg_scale = gr.Slider(label="CFG scale", minimum=0.1, maximum=20, value=3.5, step=0.1)
                settings_seed = gr.Slider(minimum=0, maximum=1000, value=42, step=1, label="Seed")
                save_frames = gr.Checkbox(label="Save individual frames",info="Save individual frames", value=True)
                remove_anomaly_frames = gr.Checkbox(label="Remove anomaly frames",info="Sometimes there are abnormal frames that spoil the picture, this option removes them from the final video", value=True)
                anomaly_detection_mode = gr.Radio(label="Anomaly detection mode", choices=["light", "hard"], value="light")

            with gr.Accordion("Advanced Settings", open=False):
                resolution_w = gr.Slider(label="Resolution Width", minimum=64, maximum=1024, value=config['resolution_w'], step=64)
                resolution_h = gr.Slider(label="Resolution Height", minimum=64, maximum=1024, value=config['resolution_h'], step=64)
                model_step = gr.Slider(label="Model Step", value=0, minimum=0, maximum=100)
                custom_output_path = gr.Textbox(label="Custom Output Path", placeholder="Leave empty for default")
                use_custom_fps = gr.Checkbox(label="Use custom FPS",info="By default the FPS is set to 7", value=False)
                output_fps = gr.Slider(label="Output FPS",info="if you upload video fps slider updates to video fps", minimum=1, maximum=60, value=7, step=1, interactive=False)

        with gr.Column(scale=1):
            result_status = gr.Label(value="Status")
            result_video = gr.Video(label="Result Video (oo)", interactive=False, height=500)
            result_video_2 = gr.Video(label="Result Video (all)", interactive=False, height=500)
            result_btn = gr.Button(value="Generate Video")
            frames_output = gr.File(label="Frames Archive ( You'll get an archive with all the frames )")

    input_video_type.change(
        fn=lambda x: (gr.update(visible=(x=="video")), gr.update(visible=(x=="npy"))),
        inputs=[input_video_type],
        outputs=[video_group, npy_group]
    )

    input_npy_refresh.click(fn=update_npy_choices, outputs=[input_npy_select])

    input_video.change(
        fn=process_input_video,
        inputs=[input_video, input_video_save],
        outputs=[result_status, input_npy, input_video_frames, output_fps]
    )

    input_npy_select.change(fn=update_frame_count, inputs=[input_npy_select], outputs=[input_video_frames])
    input_npy.change(fn=update_frame_count, inputs=[input_npy], outputs=[input_video_frames])

    show_gif_btn.click(fn=show_gif_for_npy, inputs=[input_npy_select, input_video], outputs=[gif_output, result_status])

    input_video.change(
        fn=update_gif_on_video_change,
        inputs=[input_video],
        outputs=[gif_output, result_status]
    )

    use_custom_fps.change(fn=toggle_fps_slider, inputs=[use_custom_fps], outputs=[output_fps])

    preview_crop_btn.click(
        fn=preview_crop,
        inputs=[input_img, input_npy_select, input_video, expand_x, expand_y, offset_x, offset_y],
        outputs=[crop_preview, result_status]
    )

    result_btn.click(
        fn=generate_video,
        inputs=[input_img, crop_face_checkbox, expand_x, expand_y, offset_x, offset_y, input_video_type, input_video, input_npy_select, input_npy, input_video_frames,
                settings_steps, settings_cfg_scale, settings_seed, resolution_w, resolution_h,
                model_step, custom_output_path, use_custom_fps, output_fps, save_frames, remove_anomaly_frames, anomaly_detection_mode],
        outputs=[result_status, result_video, result_video_2, frames_output]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True)