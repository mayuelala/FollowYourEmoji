import os
import imageio
import numpy as np
from PIL import Image
import cv2
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim
from collections import deque

import torch
import gc
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPVisionModelWithProjection

from models.guider import Guider
from models.referencenet import ReferenceNet2DConditionModel
from models.unet import UNet3DConditionModel
from models.video_pipeline import VideoPipeline

from dataset.val_dataset import ValDataset, val_collate_fn

def load_model_state_dict(model, model_ckpt_path, name):
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()
    model_new_sd = {}
    count = 0
    for k, v in ckpt.items():
        if k in model_state_dict:
            count += 1
            model_new_sd[k] = v
    miss, _ = model.load_state_dict(model_new_sd, strict=False)
    print(f'load {name} from {model_ckpt_path}\n - load params: {count}\n - miss params: {miss}')

def frame_analysis(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    ssim_score = ssim(prev_gray, curr_gray)
    mean_diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))

    return ssim_score, mean_diff

def is_anomaly_light(ssim_score, mean_diff, ssim_history, mean_diff_history):
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

def is_anomaly_hard(ssim_score, mean_diff, ssim_history, mean_diff_history):
    if len(ssim_history) < 5:
        return False

    ssim_avg = np.mean(ssim_history)
    mean_diff_avg = np.mean(mean_diff_history)

    ssim_threshold = 0.85
    mean_diff_threshold = 6.0

    ssim_change_threshold = 0.05
    mean_diff_change_threshold = 3.0

    if (ssim_score < ssim_threshold and mean_diff > mean_diff_threshold) or \
       (ssim_score < ssim_avg - ssim_change_threshold and mean_diff > mean_diff_avg + mean_diff_change_threshold):
        return True

    return False

@torch.no_grad()
def visualize(dataloader, pipeline, generator, W, H, video_length, num_inference_steps, guidance_scale, output_path, save_frames=False, output_fps=7, limit=1, show_stats=False, remove_anomaly_frames=False, anomaly_detection_mode="light"):
    oo_video_path = None
    all_video_path = None

    is_anomaly = is_anomaly_light if anomaly_detection_mode == "light" else is_anomaly_hard

    for i, batch in enumerate(dataloader):
        ref_frame = batch['ref_frame'][0]
        clip_image = batch['clip_image'][0]
        motions = batch['motions'][0]
        file_name = batch['file_name'][0]
        if motions is None:
            continue
        if 'lmk_name' in batch:
            lmk_name = batch['lmk_name'][0].split('.')[0]
        else:
            lmk_name = 'lmk'
        print(file_name, lmk_name)

        ref_frame = torch.clamp((ref_frame + 1.0) / 2.0, min=0, max=1)
        ref_frame = ref_frame.permute((1, 2, 3, 0)).squeeze()
        ref_frame = (ref_frame * 255).cpu().numpy().astype(np.uint8)
        ref_image = Image.fromarray(ref_frame)

        motions = motions.permute((1, 2, 3, 0))
        motions = (motions * 255).cpu().numpy().astype(np.uint8)
        lmk_images = [Image.fromarray(motion) for motion in motions]

        preds = pipeline(ref_image=ref_image,
                         lmk_images=lmk_images,
                         width=W,
                         height=H,
                         video_length=video_length,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         generator=generator,
                         clip_image=clip_image,
                        ).videos

        preds = preds.permute((0,2,3,4,1)).squeeze(0)
        preds = (preds * 255).cpu().numpy().astype(np.uint8)

        filtered_preds = []
        anomaly_frames = []
        prev_frame = None
        ssim_history = deque(maxlen=5)
        mean_diff_history = deque(maxlen=5)

        for idx, frame in enumerate(preds):
            if prev_frame is not None:
                ssim_score, mean_diff = frame_analysis(prev_frame, frame)
                ssim_history.append(ssim_score)
                mean_diff_history.append(mean_diff)

                if show_stats:
                    print(f"Frame {idx}: SSIM: {ssim_score:.4f}, Mean Diff: {mean_diff:.4f}")

                if remove_anomaly_frames and is_anomaly(ssim_score, mean_diff, ssim_history, mean_diff_history):
                    print(f"Anomaly detected in frame {idx}")
                    anomaly_frames.append((idx, frame))
                else:
                    filtered_preds.append(frame)
            else:
                filtered_preds.append(frame)

            prev_frame = frame

        if remove_anomaly_frames:
            preds = filtered_preds

        oo_video_path = os.path.join(output_path, f"{lmk_name}_oo.mp4")
        imageio.mimsave(oo_video_path, preds, fps=output_fps)

        if save_frames:
            frames_dir = os.path.join(output_path, f"frames")
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame in enumerate(preds):
                frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
                imageio.imwrite(frame_path, frame)

            if remove_anomaly_frames and anomaly_frames:
                anomaly_dir = os.path.join(frames_dir, "anomaly")
                os.makedirs(anomaly_dir, exist_ok=True)
                for idx, frame in anomaly_frames:
                    frame_path = os.path.join(anomaly_dir, f"anomaly_frame_{idx:04d}.png")
                    imageio.imwrite(frame_path, frame)

        if 'frames' in batch:
            frames = batch['frames'][0]
            frames = torch.clamp((frames + 1.0) / 2.0, min=0, max=1)
            frames = frames.permute((1, 2, 3, 0))
            frames = (frames * 255).cpu().numpy().astype(np.uint8)
            combined = [np.concatenate((frame, motion, ref_frame, pred), axis=1) for frame, motion, pred in zip(frames, motions, preds)]
        else:
            combined = [np.concatenate((motion, ref_frame, pred), axis=1) for motion, pred in zip(motions, preds)]

        all_video_path = os.path.join(output_path, f"{lmk_name}_all.mp4")
        imageio.mimsave(all_video_path, combined, fps=output_fps)

        if i >= limit:
            break

    return oo_video_path, all_video_path

# Остальная часть кода остается без изменений


def infer(config_path, model_path, input_path, lmk_path, output_path, model_step, seed,
          resolution_w, resolution_h, video_length, num_inference_steps, guidance_scale, output_fps, save_frames, show_stats, remove_anomaly_frames, anomaly_detection_mode):

    config = OmegaConf.load(config_path)
    config.init_checkpoint = model_path
    config.init_num = model_step
    config.resolution_w = resolution_w
    config.resolution_h = resolution_h
    config.video_length = video_length

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Do not support weight dtype: {config.weight_dtype}")

    vae = AutoencoderKL.from_pretrained(config.vae_model_path).to(dtype=weight_dtype, device="cuda")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(dtype=weight_dtype, device="cuda")
    referencenet = ReferenceNet2DConditionModel.from_pretrained_2d(config.base_model_path,
                                                                   referencenet_additional_kwargs=config.model.referencenet_additional_kwargs).to(device="cuda")
    unet = UNet3DConditionModel.from_pretrained_2d(config.base_model_path,
                                                   motion_module_path=config.motion_module_path,
                                                   unet_additional_kwargs=config.model.unet_additional_kwargs).to(device="cuda")
    lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device="cuda")

    load_model_state_dict(referencenet, f'{config.init_checkpoint}/referencenet.pth', 'referencenet')
    load_model_state_dict(unet, f'{config.init_checkpoint}/unet.pth', 'unet')
    load_model_state_dict(lmk_guider, f'{config.init_checkpoint}/lmk_guider.pth', 'lmk_guider')

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            referencenet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    unet.set_reentrant(use_reentrant=False)
    referencenet.set_reentrant(use_reentrant=False)

    vae.eval()
    image_encoder.eval()
    unet.eval()
    referencenet.eval()
    lmk_guider.eval()

    sched_kwargs = OmegaConf.to_container(config.scheduler)
    if config.enable_zero_snr:
        sched_kwargs.update(rescale_betas_zero_snr=True,
                            timestep_spacing="trailing",
                            prediction_type="v_prediction")
    noise_scheduler = DDIMScheduler(**sched_kwargs)

    pipeline = VideoPipeline(vae=vae,
                             image_encoder=image_encoder,
                             referencenet=referencenet,
                             unet=unet,
                             lmk_guider=lmk_guider,
                             scheduler=noise_scheduler).to(vae.device, dtype=weight_dtype)

    val_dataset = ValDataset(
        input_path=input_path,
        lmk_path=lmk_path,
        resolution_h=config.resolution_h,
        resolution_w=config.resolution_w
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=val_collate_fn,
    )

    generator = torch.Generator(device=vae.device)
    generator.manual_seed(seed)

    oo_video_path, all_video_path = visualize(
        val_dataloader,
        pipeline,
        generator,
        W=config.resolution_w,
        H=config.resolution_h,
        video_length=config.video_length,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_path=output_path,
        save_frames=save_frames,
        output_fps=output_fps,
        show_stats=show_stats,
        remove_anomaly_frames=remove_anomaly_frames,
        anomaly_detection_mode=anomaly_detection_mode,
        limit=100000000
    )

    del vae, image_encoder, referencenet, unet, lmk_guider, pipeline
    torch.cuda.empty_cache()
    gc.collect()

    return "Inference completed successfully", oo_video_path, all_video_path

def run_inference(config_path, model_path, input_path, lmk_path, output_path, model_step, seed,
                  resolution_w, resolution_h, video_length, num_inference_steps=30, guidance_scale=3.5, output_fps=30,
                  save_frames=False, show_stats=False, remove_anomaly_frames=False, anomaly_detection_mode="light"):
    try:
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return infer(config_path, model_path, input_path, lmk_path, output_path, model_step, seed,
                     resolution_w, resolution_h, video_length, num_inference_steps, guidance_scale, output_fps,
                     save_frames, show_stats, remove_anomaly_frames, anomaly_detection_mode)
    finally:
        torch.cuda.empty_cache()
        gc.collect()
