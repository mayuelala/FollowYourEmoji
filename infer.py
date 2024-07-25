import os
import imageio
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T

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


@torch.no_grad()
def visualize(dataloader, pipeline, generator, W, H, video_length, num_inference_steps, guidance_scale, output_dir, limit=1):

    for i, batch in enumerate(dataloader):
        ref_frame=batch['ref_frame'][0]
        clip_image = batch['clip_image'][0]
        motions=batch['motions'][0]
        file_name = batch['file_name'][0]
        if motions is None:
            continue
        if 'lmk_name' in batch:
            lmk_name = batch['lmk_name'][0].split('.')[0]
        else:
            lmk_name = 'lmk'
        print(file_name, lmk_name)
        # tensor to pil image
        ref_frame = torch.clamp((ref_frame + 1.0) / 2.0, min=0, max=1)
        ref_frame = ref_frame.permute((1, 2, 3, 0)).squeeze()
        ref_frame = (ref_frame * 255).cpu().numpy().astype(np.uint8)
        ref_image = Image.fromarray(ref_frame)
        # tensor to pil image
        motions = motions.permute((1, 2, 3, 0))
        motions = (motions * 255).cpu().numpy().astype(np.uint8)
        lmk_images = []
        for motion in motions:
            lmk_images.append(Image.fromarray(motion))

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

        mp4_path = os.path.join(output_dir, lmk_name+'_'+file_name.split('.')[0]+'_oo.mp4')
        mp4_writer = imageio.get_writer(mp4_path, fps=7)
        for pred in preds:
            mp4_writer.append_data(pred)
        mp4_writer.close()

        mp4_path = os.path.join(output_dir, lmk_name+'_'+file_name.split('.')[0]+'_all.mp4')
        mp4_writer = imageio.get_writer(mp4_path, fps=8)
        if 'frames' in batch:
            frames = batch['frames'][0]
            frames = torch.clamp((frames + 1.0) / 2.0, min=0, max=1)
            frames = frames.permute((1, 2, 3, 0))
            frames = (frames * 255).cpu().numpy().astype(np.uint8)
            for frame, motion, pred in zip(frames, motions, preds):
                out = np.concatenate((frame, motion, ref_frame, pred), axis=1)
                mp4_writer.append_data(out)
        else:
            for motion, pred in zip(motions, preds):
                out = np.concatenate((motion, ref_frame, pred), axis=1)
                mp4_writer.append_data(out)
        mp4_writer.close()

        if i >= limit:
            break


def main(args, config):
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    if dist.get_rank() == 0:
        os.makedirs(args.output_path, exist_ok=True)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Do not support weight dtype: {config.weight_dtype} during training")

    # init model
    print('init model')
    vae = AutoencoderKL.from_pretrained(config.vae_model_path).to(dtype=weight_dtype, device="cuda")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(dtype=weight_dtype, device="cuda")

    referencenet = ReferenceNet2DConditionModel.from_pretrained_2d(config.base_model_path, subfolder="unet",
                                                                   referencenet_additional_kwargs=config.model.referencenet_additional_kwargs).to(device="cuda")
    unet = UNet3DConditionModel.from_pretrained_2d(config.base_model_path,
                                                   motion_module_path=config.motion_module_path, subfolder="unet",
                                                   unet_additional_kwargs=config.model.unet_additional_kwargs).to(device="cuda")

    lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device="cuda")

    # load model
    print('load model')
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

    # noise scheduler
    print('init noise scheduler')
    sched_kwargs = OmegaConf.to_container(config.scheduler)
    if config.enable_zero_snr:
        sched_kwargs.update(rescale_betas_zero_snr=True,
                            timestep_spacing="trailing",
                            prediction_type="v_prediction")
    noise_scheduler = DDIMScheduler(**sched_kwargs)

    # pipeline
    pipeline = VideoPipeline(vae=vae,
                             image_encoder=image_encoder,
                             referencenet=referencenet,
                             unet=unet,
                             lmk_guider=lmk_guider,
                             scheduler=noise_scheduler).to(vae.device, dtype=weight_dtype)

    # dataset creation
    print('init dataset')
    val_dataset = ValDataset(
        input_path=args.input_path,
        lmk_path=args.lmk_path,
        resolution_h=config.resolution_h,
        resolution_w=config.resolution_w
    )
    print(len(val_dataset))
    sampler = DistributedSampler(val_dataset, shuffle=False)
    # DataLoaders creation:
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        sampler=sampler,
        collate_fn=val_collate_fn,
    )

    generator = torch.Generator(device=vae.device)
    generator.manual_seed(config.seed)
    
    # run visualize
    print('run visualize')
    with torch.no_grad():
        visualize(val_dataloader, 
                pipeline, 
                generator, 
                W=config.resolution_w, 
                H=config.resolution_h, 
                video_length=config.video_length,
                num_inference_steps=30, 
                guidance_scale=3.5,
                output_dir=args.output_path,
                limit=100000000)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--lmk_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(args, config)