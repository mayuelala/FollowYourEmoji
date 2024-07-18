import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from transformers import CLIPImageProcessor

import sys
sys.path.append("/path/to/FollowYourEmoji")
from media_pipe import FaceMeshDetector, FaceMeshAlign
from media_pipe.draw_util import FaceMeshVisualizer


def val_collate_fn(samples):
    return {
        'ref_frame': [sample['ref_frame'] for sample in samples],
        'clip_image': [sample['clip_image'] for sample in samples],
        'motions': [sample['motions'] for sample in samples],
        'file_name': [sample['file_name'] for sample in samples],
        'lmk_name': [sample['lmk_name'] for sample in samples],
    }


class ValDataset(Dataset):
    def __init__(self, input_path, lmk_path, resolution_w=512, resolution_h=512):
        all_img_paths = self._get_path_files(Path(input_path), file_suffix=['.jpg', '.jpeg', '.png', '.webp'])
        all_lmk_paths = self._get_path_files(Path(lmk_path), file_suffix=['.npy'])
        self.all_paths = []
        for lmk_path in all_lmk_paths:
            for img_path in all_img_paths:
                self.all_paths.append((img_path, lmk_path))
        
        self.W = resolution_w
        self.H = resolution_h
        self.to_tensor = T.ToTensor()

        self.detector = FaceMeshDetector()
        self.aligner = FaceMeshAlign()

        self.clip_image_processor = CLIPImageProcessor()
        self.vis = FaceMeshVisualizer(forehead_edge=False, iris_edge=False, iris_point=True)

    def __len__(self):
        return len(self.all_paths)

    def _get_path_files(self, path, file_suffix):
        all_paths = []
        if path.is_file():
            if path.suffix.lower() in file_suffix:
                all_paths = [path]
            else:
                raise ValueError('Path is not valid image file.')
        elif path.is_dir():
            all_paths = sorted(
                [
                    f
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in file_suffix
                ]
            )
            if len(all_paths) == 0:
                raise ValueError('Folder does not contain any images.')
        else:
            raise ValueError

        return all_paths

    def get_align_motion(self, ref_lmk, temp_lmks):
        motions = self.aligner(ref_lmk, temp_lmks)
        motions = [self.to_tensor(motion) for motion in motions]
        motions = torch.stack(motions).permute((1,0,2,3))
        return motions

    def __getitem__(self, index):
        img_path, lmk_path = self.all_paths[index]
        W, H = self.W, self.H

        image = Image.open(img_path).convert('RGB')

        # resize and center crop
        scale = min(W / image.size[0], H / image.size[1])
        ref_image = image.resize(
            (int(image.size[0] * scale), int(image.size[1] * scale)))
        w, h = ref_image.size[0], ref_image.size[1]
        ref_image = ref_image.crop((w//2-W//2, h//2-H//2, w//2+W//2, h//2+H//2))
        ref_image = np.array(ref_image)

        # reference image lmk
        ref_lmk_image, ref_lmk = self.detector(ref_image)

        # clip image
        clip_image = Image.fromarray(np.array(ref_image))
        clip_image = self.clip_image_processor(images=clip_image, return_tensors="pt").pixel_values[0]

        # reference image
        ref_image = self.to_tensor(ref_image).unsqueeze(1)
        ref_image = ref_image * 2.0 - 1.0

        # motion sequence
        temp_lmks = np.load(lmk_path, allow_pickle=True)
        # landmark align and draw motions
        if ref_lmk is not None:
            motions = self.get_align_motion(ref_lmk, temp_lmks)
        else:
            motions = [
                self.vis.draw_landmarks((H, W), lmk['lmks'].astype(np.float32), normed=True)
                for lmk in temp_lmks
            ]
            motions = [self.to_tensor(motion) for motion in motions]
            motions = torch.stack(motions).permute((1,0,2,3))

        example = dict()
        example["file_name"] = str(img_path.stem).split('/')[-1]
        example["lmk_name"] = str(lmk_path.stem).split('/')[-1]
        example["motions"] = motions # value in [0, 1]
        example["ref_frame"] = ref_image # value in [-1, 1]
        example["ref_lmk_image"] = ref_lmk_image
        example["clip_image"] = clip_image

        return example
