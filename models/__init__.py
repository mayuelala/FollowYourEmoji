import torch
import torch.nn as nn
import torchvision.transforms as T


class LVDM(nn.Module):
    def __init__(self, referencenet, unet, pose_guider):
        super().__init__()
        self.referencenet = referencenet
        self.unet = unet
        self.pose_guider = pose_guider

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        # noisy_latents.shape = torch.Size([4, 4, 1, 112, 80])
        # timesteps = tensor([426, 277, 802, 784], device='cuda:5')
        # ref_image_latents.shape = torch.Size([4, 4, 112, 80])
        # clip_image_embeds.shape = torch.Size([4, 1, 768])
        # pose_img.shape = torch.Size([4, 3, 1, 896, 640])
        # uncond_fwd = False

        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)
        # pose_fea.shape = torch.Size([4, 320, 1, 112, 80])

        # not uncond_fwd = True
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            reference_down_block_res_samples, reference_mid_block_res_sample, reference_up_block_res_samples = \
                self.referencenet(ref_image_latents,
                                  ref_timesteps,
                                  encoder_hidden_states=clip_image_embeds,
                                  return_dict=False)

        self.unet.set_do_classifier_free_guidance(do_classifier_free_guidance=False)
        model_pred = self.unet(noisy_latents,
                               timesteps,
                               pose_cond_fea=pose_fea,
                               encoder_hidden_states=clip_image_embeds,
                               reference_down_block_res_samples=reference_down_block_res_samples if not uncond_fwd else None,
                               reference_mid_block_res_sample=reference_mid_block_res_sample if not uncond_fwd else None,
                               reference_up_block_res_samples=reference_up_block_res_samples if not uncond_fwd else None).sample
        return model_pred