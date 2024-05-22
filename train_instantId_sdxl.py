import os
import re
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from datetime import datetime
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torchvision import transforms
from PIL import Image
import PIL
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


# Draw the input image for controlnet based on facial keypoints.
def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

# Process the dataset by loading info from a JSON file, which includes image files, image labels, feature files, keypoint coordinates.
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item["file_name"]
        text = item["additional_feature"]
        bbox = item['bbox']
        landmarks = item['landmarks']
        feature_file = item["insightface_feature_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        # draw keypoints
        kps_image = draw_kps(raw_image.convert("RGB"), landmarks)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        # transform raw_image and kps_image
        image_tensor = self.image_transforms(raw_image.convert("RGB"))
        kps_image_tensor = self.conditioning_image_transforms(kps_image)

        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h // 2 + 1)  # random top crop
            # top = np.random.randint(0, delta_h + 1)  # random crop
            left = np.random.randint(0, delta_w + 1)  # random crop

        # The image and kps_image must follow the same cropping to ensure that the facial coordinates correspond correctly.
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        kps_image = transforms.functional.crop(
            kps_image_tensor, top=top, left=left, height=self.size, width=self.size
        )

        crop_coords_top_left = torch.tensor([top, left])

        # load face feature
        face_id_embed = torch.load(os.path.join(self.image_root_path, feature_file), map_location="cpu")
        face_id_embed = torch.from_numpy(face_id_embed)
        face_id_embed = face_id_embed.reshape(1, -1)

        # set cfg drop rate
        drop_feature_embed = 0
        drop_text_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_feature_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            drop_text_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            drop_text_embed = 1
            drop_feature_embed = 1

        # CFG process
        if drop_text_embed:
            text = ""
        if drop_feature_embed:
            face_id_embed = torch.zeros_like(face_id_embed)

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "kps_image": kps_image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "face_id_embed": face_id_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    kps_images = torch.stack([example["kps_image"] for example in data])

    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "kps_images": kps_images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "face_id_embed": face_id_embed,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }


class InstantIDAdapter(torch.nn.Module):
    """InstantIDAdapter"""
    def __init__(self, unet, controlnet, feature_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.feature_proj_model = feature_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self,noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, feature_embeds, controlnet_image):
        face_embedding = self.feature_proj_model(feature_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, face_embedding], dim=1)
        # ControlNet conditioning.
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=face_embedding,  # Insightface feature
            added_cond_kwargs=unet_added_cond_kwargs,
            controlnet_cond=controlnet_image,  # keypoints image
            return_dict=False,
        )
        # Predict the noise residual.
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
            down_block_additional_residuals=[sample for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_feature_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.feature_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.feature_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_feature_proj_model = False

        # Load state dict for feature_proj_model and adapter_modules
        self.feature_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_feature_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of feature_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model. If not specified weights are initialized from unet.",
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument('--clip_proc_mode',
                        choices=["seg_align", "seg_crop", "orig_align", "orig_crop", "seg_align_pad",
                                 "orig_align_pad"],
                        default="orig_crop",
                        help='The mode to preprocess clip image encoder input.')

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    num_devices = accelerator.num_processes

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    if args.controlnet_model_name_or_path:
        print("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        print("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    controlnet.requires_grad_(True)
    controlnet.train()

    # ip-adapter: insightface feature
    num_tokens = 16

    feature_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=512,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    # Instantiate InstantIDAdapter from pretrained model or from scratch.
    ip_adapter = InstantIDAdapter(unet, controlnet, feature_proj_model, adapter_modules, args.pretrained_ip_adapter_path)

    # Register a hook function to process the state of a specific module before saving.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # find instance of InstantIDAdapter Model.
            for i, model_instance in enumerate(models):
                if isinstance(model_instance, InstantIDAdapter):
                    # When saving a checkpoint, only save the ip-adapter and image_proj, do not save the unet.
                    ip_adapter_state = {
                        'image_proj': model_instance.feature_proj_model.state_dict(),
                        'ip_adapter': model_instance.adapter_modules.state_dict(),
                    }
                    torch.save(ip_adapter_state, os.path.join(output_dir, 'pytorch_model.bin'))
                    print(f"IP-Adapter Model weights saved in {os.path.join(output_dir, 'pytorch_model.bin')}")
                    # Save controlnet separately.
                    sub_dir = "controlnet"
                    model_instance.controlnet.save_pretrained(os.path.join(output_dir, sub_dir))
                    print(f"Controlnet weights saved in {os.path.join(output_dir, controlnet)}")
                    # Remove the corresponding weights from the weights list because they have been saved separately.
                    # Remember not to delete the corresponding model, otherwise, you will not be able to save the model
                    # starting from the second epoch.
                    weights.pop(i)
                    break

    def load_model_hook(models, input_dir):
        # find instance of InstantIDAdapter Model.
        while len(models) > 0:
            model_instance = models.pop()
            if isinstance(model_instance, InstantIDAdapter):
                ip_adapter_path = os.path.join(input_dir, 'pytorch_model.bin')
                if os.path.exists(ip_adapter_path):
                    ip_adapter_state = torch.load(ip_adapter_path)
                    model_instance.feature_proj_model.load_state_dict(ip_adapter_state['image_proj'])
                    model_instance.adapter_modules.load_state_dict(ip_adapter_state['ip_adapter'])
                    sub_dir = "controlnet"
                    model_instance.controlnet.from_pretrained(os.path.join(input_dir, sub_dir))
                    print(f"Model weights loaded from {ip_adapter_path}")
                else:
                    print(f"No saved weights found at {ip_adapter_path}")


    # Register hook functions for saving  and loading.
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)  # error
    vae.to(accelerator.device)  # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)  # error
    controlnet.to(accelerator.device)

    # trainable params
    params_to_opt = itertools.chain(ip_adapter.feature_proj_model.parameters(),
                                    ip_adapter.adapter_modules.parameters(),
                                    ip_adapter.controlnet.parameters())

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution,
                              center_crop=args.center_crop, image_root_path=args.data_root_path)
    total_data_size = len(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    # # Restore checkpoints
    # checkpoint_folders = [folder for folder in os.listdir(args.output_dir) if folder.startswith('checkpoint-')]
    # if checkpoint_folders:
    #     # Extract step numbers from all checkpoints and find the maximum step number
    #     global_step = max(int(folder.split('-')[-1]) for folder in checkpoint_folders if folder.split('-')[-1].isdigit())
    #     checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    #     # Load the checkpoint
    #     accelerator.load_state(checkpoint_path)
    # else:
    #     global_step = 0
    #     print("No checkpoint folders found.")
    global_step = 0
    # Calculate steps per epoch and the current epoch and its step number
    # steps_per_epoch = total_data_size // (args.train_batch_size * num_devices)
    # current_epoch = global_step // steps_per_epoch
    # current_step_in_epoch = global_step % steps_per_epoch

    # Training loop
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                        accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get feature embeddings, with cfg
                feat_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                kps_images = batch["kps_images"].to(accelerator.device, dtype=weight_dtype)

                # for other experiments
                # clip_images = []
                # for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_image_embeds"]):
                #     if drop_image_embed == 1:
                #         clip_images.append(torch.zeros_like(clip_image))
                #     else:
                #         clip_images.append(clip_image)
                # clip_images = torch.stack(clip_images, dim=0)
                # with torch.no_grad():
                #     image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype),
                #                                  output_hidden_states=True).hidden_states[-2]

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)  # concat

                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, feat_embeds, kps_images)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                now = datetime.now()
                formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if accelerator.is_main_process and step % 10 == 0:
                    print("[{}]: Epoch {}, global_step {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        formatted_time, epoch, global_step, step, load_data_time, time.perf_counter() - begin,
                        avg_loss))

            global_step += 1
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                # before saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
