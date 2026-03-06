import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange

from image_datasets.dataset_cc3m import loader
from torchvision import transforms

from clip_models.build_CLIP import load_clip_model_SigLIP
from clip_models.sampling import prepare_clip

from peft import LoraConfig, get_peft_model
from copy import deepcopy

from src.stable_diffusion import build_sd_model

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    sd_parts = build_sd_model(model_name=args.clip_config.sd_model_name, device=accelerator.device, dtype=torch.bfloat16)
    vae = sd_parts["vae"]
    unet = sd_parts["unet"]
    clip_vis = load_clip_model_SigLIP(args.clip_config, device=accelerator.device)

    # set LoRA
    lora_config = LoraConfig(
        r=args.lora_config.r,
        lora_alpha=args.lora_config.lora_alpha,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2'],
        lora_dropout=args.lora_config.lora_dropout,
        bias=args.lora_config.bias,
    )
    clip_vis.model = get_peft_model(clip_vis.model, lora_config)
    clip_vis.model.print_trainable_parameters()


    print('loading projection params...')
    load_path_project_clip = os.path.join(args.load_dir, f"checkpoint-project-clip-{args.load_step}.bin")
    clip_vis.project_clip.load_state_dict(torch.load(load_path_project_clip, map_location=torch.device('cpu')))
    print('loading successfully!')

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet = unet.to(torch.bfloat16)
    unet.to(accelerator.device)
    clip_vis.train()
    unet.train()

    for name_, param in clip_vis.named_parameters():
        if 'project_clip' in name_:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in clip_vis.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(int(3e6) / args.data_config.train_batch_size) / args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    clip_vis, optimizer, _, lr_scheduler = accelerator.prepare(
        clip_vis, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(clip_vis):
                original_img, prompts = batch['image'], batch['text']
                original_img = original_img.to(accelerator.device)
                
                with torch.no_grad():
                    x_1 = vae.encode(NORMALIZE_VAE(original_img).to(device=vae.device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                inp = prepare_clip(clip=clip_vis, original_img=NORMALIZE_CLIP(original_img).to(weight_dtype), img=x_1.to(weight_dtype))
                bs = original_img.shape[0]
                t = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=accelerator.device).long()
                noise = torch.randn_like(x_1).to(accelerator.device)
                noisy_latents = noise_scheduler.add_noise(x_1, noise, t)
                model_pred = unet(
                    noisy_latents[:, None].repeat(1, bs, 1, 1, 1).reshape(bs * bs, *noisy_latents.shape[1:]),
                    t[:, None].repeat(1, bs).reshape(bs * bs),
                    encoder_hidden_states=inp['vec'].to(weight_dtype)[None].repeat(bs, 1, 1, 1).reshape(bs * bs, inp['vec'].shape[1], inp['vec'].shape[2])
                ).sample.reshape(bs, bs, *noisy_latents.shape[1:])
                model_pred_plus = unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=inp['vec_plus'].to(weight_dtype)
                ).sample
                target_v = noise_scheduler.get_velocity(x_1, noise, t)
                tau = 0.1

                anchor_pred = model_pred[torch.arange(bs), torch.arange(bs)]
                anchor_vec  = F.normalize(anchor_pred.flatten(1), dim=1)
                gt_vec      = F.normalize(target_v.flatten(1), dim=1)
                pos_plus_vec= F.normalize(model_pred_plus.flatten(1), dim=1)

                all_vec     = F.normalize(model_pred.reshape(bs, bs, -1), dim=2)
                sim_anchor_all = torch.einsum("bd,bkd->bk", anchor_vec, all_vec)
                logits_all = sim_anchor_all / tau

                neg_mask = ~torch.eye(bs, dtype=torch.bool, device=logits_all.device)
                neg_logit = logits_all.masked_fill(~neg_mask, float("-inf"))

                logits_pos_pred = (anchor_vec * pos_plus_vec).sum(dim=1) / tau
                logits_pos_gt   = (anchor_vec * gt_vec).sum(dim=1) / tau

                denom = torch.exp(logits_pos_pred) + torch.exp(logits_pos_gt) + torch.exp(neg_logit).sum(dim=1)
                loss_pos_pred = -torch.log(torch.exp(logits_pos_pred) / denom)
                loss_pos_gt   = -torch.log(torch.exp(logits_pos_gt)   / denom)
                loss = 0.5 * (loss_pos_pred + loss_pos_gt)
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(clip_vis.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or global_step in [100, 200, 300, 500, 1000, 2000, 3000]:
                    if accelerator.is_main_process:
                        if args.clip_config.clip_image_size == 384:
                            save_path = os.path.join(args.output_dir, f"siglip-so400m-patch14-384-{global_step}")
                        else:
                            save_path = os.path.join(args.output_dir, f"siglip-so400m-patch14-224-{global_step}")
                        unwrapped_clip_vis = accelerator.unwrap_model(clip_vis)
                        save_model = deepcopy(unwrapped_clip_vis.model).merge_and_unload()
                        save_model.save_pretrained(save_path, safe_serialization=False)
                        logger.info(f"Saved ckpts to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    if args.clip_config.clip_image_size == 384:
                        save_path = os.path.join(args.output_dir, f"siglip-so400m-patch14-384-{global_step}")
                    else:
                        save_path = os.path.join(args.output_dir, f"siglip-so400m-patch14-224-{global_step}")
                    unwrapped_clip_vis = accelerator.unwrap_model(clip_vis)
                    save_model = deepcopy(unwrapped_clip_vis.model).merge_and_unload()
                    save_model.save_pretrained(save_path, safe_serialization=False)
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
