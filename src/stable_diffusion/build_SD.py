import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

_REPO_MAP = {
    "sd-1.4": "pretrained_weights/SD/stable-diffusion-v1-4",
    "sd-1.5": "pretrained_weights/SD/stable-diffusion-v1-5",
    "sd-2.1": "pretrained_weights/SD/stable-diffusion-v2-1"
}

def build_sd_model(model_name: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    repo = _REPO_MAP[model_name]

    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=dtype)

    vae.to(device).eval()
    unet.to(device).train()

    return {
        "unet": unet,
        "vae": vae
    }

def build_clip_from_sd_model(model_name: str):
    repo = _REPO_MAP[model_name]

    tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    sd_text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder")

    return {
        "tokenizer": tokenizer,
        "sd_text_encoder": sd_text_encoder
    }