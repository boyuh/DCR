import torch.nn as nn
from transformers import CLIPModel
from transformers import SiglipModel
import torch
from torchvision import transforms

from src.stable_diffusion import build_clip_from_sd_model

class OpenAICLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model = CLIPModel.from_pretrained('pretrained_weights/OpenAICLIP/clip-vit-large-patch14')
        if config.clip_image_size == 336:
            model = CLIPModel.from_pretrained('pretrained_weights/OpenAICLIP/clip-vit-large-patch14-336')

        clip_from_sd = build_clip_from_sd_model(config.sd_model_name)
        tokenizer = clip_from_sd["tokenizer"]
        sd_text_encoder = clip_from_sd["sd_text_encoder"]
        
        self.project_clip = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, config.clip_dim),
            nn.GELU(),
            nn.Linear(config.clip_dim, config.clip_dim),
        )
        self.model = model
        self.tokenizer = tokenizer
        self.sd_text_encoder = sd_text_encoder
        self.config = config

        self.aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=(config.clip_image_size, config.clip_image_size),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1)
            ),
        ])

    def forward(self, images):
        uc = self.tokenizer([""] * images.shape[0], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uc = {k: v.to(images.device) for k, v in uc.items()}
        with torch.no_grad():
            txt_out = self.sd_text_encoder(
                input_ids=uc["input_ids"],
                attention_mask=uc["attention_mask"],
            )
            weight_dtype = next(self.model.parameters()).dtype
            uc_emb = txt_out.last_hidden_state.to(dtype=weight_dtype)

        class_token_pre = self.model.vision_model(images).pooler_output
        class_token = self.model.visual_projection(class_token_pre)

        projection_clip = self.project_clip(class_token)
        projection_clip = projection_clip.unsqueeze(1).expand(-1, 77, -1) + uc_emb

        with torch.no_grad():
            images_plus = torch.stack(
                [self.aug(img.cpu()).to(images.device) for img in images],
                dim=0
            )
        class_token_pre_plus = self.model.vision_model(images_plus).pooler_output
        class_token_plus = self.model.visual_projection(class_token_pre_plus)
        projection_clip_plus = self.project_clip(class_token_plus)
        projection_clip_plus = projection_clip_plus.unsqueeze(1).expand(-1, 77, -1) + uc_emb

        return class_token, projection_clip, uc_emb, projection_clip_plus


class SigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model = SiglipModel.from_pretrained('pretrained_weights/SigLIP/siglip-so400m-patch14-224')
        if config.clip_image_size == 384:
            model = SiglipModel.from_pretrained('pretrained_weights/SigLIP/siglip-so400m-patch14-384')

        clip_from_sd = build_clip_from_sd_model(config.sd_model_name)
        tokenizer = clip_from_sd["tokenizer"]
        sd_text_encoder = clip_from_sd["sd_text_encoder"]

        self.project_clip = nn.Sequential(
            nn.LayerNorm(1152),
            nn.Linear(1152, config.clip_dim),
            nn.GELU(),
            nn.Linear(config.clip_dim, config.clip_dim),
        )
        self.model = model
        self.tokenizer = tokenizer
        self.sd_text_encoder = sd_text_encoder
        self.config = config

        self.aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=(config.clip_image_size, config.clip_image_size),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1)
            ),
        ])

    def forward(self, images):
        model_device = next(self.model.parameters()).device
        model_dtype  = next(self.model.parameters()).dtype
        images = images.to(device=model_device, dtype=model_dtype)
        
        uc = self.tokenizer([""] * images.shape[0], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uc = {k: v.to(images.device) for k, v in uc.items()}
        with torch.no_grad():
            txt_out = self.sd_text_encoder(
                input_ids=uc["input_ids"],
                attention_mask=uc["attention_mask"],
            )
            weight_dtype = next(self.model.parameters()).dtype
            uc_emb = txt_out.last_hidden_state.to(dtype=weight_dtype)

        class_token = self.model.vision_model(images).pooler_output
        projection_clip = self.project_clip(class_token)
        projection_clip = projection_clip.unsqueeze(1).expand(-1, 77, -1) + uc_emb

        with torch.no_grad():
            images_plus = torch.stack(
                [self.aug(img.cpu()).to(images.device) for img in images],
                dim=0
            )
        class_token_plus = self.model.vision_model(images_plus).pooler_output
        projection_clip_plus = self.project_clip(class_token_plus)
        projection_clip_plus = projection_clip_plus.unsqueeze(1).expand(-1, 77, -1) + uc_emb

        return class_token, projection_clip, uc_emb, projection_clip_plus


class MetaCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_type == 'large':
            model = CLIPModel.from_pretrained('pretrained_weights/MetaCLIP/metaclip-l14-fullcc2.5b')

            self.project_clip = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, config.clip_dim),
                nn.GELU(),
                nn.Linear(config.clip_dim, config.clip_dim),
            )
        elif config.clip_type == 'huge':
            model = CLIPModel.from_pretrained('pretrained_weights/MetaCLIP/metaclip-h14-fullcc2.5b')

            self.project_clip = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Linear(1024, config.clip_dim),
                nn.GELU(),
                nn.Linear(config.clip_dim, config.clip_dim),
            )

        clip_from_sd = build_clip_from_sd_model(config.sd_model_name)
        tokenizer = clip_from_sd["tokenizer"]
        sd_text_encoder = clip_from_sd["sd_text_encoder"]

        self.model = model
        self.tokenizer = tokenizer
        self.sd_text_encoder = sd_text_encoder
        self.config = config

        self.aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=(config.clip_image_size, config.clip_image_size),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1)
            ),
        ])

    def forward(self, images):
        uc = self.tokenizer([""] * images.shape[0], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uc = {k: v.to(images.device) for k, v in uc.items()}
        with torch.no_grad():
            txt_out = self.sd_text_encoder(
                input_ids=uc["input_ids"],
                attention_mask=uc["attention_mask"],
            )
            weight_dtype = next(self.model.parameters()).dtype
            uc_emb = txt_out.last_hidden_state.to(dtype=weight_dtype)

        class_token_pre = self.model.vision_model(images).pooler_output
        class_token = self.model.visual_projection(class_token_pre)
        projection_clip = self.project_clip(class_token)
        projection_clip = projection_clip.unsqueeze(1).expand(-1, 77, -1) + uc_emb

        with torch.no_grad():
            images_plus = torch.stack(
                [self.aug(img.cpu()).to(images.device) for img in images],
                dim=0
            )
        class_token_pre_plus = self.model.vision_model(images_plus).pooler_output
        class_token_plus = self.model.visual_projection(class_token_pre_plus)
        projection_clip_plus = self.project_clip(class_token_plus)
        projection_clip_plus = projection_clip_plus.unsqueeze(1).expand(-1, 77, -1) + uc_emb

        return class_token, projection_clip, uc_emb, projection_clip_plus
