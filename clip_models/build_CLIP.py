import torch
from .CLIP_bank import OpenAICLIP, SigLIP, MetaCLIP 


def load_clip_model_OpenAICLIP(config, device):

    class_model = OpenAICLIP(config)
    class_model.to(device)
    class_model.to(torch.float32)

    return class_model


def load_clip_model_SigLIP(config, device):

    class_model = SigLIP(config)
    class_model.to(device)
    class_model.to(torch.float32)

    return class_model


def load_clip_model_MetaCLIP(config, device):

    class_model = MetaCLIP(config)
    class_model.to(device)
    class_model.to(torch.float32)

    return class_model