from torch import Tensor

def prepare_clip(clip, original_img, img) -> dict[str, Tensor]:
    _, projection_clip, uc_emb, projection_clip_plus = clip(original_img)
    
    vec = projection_clip
    vec_plus = projection_clip_plus

    return {
        "img": img,
        "vec": vec.to(img.device),
        "uc_emb": uc_emb.to(img.device),
        "vec_plus": vec_plus.to(img.device)
    }