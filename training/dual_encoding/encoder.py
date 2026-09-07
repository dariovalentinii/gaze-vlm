"""Heatmap encoder setup and preprocessing for Dual Encoding training."""

from __future__ import annotations

import re
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import nn

from peft import PeftModel


def _heatmap_to_rgb_pil(hm: torch.Tensor) -> Image.Image:
    if hm.ndim != 4 or hm.shape[0] != 1 or hm.shape[1] != 1:
        raise ValueError(f"Expected heatmap [1,1,H,W], got {tuple(hm.shape)}")
    hm_2d = hm[0, 0].float().cpu().numpy()
    hm_2d = hm_2d - hm_2d.min()
    if hm_2d.max() > 0:
        hm_2d = hm_2d / hm_2d.max()
    hm_uint8 = (hm_2d * 255.0).astype(np.uint8)
    return Image.fromarray(hm_uint8, mode="L").convert("RGB")


def heatmaps_to_rgb_pils(heatmaps: List[torch.Tensor]) -> List[Image.Image]:
    return [_heatmap_to_rgb_pil(hm) for hm in heatmaps]


def forward_heatmap_encoder(
    heatmap_encoder: nn.Module,
    pixel_values: torch.Tensor,
    output_hidden_states: bool = False,
    return_dict: bool = True,
) -> Any:
    """
    Forward helper for CLIP vision tower with/without PEFT.

    Some PEFT wrappers for FEATURE_EXTRACTION can forward an `inputs_embeds`
    kwarg that conflicts with CLIPVisionModel internals. If that happens,
    fallback to calling the wrapped base_model directly (LoRA modules stay active).
    """
    kwargs = {
        "pixel_values": pixel_values,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict,
    }

    if isinstance(heatmap_encoder, PeftModel):
        try:
            return heatmap_encoder(**kwargs)
        except (TypeError, KeyError) as e:
            msg = str(e)
            if "inputs_embeds" not in msg:
                raise
            if not hasattr(heatmap_encoder, "base_model"):
                raise
            return heatmap_encoder.base_model(**kwargs)

    return heatmap_encoder(**kwargs)


def freeze_all_params(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def clone_vision_tower(vision_tower: nn.Module) -> nn.Module:
    """Create a new vision tower instance with identical weights."""
    cls = vision_tower.__class__
    new = cls(vision_tower.config)
    new.load_state_dict(vision_tower.state_dict(), strict=True)
    return new


def restrict_vision_lora_to_last_k_layers(vision_peft: nn.Module, last_k: int, num_layers: int) -> None:
    """Freeze LoRA params that belong to layers < num_layers-last_k."""
    if last_k <= 0 or last_k >= num_layers:
        return
    cutoff = num_layers - last_k
    pat = re.compile(r"vision_model\.encoder\.layers\.(\d+)\.")
    for n, p in vision_peft.named_parameters():
        if "lora_" not in n:
            continue
        m = pat.search(n)
        if m is None:
            # keep non-layer LoRA params trainable
            continue
        li = int(m.group(1))
        if li < cutoff:
            p.requires_grad = False


def heatmaps_to_pixel_values_for_encoder(
    processor: Any,
    device: torch.device,
    heatmap_pils: Optional[List[Image.Image]] = None,
) -> torch.Tensor:
    """Returns CLIP-normalized pixel_values [B,3,H',W'] for the heatmap encoder."""
    pix = processor.image_processor(images=heatmap_pils, return_tensors="pt")["pixel_values"]
    if pix.ndim == 5:
        b, t, c, h, w = pix.shape
        pix = pix.view(b * t, c, h, w)
    return pix.to(device)
