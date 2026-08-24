"""Shared helpers for Dual Encoding training."""

from __future__ import annotations

import re
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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


def distill_kl_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    temperature: float = 2.0,
) -> torch.Tensor:
    T = float(temperature)
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    p_t = F.softmax(logits_t / T, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)  # [B,S]
    if attention_mask is None:
        return kl.mean() * (T * T)
    m = attention_mask.to(dtype=kl.dtype)
    denom = m.sum().clamp_min(1.0)
    return (kl * m).sum() / denom * (T * T)


@torch.no_grad()
def heatmaps_to_weights_for_attention(
    adapter_cls: Any,
    processor: Any,
    vision_patch_size: int,
    heatmaps: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Reuse adapter-specific _preprocess_heatmaps_to_weights without constructing full adapter/model."""

    class _HeatmapPreprocessProxy:
        def __init__(self, processor: Any, vision_patch_size: int, device: torch.device):
            self.processor = processor
            self.vision_patch_size = vision_patch_size
            self.device = device

    proxy = _HeatmapPreprocessProxy(
        processor=processor,
        vision_patch_size=vision_patch_size,
        device=device,
    )
    if not hasattr(adapter_cls, "_preprocess_heatmaps_to_weights"):
        raise RuntimeError(f"Adapter {adapter_cls.__name__} does not define _preprocess_heatmaps_to_weights")
    return adapter_cls._preprocess_heatmaps_to_weights(proxy, heatmaps)


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


def build_per_sample_gaze_targets(
    weights: torch.Tensor,
    batch_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build per-sample gaze targets for attention alignment.

    Supports:
      - weights [B, N] (non-AnyRes or already per-sample)
      - weights [B*T, N] (AnyRes tile-level), flattened to [B, T*N]
    """
    if weights.ndim != 2:
        raise RuntimeError(f"Expected weights to be rank-2 [*,N], got shape={tuple(weights.shape)}")

    if weights.shape[0] == batch_size:
        g = weights.clamp_min(0)
        return g / (g.sum(dim=1, keepdim=True) + eps)

    if weights.shape[0] > batch_size and (weights.shape[0] % batch_size == 0):
        tiles_per_sample = weights.shape[0] // batch_size
        n_patch = weights.shape[1]
        g = weights.view(batch_size, tiles_per_sample, n_patch)
        g = g.reshape(batch_size, tiles_per_sample * n_patch).clamp_min(0)
        return g / (g.sum(dim=1, keepdim=True) + eps)

    raise RuntimeError(
        "Could not map gaze weights to per-sample targets for attention alignment. "
        f"weights.shape={tuple(weights.shape)}, batch_size={batch_size}. "
        "Expected [B,N] or [B*T,N] with integer T."
    )
