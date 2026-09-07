"""Losses and gaze targets for Dual Encoding training."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn.functional as F


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
