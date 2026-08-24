from __future__ import annotations
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class GazeInjector(nn.Module):
    """
    Same injector used by train_s2_llava15_attn_align.py:
      g = min_gate + (1-min_gate) * sigmoid(scale * w + bias)
    w is expected in [0,1], shape [B,N]. Output g in [min_gate,1], shape [B,N].
    """
    def __init__(self, init_scale: float = 1.0, init_bias: float = 0.0, min_gate: float = 0.05):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.min_gate = float(min_gate)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.scale * w + self.bias)
        if self.min_gate > 0:
            g = self.min_gate + (1.0 - self.min_gate) * g
        return g.unsqueeze(-1)  # [B,N,1]
    
class GazeInjectorScenario3(nn.Module):
    """
    Scenario 3 injector:
      input:  h in R^{B,N,D} (heatmap patch embeddings; not necessarily in [0,1])
      output: g in [min_gate, 1] with shape [B,N,1] (scalar gate per patch)
    """
    def __init__(
        self,
        d_model: int,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        min_gate: float = 0.05,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.min_gate = float(min_gate)

        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.proj = nn.Linear(d_model, 1, bias=False)   # h -> scalar per patch

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias  = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,N,D]
        z = self.proj(self.norm(h)).squeeze(-1)         # [B,N]
        g = torch.sigmoid(self.scale * z + self.bias)   # [B,N]
        if self.min_gate > 0:
            g = self.min_gate + (1.0 - self.min_gate) * g
        return g.unsqueeze(-1)                          # [B,N,1]
    
    
@torch.no_grad()
def heatmaps_to_rgb_pils(heatmaps: list[torch.Tensor]) -> list[Image.Image]:
    heatmap_pils: list[Image.Image] = []
    for hm in heatmaps:
        assert hm.ndim == 4 and hm.shape[0] == 1 and hm.shape[1] == 1
        hm_2d = hm[0, 0].detach().float().cpu().numpy()
        hm_2d = hm_2d - hm_2d.min()
        if hm_2d.max() > 0:
            hm_2d = hm_2d / hm_2d.max()
        hm_uint8 = (hm_2d * 255.0).astype(np.uint8)
        heatmap_pils.append(Image.fromarray(hm_uint8, mode="L").convert("RGB"))
    return heatmap_pils


def load_heatmap_npy(path: str, device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Returns heatmap tensor as [1, 1, H, W] with specified dtype.
    """
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, :, :]  # [1,H,W]
    if arr.ndim == 3 and arr.shape[0] != 1:
        # if saved as [H,W,1] etc., try to squeeze to [1,H,W]
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[None, :, :]
    assert arr.ndim == 3 and arr.shape[0] == 1, f"Unexpected heatmap shape: {arr.shape}"

    t = torch.from_numpy(arr).to(dtype=dtype).unsqueeze(0)  # [1,1,H,W]
    return t.to(device)


def heatmap_to_patch_weights(
    heatmap_b1hw: torch.Tensor,
    grid_h: int,
    grid_w: int,
    *,
    clip_min_q: float = 0.05,
    clip_max_q: float = 0.98,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Convert heatmaps [B,1,H,W] to patch weights [B, N] where N = grid_h*grid_w
    using adaptive average pooling.
    Clip the weights by quantiles to reduce the impact of outliers.
    """
    assert heatmap_b1hw.ndim == 4 and heatmap_b1hw.size(1) == 1
    B = heatmap_b1hw.size(0)

    pooled = F.adaptive_avg_pool2d(heatmap_b1hw, (grid_h, grid_w))  # [B,1,gh,gw]
    w = pooled.view(B, -1)  # [B, N]

    # nota per il dario del futuro:
    # clipping e normalization sono applicate heamap-wise (o tile-wise in LLaVA1.6), non patch-wise
    # computing quantiles and clamping
    if clip_min_q > 0 or clip_max_q < 1.0:
        low = torch.quantile(w, clip_min_q, dim=1, keepdim=True)
        high = torch.quantile(w, clip_max_q, dim=1, keepdim=True)
        w = torch.clamp(w, min=low, max=high)

    # Min-max normalization to bring weights to 0..1 range
    w_min = w.min(dim=1, keepdim=True)[0]
    w_max = w.max(dim=1, keepdim=True)[0]
    w_norm = (w - w_min) / (w_max - w_min + eps)
    
    return w_norm
    


def apply_patch_weighting(
    patch_tokens: torch.Tensor,
    weights_bn: torch.Tensor,
) -> torch.Tensor:
    """
    patch_tokens: [B, N, D]
    weights_bn:   [B, N]
    weighting: v' = w * v
    """
    assert patch_tokens.ndim == 3
    assert weights_bn.ndim == 2
    B, N, D = patch_tokens.shape
    assert weights_bn.shape == (B, N)
    
    # Ensure weights match patch_tokens dtype
    weights_bn = weights_bn.to(patch_tokens.dtype)

    w = weights_bn.unsqueeze(-1)  # [B,N,1]
    return  w * patch_tokens  # [B,N,D]


########################################################################################################################
# LLaVA-Next/LLaVa-OV specific AnyRes tiling handling for attention alignment loss

def pack_gaze_probs_like_llava_next(
    gaze_tiles: torch.Tensor,      # [B, P, T]  base + tiles (P may be padded); T = (tile_size/patch_size)^2
    image_sizes: torch.Tensor,     # [B, 2]     (H, W) original sizes, same as inputs["image_sizes"]
    model,                         # LlavaNextForConditionalGeneration or LlavaOneVisionForConditionalGeneration
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Replicates LlavaNextModel.pack_image_features() behavior for scalar weights.
    Output is [B, N_img_tokens] where N_img_tokens matches the expanded <image> placeholders.

    - builds big grid from tiles in row-major order
    - unpads with unpad_image() from HF (exact same cropping)
    - appends newline-per-row (extra column of zeros)
    - flattens row-major
    - prepends base tile tokens
    - renormalizes to sum to 1
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape,
        unpad_image,
    )

    cfg = model.config
    tile_size = cfg.vision_config.image_size
    patch_size = cfg.vision_config.patch_size

    grid_h = tile_size // patch_size
    grid_w = tile_size // patch_size
    T_expected = grid_h * grid_w
    if gaze_tiles.shape[-1] != T_expected:
        raise RuntimeError(
            f"Expected T={T_expected} (tile_size/patch_size squared), got {gaze_tiles.shape[-1]}."
        )

    B = gaze_tiles.shape[0]
    outs = []
    for b in range(B):
        # how many tiles (excluding base) does AnyRes create for this image?
        num_patch_h, num_patch_w = get_anyres_image_grid_shape(
            image_sizes[b], cfg.image_grid_pinpoints, tile_size
        )
        num_tiles = int(num_patch_h * num_patch_w)
        P_needed = 1 + num_tiles

        if gaze_tiles.shape[1] < P_needed:
            raise RuntimeError(
                f"Not enough patches in gaze_tiles: have P={gaze_tiles.shape[1]}, need {P_needed} "
                f"(1 + {num_patch_h}*{num_patch_w})."
            )

        g = gaze_tiles[b, :P_needed]         # [1+num_tiles, T]
        base = g[0]                          # [T]
        tiles = g[1:]                        # [num_tiles, T]

        # reshape tiles to [num_patch_h, num_patch_w, grid_h, grid_w]
        tiles = tiles.view(num_patch_h, num_patch_w, grid_h, grid_w)

        # same spatial reordering as pack_image_features:
        # (tile_r, patch_r, tile_c, patch_c) -> big grid [H_big, W_big]
        big = tiles.permute(0, 2, 1, 3).contiguous().view(num_patch_h * grid_h, num_patch_w * grid_w)

        # unpad EXACTLY like HF (expects [C,H,W])
        big = unpad_image(big.unsqueeze(0), image_sizes[b]).squeeze(0)  # [H_unpad, W_unpad]

        # append newline per row (extra column). weights for newline tokens = 0
        big = torch.cat([big, big.new_zeros(big.shape[0], 1)], dim=1)   # [H_unpad, W_unpad+1]

        packed = torch.cat([base, big.reshape(-1)], dim=0)              # [T + H_unpad*(W_unpad+1)]

        packed = packed.clamp_min(0)
        s = packed.sum()
        if s <= eps:
            packed = torch.full_like(packed, 1.0 / packed.numel())
        else:
            packed = packed / s

        outs.append(packed)

    # NOTE: if B>1 and images have different anyres token counts, lengths will differ.
    # To keep attention_alignment_loss IDENTICAL, batch_size=1 for llava-next,
    lens = [x.numel() for x in outs]
    if len(set(lens)) != 1:
        raise RuntimeError(
            f"LLaVA-Next anyres produced variable image-token lengths in batch: {lens}. "
            f"Use batch_size=1 or bucket by aspect ratio so each batch has equal packed length."
        )

    return torch.stack(outs, dim=0)  # [B, N_img_tokens]
