#!/usr/bin/env python3
"""Learnable Gaze Gating training for LLaVA-OneVision."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from contextlib import nullcontext
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import types

try:
    # Prefer Qwen2 rotary implementation for llava-onevision-qwen2 models.
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as _apply_rotary_pos_emb
except Exception:
    try:
        # Fallback for LLaMA-family checkpoints.
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _apply_rotary_pos_emb
    except Exception:
        _apply_rotary_pos_emb = None

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


# -----------------------
# Imports from repo
# -----------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.common import JsonlGazePromptOnly, set_tokenizer_padding
from training.onevision_prompting import collate_fn
from training.modeling import select_model_and_adapter_classes
from training.attention_utils import _find_longest_run_positions
from training.lgg.common import (
    build_per_sample_gaze_targets,
    distill_kl_loss,
    freeze_all_params,
    preprocess_heatmaps_to_weights,
)
from src.data.heatmaps import apply_patch_weighting, GazeInjector, pack_gaze_probs_like_llava_next
from src.data.prompts import PROMPTS_FT
from src.models.utils import unwrap_to_llava
from src.models.llava_15 import LlavaHFAdapter
from src.models.llava_next import LlavaNextHFAdapter
from src.models.llava_ov import LlavaOnevisionHFAdapter


# -----------------------
# Attention alignment
# -----------------------



def infer_image_token_positions_per_sample(
    input_ids_b: torch.Tensor,
    attn_mask_b: torch.Tensor,
    attn_seq_len: int,
    image_token_id: int,
    num_image_tokens: int,
) -> Tuple[List[int], int, int]:
    """Infer positions of image tokens in the *attention* sequence.

    Returns (img_positions, q_idx, valid_len_out)
      - img_positions: positions of image tokens (len = num_image_tokens)
      - q_idx: last valid token index (query)
      - valid_len_out: effective valid length in attention sequence (for padding mask)
    """
    valid_len_in = int(attn_mask_b.sum().item())
    input_ids_valid = input_ids_b[:valid_len_in]

    img_positions_in = (input_ids_valid == image_token_id).nonzero(as_tuple=False).view(-1).tolist()

    # A) Already expanded: exactly num_image_tokens placeholders are present in input_ids.
    if len(img_positions_in) == num_image_tokens:
        valid_len_out = min(valid_len_in, attn_seq_len)
        q_idx = valid_len_out - 1
        return img_positions_in, q_idx, valid_len_out

    if len(img_positions_in) == 0:
        raise RuntimeError(
            "Could not infer image token positions: no image tokens/placeholders found in valid input_ids. "
            f"valid_len={valid_len_in}, image_token_id={image_token_id}."
        )

    run = _find_longest_run_positions(input_ids_valid == image_token_id)
    if len(run) != len(img_positions_in):
        raise RuntimeError(
            "Could not infer image token positions with non-contiguous image placeholders. "
            f"Found {len(img_positions_in)} placeholders but longest contiguous run is {len(run)}. "
            "Expected a single contiguous image-placeholder span per sample."
        )

    placeholder_count = len(run)
    if placeholder_count > num_image_tokens:
        raise RuntimeError(
            "Could not infer image token positions: placeholder count exceeds expected image-token count. "
            f"placeholders={placeholder_count}, expected_tokens={num_image_tokens}."
        )

    placeholder_pos0 = run[0]
    valid_len_out = valid_len_in - placeholder_count + num_image_tokens
    q_idx = valid_len_out - 1
    img_positions = list(range(placeholder_pos0, placeholder_pos0 + num_image_tokens))

    if valid_len_out > attn_seq_len:
        raise RuntimeError(
            f"Expanded valid len {valid_len_out} exceeds attention seq len {attn_seq_len}. "
            "Model/processor tokenization differs from assumptions."
        )

    return img_positions, q_idx, valid_len_out


def attention_alignment_loss(
    attentions: Tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gaze_probs: torch.Tensor,
    image_token_id: int,
    num_last_layers: int = 1,
    eps: float = 1e-8,
    loss_type: str = "kl",
) -> torch.Tensor:
    """Compute a distribution-matching loss between LLM attention on image tokens and gaze.

    attentions: tuple of length L, each [B, H, S, S]
    gaze_probs: [B, N] (already normalized to sum=1)

    We use attention from the last valid token (q_idx) to image token positions.
    We average over heads and over the last `num_last_layers` layers.
    """
    if len(attentions) == 0:
        raise RuntimeError("Model did not return attentions. Ensure output_attentions=True.")

    B = input_ids.shape[0]
    N = gaze_probs.shape[1]

    use_layers = attentions[-num_last_layers:]

    # Accumulate per-layer attention distributions over image tokens.
    attn_img_sum = None

    for attn in use_layers:
        # [B,H,S,S]
        S = attn.shape[-1]
        per_b: List[torch.Tensor] = []
        for b in range(B):
            img_pos, q_idx, _valid_len_out = infer_image_token_positions_per_sample(
                input_ids[b], attention_mask[b], S, image_token_id, N
            )
            # Clamp q_idx in case of padding differences.
            q_idx = min(q_idx, S - 1)

            # attention from query token -> all keys
            # [H, S]
            a_q = attn[b, :, q_idx, :]
            # mean over heads -> [S]
            a_q = a_q.mean(dim=0)
            # select image token positions -> [N]
            a_img = a_q[torch.tensor(img_pos, device=a_q.device, dtype=torch.long)]
            # normalize to probability
            a_img = a_img.clamp_min(0)
            a_img = a_img / (a_img.sum() + eps)
            per_b.append(a_img)

        attn_img = torch.stack(per_b, dim=0)  # [B,N]
        attn_img_sum = attn_img if attn_img_sum is None else (attn_img_sum + attn_img)

    attn_probs = attn_img_sum / float(len(use_layers))

    if loss_type == "mse":
        return F.mse_loss(attn_probs, gaze_probs)

    if loss_type == "kl":
        # KL(gaze || attn) or KL(attn || gaze)? We want attn to match gaze.
        # Use KL(gaze || attn) to penalize missing mass on gaze hotspots.
        log_attn = (attn_probs + eps).log()
        return F.kl_div(log_attn, gaze_probs, reduction="batchmean")

    if loss_type == "ce":
        # Cross-entropy with gaze as soft labels: -sum g * log(attn)
        log_attn = (attn_probs + eps).log()
        return -(gaze_probs * log_attn).sum(dim=1).mean()

    raise ValueError(f"Unknown loss_type='{loss_type}'. Choose from: kl, mse, ce")


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # hidden_states: [B, kv_heads, S, d]
    if n_rep == 1:
        return hidden_states
    b, kvh, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, kvh, n_rep, s, d)
    return hidden_states.reshape(b, kvh * n_rep, s, d)


def _resolve_attn_layout(attn_mod: nn.Module) -> Tuple[int, int, int, int]:
    """Resolve attention layout across transformers variants.

    Returns (num_heads, num_kv_heads, head_dim, num_kv_groups).
    """
    num_heads = getattr(attn_mod, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn_mod, "num_attention_heads", None)
    if num_heads is None and hasattr(attn_mod, "config"):
        num_heads = getattr(attn_mod.config, "num_attention_heads", None)

    head_dim = getattr(attn_mod, "head_dim", None)

    if num_heads is None and head_dim is not None:
        num_heads = attn_mod.q_proj.out_features // head_dim
    if head_dim is None and num_heads is not None:
        head_dim = attn_mod.q_proj.out_features // num_heads

    if num_heads is None or head_dim is None:
        raise RuntimeError(
            "Could not resolve attention layout (num_heads/head_dim) from attention module. "
            "Check transformers version compatibility."
        )

    num_kv_heads = getattr(attn_mod, "num_key_value_heads", None)
    if num_kv_heads is None and hasattr(attn_mod, "config"):
        num_kv_heads = getattr(attn_mod.config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = attn_mod.k_proj.out_features // head_dim

    num_kv_groups = getattr(attn_mod, "num_key_value_groups", None)
    if num_kv_groups is None:
        num_kv_groups = max(1, num_heads // max(1, num_kv_heads))

    return int(num_heads), int(num_kv_heads), int(head_dim), int(num_kv_groups)


def _resolve_rope_cos_sin(
    attn_mod: nn.Module,
    k_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    kv_seq_len: int,
    kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    position_embeddings = kwargs.get("position_embeddings", None)
    if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2:
        return position_embeddings[0], position_embeddings[1]

    rotary_emb = getattr(attn_mod, "rotary_emb", None)
    if rotary_emb is None:
        raise RuntimeError(
            "Could not resolve RoPE embeddings: missing both kwargs['position_embeddings'] and self.rotary_emb."
        )

    try:
        return rotary_emb(k_states, position_ids=position_ids)
    except TypeError:
        try:
            return rotary_emb(k_states, seq_len=kv_seq_len)
        except TypeError:
            return rotary_emb(k_states, position_ids)


def _apply_rope_qk(
    attn_mod: nn.Module,
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_fn = getattr(attn_mod, "rotary_fn", None)
    if rotary_fn is not None:
        try:
            return rotary_fn(q_states, k_states, cos, sin)
        except TypeError:
            pass

    if _apply_rotary_pos_emb is None:
        raise RuntimeError(
            "Could not import any supported apply_rotary_pos_emb and self.rotary_fn is unavailable; cannot apply RoPE."
        )

    return _apply_rotary_pos_emb(q_states, k_states, cos, sin)


def _get_llm_layers(llava_core: nn.Module):
    # LlavaNextForConditionalGeneration has .language_model (LlamaForCausalLM)
    lm = getattr(llava_core, "language_model", None) or getattr(llava_core.model, "language_model", None)
    if lm is None:
        print("NONEEEEE")
        lm = llava_core
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    raise RuntimeError("Could not locate LLM layers (expected lm.model.layers).")


def attention_alignment_loss_from_captured(
    attn_img_layers: List[torch.Tensor],  # list of [B,N]
    gaze_probs: torch.Tensor,             # [B,N]
    loss_type: str = "kl",
    eps: float = 1e-8,
) -> torch.Tensor:
    if not attn_img_layers:
        raise RuntimeError("No captured attention slices found (attn_img_layers is empty).")
    attn_probs = torch.stack(attn_img_layers, dim=0).mean(dim=0)  # [B,N]

    if loss_type == "mse":
        return F.mse_loss(attn_probs, gaze_probs)

    log_attn = (attn_probs + eps).log()
    if loss_type == "kl":
        return F.kl_div(log_attn, gaze_probs, reduction="batchmean")
    if loss_type == "ce":
        return -(gaze_probs * log_attn).sum(dim=1).mean()

    raise ValueError(f"Unknown loss_type='{loss_type}'. Choose from: kl, mse, ce")


def install_lastk_attn_slice_capture(
    model: nn.Module,
    last_k: int,
    capture_state: Dict[str, Any],
    llava_core: nn.Module,
) -> List[Any]:
    """Monkeypatch last K self-attn layers to capture only query->image attention slice.

    capture_state must contain (set per batch before forward):
      - enabled: bool
      - input_ids: [B,S_in]
      - attention_mask: [B,S_in]
      - gaze_N: int
      - image_token_id: int
      - attn_img_layers: list (will be appended with [B,N] per wrapped layer)
    """

    layers = _get_llm_layers(llava_core)
    n_layers = len(layers)
    if last_k < 1 or last_k > n_layers:
        raise ValueError(f"last_k must be in [1,{n_layers}], got {last_k}")

    keep = list(range(n_layers - last_k, n_layers))
    handles = []

    for li in keep:
        attn_mod = layers[li].self_attn
        orig_forward = attn_mod.forward

        def wrapped_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            # Always run the real forward with output_attentions=False to avoid [S,S] materialization.
            # Capture only if enabled for this batch.
            if capture_state.get("enabled", False):
                input_ids = capture_state["input_ids"]
                attn_mask_1d = capture_state["attention_mask"]
                N = int(capture_state["gaze_N"])
                image_token_id = int(capture_state["image_token_id"])
                # debug_nan = bool(capture_state.get("debug_nan", False))

                bsz, seqlen, _ = hidden_states.shape
                # if debug_nan and (not torch.isfinite(hidden_states).all()):
                #     hs = hidden_states.detach()
                #     raise RuntimeError(
                #         "[NaN-DEBUG][attn_capture] non-finite hidden_states before q/k projections "
                #         f"(layer={li}, seqlen={seqlen}, dtype={hs.dtype}, "
                #         f"min={float(hs.nan_to_num().min().cpu()):.6g}, max={float(hs.nan_to_num().max().cpu()):.6g})."
                #     )

                # Projections (match HF LlamaAttention shapes)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                # if debug_nan and ((not torch.isfinite(q).all()) or (not torch.isfinite(k).all())):
                #     qd = q.detach()
                #     kd = k.detach()
                #     raise RuntimeError(
                #         "[NaN-DEBUG][attn_capture] non-finite q/k projections "
                #         f"(layer={li}, q_dtype={qd.dtype}, k_dtype={kd.dtype}, "
                #         f"q_min={float(qd.nan_to_num().min().cpu()):.6g}, q_max={float(qd.nan_to_num().max().cpu()):.6g}, "
                #         f"k_min={float(kd.nan_to_num().min().cpu()):.6g}, k_max={float(kd.nan_to_num().max().cpu()):.6g})."
                #     )

                num_heads, num_kv_heads, head_dim, num_kv_groups = _resolve_attn_layout(self)

                q = q.reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2)  # [B,H,S,d]
                k = k.reshape(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)  # [B,kvH,S,d]

                # RoPE
                # HF variants differ in rotary_emb signature; try both.
                kv_seq_len = seqlen
                cos, sin = _resolve_rope_cos_sin(
                    attn_mod=self,
                    k_states=k,
                    position_ids=position_ids,
                    kv_seq_len=kv_seq_len,
                    kwargs=kwargs,
                )
                q, k = _apply_rope_qk(
                    attn_mod=self,
                    q_states=q,
                    k_states=k,
                    cos=cos,
                    sin=sin,
                    position_ids=position_ids,
                )
                # if debug_nan and ((not torch.isfinite(q).all()) or (not torch.isfinite(k).all())):
                #     qd = q.detach()
                #     kd = k.detach()
                #     raise RuntimeError(
                #         "[NaN-DEBUG][attn_capture] non-finite q/k after RoPE "
                #         f"(layer={li}, q_dtype={qd.dtype}, k_dtype={kd.dtype}, "
                #         f"q_min={float(qd.nan_to_num().min().cpu()):.6g}, q_max={float(qd.nan_to_num().max().cpu()):.6g}, "
                #         f"k_min={float(kd.nan_to_num().min().cpu()):.6g}, k_max={float(kd.nan_to_num().max().cpu()):.6g})."
                #     )

                # expand kv heads if needed
                if num_kv_groups != 1:
                    k = _repeat_kv(k, num_kv_groups)  # -> [B,H,S,d]

                # Build per-sample attention slice: last valid token -> image positions
                per_b = []
                scale = 1.0 / math.sqrt(head_dim)

                for b in range(bsz):
                    img_pos, q_idx, valid_len_out = infer_image_token_positions_per_sample(
                        input_ids[b], attn_mask_1d[b], seqlen, image_token_id, N
                    )
                    q_idx = min(q_idx, seqlen - 1)
                    valid_len_out = min(valid_len_out, seqlen)

                    # logits: [H,S]
                    q_b = q[b, :, q_idx, :]                    # [H,d]
                    k_b = k[b]                                  # [H,S,d]
                    with torch.autocast(device_type="cuda", enabled=False) if q_b.is_cuda else nullcontext():
                        q_b_f = q_b.float()
                        k_b_f = k_b.float()
                        logits = torch.einsum("hd,hsd->hs", q_b_f, k_b_f) * scale  # [H,S] fp32
                    # if debug_nan and (not torch.isfinite(logits).all()):
                    #     q_b_finite = bool(torch.isfinite(q_b).all())
                    #     k_b_finite = bool(torch.isfinite(k_b).all())
                    #     q_absmax = float(q_b.detach().abs().amax().cpu())
                    #     k_absmax = float(k_b.detach().abs().amax().cpu())
                    #     autocast_off_ctx = (
                    #         torch.autocast(device_type="cuda", enabled=False)
                    #         if q_b.is_cuda
                    #         else nullcontext()
                    #     )
                    #     with autocast_off_ctx:
                    #         logits_fp32 = torch.einsum("hd,hsd->hs", q_b.float(), k_b.float()) * scale
                    #     logits_fp32_finite = bool(torch.isfinite(logits_fp32).all())
                    #     logits_fp32_absmax = float(logits_fp32.detach().abs().amax().cpu())
                    #     logits_fp32_dtype = str(logits_fp32.dtype)
                    #     autocast_enabled = bool(torch.is_autocast_enabled())
                    #     autocast_cuda_dtype = str(torch.get_autocast_gpu_dtype()) if torch.cuda.is_available() else "cpu"
                    #     raise RuntimeError(
                    #         f"[NaN-DEBUG][attn_capture] non-finite logits before mask "
                    #         f"(layer={li}, sample={b}, q_idx={q_idx}, valid_len_out={valid_len_out}, seqlen={seqlen}, "
                    #         f"q_finite={q_b_finite}, k_finite={k_b_finite}, q_absmax={q_absmax:.6g}, k_absmax={k_absmax:.6g}, "
                    #         f"logits_dtype={logits.dtype}, logits_fp32_finite={logits_fp32_finite}, "
                    #         f"logits_fp32_absmax={logits_fp32_absmax:.6g}, logits_fp32_dtype={logits_fp32_dtype}, "
                    #         f"autocast_enabled={autocast_enabled}, autocast_cuda_dtype={autocast_cuda_dtype})."
                    #     )

                    # mask padding keys beyond valid_len_out
                    if valid_len_out < seqlen:
                        logits[:, valid_len_out:] = float("-inf")
                    # causal (optional safety)
                    if q_idx + 1 < seqlen:
                        logits[:, q_idx + 1 :] = float("-inf")

                    probs = torch.softmax(logits, dim=-1)  # [H,S] fp32
                    # if capture_state.get("debug_nan", False) and (not torch.isfinite(probs).all()):
                    #     raise RuntimeError(
                    #         f"[NaN-DEBUG][attn_capture] non-finite probs after softmax "
                    #         f"(layer={li}, sample={b}, q_idx={q_idx}, valid_len_out={valid_len_out}, seqlen={seqlen})."
                    #     )
                    a_q = probs.mean(dim=0)  # [S]

                    img_idx = torch.tensor(img_pos, device=a_q.device, dtype=torch.long)
                    a_img = a_q[img_idx]  # [N]
                    a_img = a_img.clamp_min(0)
                    a_img = a_img / (a_img.sum() + 1e-8)
                    # if capture_state.get("debug_nan", False) and (not torch.isfinite(a_img).all()):
                    #     raise RuntimeError(
                    #         f"[NaN-DEBUG][attn_capture] non-finite normalized image attention "
                    #         f"(layer={li}, sample={b}, N={N})."
                    #     )
                    per_b.append(a_img)

                attn_img = torch.stack(per_b, dim=0)  # [B,N]
                capture_state["attn_img_layers"].append(attn_img)

            return orig_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
                **kwargs,
            )

        attn_mod.forward = types.MethodType(wrapped_forward, attn_mod)
        handles.append((attn_mod, orig_forward))

    return handles






# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--train_jsonl", type=str, required=True, help="Comma-separated JSONL paths")
    ap.add_argument("--val_jsonl", type=str, default=None, help="Comma-separated JSONL paths (optional)")
    ap.add_argument("--output_dir_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)

    # Attention alignment
    ap.add_argument("--loss", type=str, default="kl", choices=["kl", "mse", "ce"])
    ap.add_argument("--attn_last_layers", type=int, default=1, help="Use the last K LLM layers' attentions")
    
    # Mix objectives (mitigate nonsense outputs)
    ap.add_argument("--lambda_attn", type=float, default=0.05, help="Weight for attention-alignment loss.")
    ap.add_argument("--lambda_distill", type=float, default=1.0, help="Weight for logit distillation (stability).")
    ap.add_argument("--lambda_gate", type=float, default=0.05, help="Weight for keeping gates near identity.")
    ap.add_argument("--distill_temp", type=float, default=2.0, help="Temperature for distillation KL.")
    ap.add_argument("--attn_ramp_updates", type=int, default=200)
    ap.add_argument("--gaze_label_smoothing", type=float, default=0.05)

    # Regularization / mixing
    ap.add_argument(
        "--p_no_gaze",
        type=float,
        default=0.0,
        help=(
            "Probability to disable gaze injection and use a uniform gaze target for that batch. "
            "This can reduce overfitting to gaze hotspots and keeps attention from collapsing."
        ),
    )

    # Learnable injector
    ap.add_argument("--injector_init_scale", type=float, default=1.0)
    ap.add_argument("--injector_init_bias", type=float, default=0.0)
    ap.add_argument("--injector_min_gate", type=float, default=0.05)

    # Optional: LoRA on projector (trained with attention loss)
    ap.add_argument("--train_projector_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--init_lora_dir",
        type=str,
        default=None,
        help="Path to an existing PEFT LoRA adapter dir to initialize from.",
    )

    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)
    # ap.add_argument("--debug_nan", action="store_true", help="Print tensor stats and stop at first NaN/Inf.")
    # ap.add_argument("--debug_nan_max_prints", type=int, default=20, help="Limit number of debug stat prints.")

    args = ap.parse_args()

    # _debug_state = {"prints": 0}

    # def _dbg_stats(name: str, tensor: torch.Tensor) -> None:
    #     if not args.debug_nan:
    #         return
    #     if _debug_state["prints"] >= max(0, int(args.debug_nan_max_prints)):
    #         return
    #     with torch.no_grad():
    #         x = tensor.detach()
    #         finite = torch.isfinite(x)
    #         total = int(x.numel())
    #         n_finite = int(finite.sum().item())
    #         n_bad = total - n_finite
    #         if n_finite > 0:
    #             x_ok = x[finite]
    #             mn = float(x_ok.min().cpu())
    #             mx = float(x_ok.max().cpu())
    #             mean = float(x_ok.mean().cpu())
    #         else:
    #             mn = float("nan")
    #             mx = float("nan")
    #             mean = float("nan")
    #         print(
    #             f"[NaN-DEBUG] {name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
    #             f"finite={n_finite}/{total} bad={n_bad} min={mn:.6g} max={mx:.6g} mean={mean:.6g}"
    #         )
    #         _debug_state["prints"] += 1

    # def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    #     if torch.isfinite(tensor).all():
    #         return
    #     _dbg_stats(name, tensor)
    #     raise RuntimeError(f"[NaN-DEBUG] non-finite detected in '{name}'.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    base_dir = Path(__file__).resolve().parent / args.model.split("/")[-1]
    out_dir = base_dir / args.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Processor
    processor = AutoProcessor.from_pretrained(args.model)
    set_tokenizer_padding(processor)

    # Base model / adapter class
    model_cls, adapter_cls = select_model_and_adapter_classes(args.model)

    # Base model
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": amp_dtype,
        "device_map": None,
        "attn_implementation": "sdpa",
    }
    if model_cls is LlavaOnevisionForConditionalGeneration:
        model_kwargs["trust_remote_code"] = True

    base = model_cls.from_pretrained(
        args.model,
        **model_kwargs,
    ).to(device)

    # Ensure we can get attentions
    base.config.use_cache = False

    if args.grad_ckpt and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    # Apply (optional) projector LoRA
    if args.train_projector_lora:
        if args.init_lora_dir:
            kwargs = {}
            sig = inspect.signature(PeftModel.from_pretrained)
            if "is_trainable" in sig.parameters:
                kwargs["is_trainable"] = True
            model = PeftModel.from_pretrained(base, args.init_lora_dir, **kwargs)
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=[
                    "multi_modal_projector.linear_1",
                    "multi_modal_projector.linear_2",
                ],
            )
            model = get_peft_model(base, lora_cfg)
    else:
        model = base

    # Freeze everything; re-enable gradients only for injector + (optional) LoRA params.
    freeze_all_params(model)

    # Trainable injector
    injector = GazeInjector(
        init_scale=args.injector_init_scale,
        init_bias=args.injector_init_bias,
        min_gate=args.injector_min_gate,
    ).to(device)

    # Ensure LoRA params are trainable if enabled
    if args.train_projector_lora:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

    # Collect trainable params
    trainable_params = list(injector.parameters())
    # trainable_params = []
    if args.train_projector_lora:
        trainable_params += [p for p in model.parameters() if p.requires_grad]

    # Deduplicate
    seen = set()
    unique_params = []
    for p in trainable_params:
        if id(p) not in seen:
            unique_params.append(p)
            seen.add(id(p))
    trainable_params = unique_params

    if not trainable_params:
        raise RuntimeError("No trainable parameters (injector or LoRA) found.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Datasets
    train_paths = [p.strip() for p in args.train_jsonl.split(",") if p.strip()]
    train_dss = [JsonlGazePromptOnly(p) for p in train_paths]
    train_ds = train_dss[0] if len(train_dss) == 1 else ConcatDataset(train_dss)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, processor, args.max_length),
    )

    val_dl = None
    if args.val_jsonl:
        val_paths = [p.strip() for p in args.val_jsonl.split(",") if p.strip()]
        val_dss = [JsonlGazePromptOnly(p) for p in val_paths]
        val_ds = val_dss[0] if len(val_dss) == 1 else ConcatDataset(val_dss)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, processor, args.max_length),
        )

    # Scheduler
    updates_per_epoch = math.ceil(len(train_dl) / max(1, args.grad_accum))
    total_updates = updates_per_epoch * args.epochs
    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)

    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_updates, total_updates)

    # Vision / token ids
    core_llava = unwrap_to_llava(model)
    patch_size = getattr(core_llava.config.vision_config, "patch_size", 14)
    vision_layer = getattr(core_llava.config, "vision_feature_layer", None)

    image_token_id = getattr(core_llava.config, "image_token_index", None)
    if image_token_id is None:
        # Common fallback
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            image_token_id = None
    if image_token_id is None or image_token_id < 0:
        raise RuntimeError(
            "Could not resolve image token id (config.image_token_index or '<image>'). "
            "Cannot align attention to image tokens."
        )

    # -----------------------
    # Attention slice capture state + install on last K layers
    # -----------------------
    attn_capture_state: Dict[str, Any] = {
        "enabled": False,
        "input_ids": None,
        "attention_mask": None,
        "gaze_N": None,
        "image_token_id": int(image_token_id),
        "attn_img_layers": [],
    }

    llava_core = unwrap_to_llava(model)
    _attn_patch_handles = install_lastk_attn_slice_capture(
        model=model,
        last_k=int(args.attn_last_layers),
        capture_state=attn_capture_state,
        llava_core=llava_core,
    )
    # attn_capture_state["debug_nan"] = bool(args.debug_nan)

    # Hook state
    gaze_state: Dict[str, Optional[torch.Tensor]] = {"weights": None, "gates": None}

    def _vision_hook(_module, _inp, out):
        w = gaze_state["weights"]
        if w is None:
            return out
        # if args.debug_nan:
        #     _assert_finite("vision_hook.weights", w)

        # Pick which vision features are used later.
        if hasattr(out, "hidden_states") and vision_layer is not None and out.hidden_states is not None:
            feats = out.hidden_states[vision_layer]
            hs = list(out.hidden_states)
        else:
            feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            hs = None

        
        patches = feats
        # if args.debug_nan:
        #     _assert_finite("vision_hook.patches_in", patches)

        if patches.shape[1] != w.shape[1]:
            raise RuntimeError(f"Patch count mismatch: {patches.shape[1]} vs weights {w.shape[1]}")

        g = injector(w).to(patches.dtype)  # [B,N]
        # if args.debug_nan:
        #     _assert_finite("vision_hook.gates", g)
        gaze_state["gates"] = g
        new_feats = patches * g
        # if args.debug_nan:
        #     _assert_finite("vision_hook.patches_out", new_feats)

        if hs is not None:
            hs[vision_layer] = new_feats
            out.hidden_states = tuple(hs)
        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = new_feats
        return out

    handle = core_llava.model.vision_tower.register_forward_hook(_vision_hook)

    # Logging
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as flog:
        flog.write(
            json.dumps(
                {
                    "model": args.model,
                    "output_dir": str(out_dir),
                    "objective": "attention_alignment",
                    "loss": args.loss,
                    "attn_last_layers": args.attn_last_layers,
                    "train_projector_lora": bool(args.train_projector_lora),
                    "injector": {
                        "type": "affine_sigmoid",
                        "init_scale": args.injector_init_scale,
                        "init_bias": args.injector_init_bias,
                        "min_gate": args.injector_min_gate,
                    },
                    "lora": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "targets": ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"],
                    },
                    "dtype": args.dtype,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "p_no_gaze": args.p_no_gaze,
                }
            )
            + "\n"
        )

    # Train
    model.train()
    injector.train()

    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    accum = 0
    running_loss_total = 0.0
    running_loss_attn = 0.0
    running_loss_distill = 0.0
    running_loss_gate = 0.0
    
    def _lambda_attn_now(update_idx: int) -> float:
        if args.attn_ramp_updates <= 0:
            return float(args.lambda_attn)
        t = min(1.0, float(update_idx) / float(args.attn_ramp_updates))
        return float(args.lambda_attn) * t

    def _forward_batch(b: Batch, update_idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute gaze patch weights
        w = preprocess_heatmaps_to_weights(adapter_cls, processor, patch_size, b.heatmaps, device)
        # if args.debug_nan:
        #     _assert_finite("forward.patch_weights", w)
        bsz = b.inputs["input_ids"].shape[0]

        gaze_state["weights"] = w
        # Check if AnyRes was applied, if so we need to unpad for the attention loss
        if w.shape[0] != bsz:
            gaze_tiles = w.view(bsz, -1, w.shape[-1])
            gaze_target = pack_gaze_probs_like_llava_next(
                gaze_tiles=gaze_tiles, 
                image_sizes=b.inputs["image_sizes"],
                model=model,
            )
        else:
            gaze_target = build_per_sample_gaze_targets(w, batch_size=bsz)
        # if args.debug_nan:
        #     _assert_finite("forward.gaze_target_pre_smooth", gaze_target)
            
        # Gaze label smoothing to avoid peaky collapse
        if args.gaze_label_smoothing > 0:
            beta = float(args.gaze_label_smoothing)
            gaze_target = (1.0 - beta) * gaze_target + beta * (1.0 / gaze_target.shape[1])
            gaze_target = gaze_target / (gaze_target.sum(dim=1, keepdim=True) + 1e-8)
        # if args.debug_nan:
        #     _assert_finite("forward.gaze_target", gaze_target)

        inputs = {k: v.to(device) for k, v in b.inputs.items()}
        attn_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
        # if args.debug_nan:
        #     _assert_finite("forward.input_ids", inputs["input_ids"].float())
        #     _assert_finite("forward.attention_mask", attn_mask.float())

        if update_idx < 5:
            with torch.no_grad():
                B = inputs["input_ids"].shape[0]
                for b in range(B):
                    valid_len = int(attn_mask[b].sum().item())
                    n_placeholders = int((inputs["input_ids"][b, :valid_len] == int(image_token_id)).sum().item())
                    if n_placeholders != gaze_target.shape[1]:
                        raise RuntimeError(
                            f"Mismatch (b={b}): <image> placeholders={n_placeholders} "
                            f"but gaze_target_len={gaze_target.shape[1]}."
                        )

        # Ask for attentions
        # --- enable capture for this batch
        attn_capture_state["enabled"] = True
        attn_capture_state["input_ids"] = inputs["input_ids"]
        attn_capture_state["attention_mask"] = attn_mask
        attn_capture_state["gaze_N"] = int(gaze_target.shape[1])
        attn_capture_state["attn_img_layers"].clear()

        # forward WITHOUT output_attentions to avoid [S,S]
        out_s = model(**inputs, return_dict=True)
        logits_s = out_s.logits.float()
        # if args.debug_nan:
        #     _assert_finite("forward.logits_student", logits_s)
        del out_s

        # disable capture immediately after forward
        attn_capture_state["enabled"] = False

        # alignment loss from captured [B,N] slices (one per wrapped layer)
        loss_attn = attention_alignment_loss_from_captured(
            attn_img_layers=attn_capture_state["attn_img_layers"],
            gaze_probs=gaze_target,
            loss_type=args.loss,
        )
        # if args.debug_nan:
        #     _assert_finite("forward.loss_attn", loss_attn)

        # clear list reference (doesn't break autograd; loss keeps graph)
        attn_capture_state["attn_img_layers"].clear()


        # Logit distillation (stability): teacher = same model w/ adapters disabled (if PEFT) and NO gaze injection
        loss_distill = torch.tensor(0.0, device=device)
        if args.lambda_distill > 0:
            prev = gaze_state["weights"]
            gaze_state["weights"] = None
            was_training = model.training
            model.eval()
            ctx = model.disable_adapter() if isinstance(model, PeftModel) and hasattr(model, "disable_adapter") else nullcontext()
            with torch.no_grad():
                with ctx:
                    out_t = model(**inputs, output_attentions=False, return_dict=True)
                    logits_t = out_t.logits.float()
                    # if args.debug_nan:
                    #     _assert_finite("forward.logits_teacher", logits_t)
                    del out_t
            if was_training:
                model.train()
            gaze_state["weights"] = prev
            loss_distill = distill_kl_loss(
                logits_s=logits_s,
                logits_t=logits_t,
                attention_mask=attn_mask,
                temperature=args.distill_temp,
            )
            # if args.debug_nan:
            #     _assert_finite("forward.loss_distill", loss_distill)

        # Gate regularization (keep near identity)
        g_actual = gaze_state["gates"]
        if g_actual is not None:
            loss_gate = ((g_actual - 1.0) ** 2).mean()
        else:
            loss_gate = torch.tensor(0.0, device=device)
        # if args.debug_nan:
        #     _assert_finite("forward.loss_gate", loss_gate)
        gaze_state["gates"] = None

        lam_attn = _lambda_attn_now(update_idx)
        loss_total = lam_attn * loss_attn + float(args.lambda_distill) * loss_distill + float(args.lambda_gate) * loss_gate
        # if args.debug_nan:
        #     _assert_finite("forward.loss_total", loss_total)
        metrics = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_attn": float(loss_attn.detach().cpu()),
            "loss_distill": float(loss_distill.detach().cpu()),
            "loss_gate": float(loss_gate.detach().cpu()),
            "lambda_attn": float(lam_attn),
            "lambda_distill": float(args.lambda_distill),
            "lambda_gate": float(args.lambda_gate),
        }
        return loss_total, metrics
    

    @torch.no_grad()
    def _eval(dl: DataLoader) -> float:
        model.eval()
        injector.eval()
        tot = 0.0
        n = 0
        for b in dl:
            loss, _ = _forward_batch(b, update_idx=global_update)
            tot += float(loss.detach().cpu())
            n += 1
        model.train()
        injector.train()
        return tot / max(n, 1)

    pbar = tqdm(total=total_updates, desc="updates")
    try:
        for epoch in range(args.epochs):
            for b in train_dl:
                if global_update >= total_updates:
                    break

                with torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else torch.enable_grad():
                    loss, loss_metrics = _forward_batch(b, update_idx=global_update)
                    loss = loss / args.grad_accum
                # if args.debug_nan:
                #     _assert_finite("train.loss_scaled", loss)

                loss.backward()
                # if args.debug_nan:
                #     for n, p in trainable_named_params:
                #         if p.grad is None:
                #             continue
                #         if not torch.isfinite(p.grad).all():
                #             _dbg_stats(f"grad.{n}", p.grad)
                #             _dbg_stats(f"param.{n}", p.data)
                #             raise RuntimeError(f"[NaN-DEBUG] non-finite gradient in '{n}' at update={global_update}.")
                running_loss_total += loss_metrics["loss_total"]
                running_loss_attn += loss_metrics["loss_attn"]
                running_loss_distill += loss_metrics["loss_distill"]
                running_loss_gate += loss_metrics["loss_gate"]
                accum += 1

                if accum >= args.grad_accum:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                    optimizer.step()
                    # if args.debug_nan:
                    #     for n, p in trainable_named_params:
                    #         if not torch.isfinite(p).all():
                    #             _dbg_stats(f"param_after_step.{n}", p.data)
                    #             raise RuntimeError(f"[NaN-DEBUG] non-finite parameter after optimizer.step() in '{n}'.")
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_update += 1
                    pbar.update(1)

                    if global_update % args.log_every == 0:
                        denom = max(accum, 1)
                        with log_path.open("a", encoding="utf-8") as flog:
                            flog.write(
                                json.dumps(
                                    {
                                        "update": global_update,
                                        "epoch": epoch,
                                        "loss_total": running_loss_total / denom,
                                        "loss_attn": running_loss_attn / denom,
                                        "loss_distill": running_loss_distill / denom,
                                        "loss_gate": running_loss_gate / denom,
                                        "lambda_attn": loss_metrics["lambda_attn"],
                                        "lambda_distill": loss_metrics["lambda_distill"],
                                        "lambda_gate": loss_metrics["lambda_gate"],
                                        "lr": float(scheduler.get_last_lr()[0]),
                                        "injector_scale": float(injector.scale.detach().cpu()),
                                        "injector_bias": float(injector.bias.detach().cpu()),
                                    }
                                )
                                + "\n"
                            )

                    running_loss_total = 0.0
                    running_loss_attn = 0.0
                    running_loss_distill = 0.0
                    running_loss_gate = 0.0
                    accum = 0

                    # if val_dl is not None and (global_update % args.eval_every == 0):
                    #     val_loss = _eval(val_dl)
                    #     with log_path.open("a", encoding="utf-8") as flog:
                    #         flog.write(json.dumps({"update": global_update, "val_loss": float(val_loss)}) + "\n")

                    # if global_update % args.save_every == 0:
                    #     ckpt_dir = out_dir / f"step_{global_update:07d}"
                    #     ckpt_dir.mkdir(parents=True, exist_ok=True)
                    #     # Save injector
                    #     torch.save(injector.state_dict(), ckpt_dir / "gaze_injector.pt")
                    #     with (ckpt_dir / "components.json").open("w", encoding="utf-8") as f:
                    #         json.dump(
                    #             {
                    #                 "injector": {"type": "affine_sigmoid", "min_gate": args.injector_min_gate},
                    #                 "image_token_id": int(image_token_id),
                    #                 "vision_feature_layer": vision_layer,
                    #             },
                    #             f,
                    #             indent=2,
                    #         )
                    #     # Save LoRA adapter if any
                    #     if args.train_projector_lora:
                    #         model.save_pretrained(str(ckpt_dir / "projector_lora"))

        if val_dl is not None:
            final_val_loss = _eval(val_dl)
            with log_path.open("a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "update": global_update,
                            "val_loss": float(final_val_loss),
                            "event": "final_eval",
                        }
                    )
                    + "\n"
                )

        # Final save
        torch.save(injector.state_dict(), out_dir / "gaze_injector.pt")
        with (out_dir / "components.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "injector": {"type": "affine_sigmoid", "min_gate": args.injector_min_gate},
                    "image_token_id": int(image_token_id),
                    "vision_feature_layer": vision_layer,
                },
                f,
                indent=2,
            )
        if args.train_projector_lora:
            model.save_pretrained(str(out_dir / "projector_lora"))

    finally:
        handle.remove()
        for attn_mod, orig_fwd in _attn_patch_handles:
            attn_mod.forward = orig_fwd
        pbar.close()

    print(f"Saved gaze injector to: {out_dir / 'gaze_injector.pt'}")
    if args.train_projector_lora:
        print(f"Saved LoRA adapter to: {out_dir / 'projector_lora'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
