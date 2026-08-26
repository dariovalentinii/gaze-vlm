"""LLaVA-OneVision attention-alignment helpers for LGG training."""

from __future__ import annotations

import math
from contextlib import nullcontext
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    # Prefer Qwen2 rotary implementation for llava-onevision-qwen2 models.
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as _apply_rotary_pos_emb
except Exception:
    try:
        # Fallback for LLaMA-family checkpoints.
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _apply_rotary_pos_emb
    except Exception:
        _apply_rotary_pos_emb = None

from training.attention_utils import _find_longest_run_positions


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
