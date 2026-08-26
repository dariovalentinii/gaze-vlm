"""LLaVA 1.5 attention-alignment helpers for Dual Encoding training."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _find_longest_run_positions(mask_1d: torch.Tensor) -> List[int]:
    idx = mask_1d.nonzero(as_tuple=False).view(-1).tolist()
    if not idx:
        return []
    best_run: List[int] = []
    cur: List[int] = [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            if len(cur) > len(best_run):
                best_run = cur
            cur = [i]
    if len(cur) > len(best_run):
        best_run = cur
    return best_run


def infer_image_token_positions_per_sample(
    input_ids_b: torch.Tensor,
    attn_mask_b: torch.Tensor,
    attn_seq_len: int,
    image_token_id: int,
    num_image_tokens: int,
) -> Tuple[List[int], int]:
    valid_len_in = int(attn_mask_b.sum().item())
    input_ids_valid = input_ids_b[:valid_len_in]

    img_mask = (input_ids_valid == image_token_id)
    run = _find_longest_run_positions(img_mask)

    if len(run) == num_image_tokens:
        q_idx = valid_len_in - 1
        return run, q_idx

    if len(run) == 1:
        placeholder_pos = run[0]
        valid_len_out = valid_len_in - 1 + num_image_tokens
        q_idx = valid_len_out - 1
        img_positions = list(range(placeholder_pos, placeholder_pos + num_image_tokens))
        if valid_len_out > attn_seq_len:
            raise RuntimeError(
                f"Expanded valid len {valid_len_out} exceeds attention seq len {attn_seq_len}. "
                f"This likely means the model/processor tokenization differs from assumptions."
            )
        return img_positions, q_idx

    raise RuntimeError(
        "Could not infer image token positions. "
        f"Found image-token run length={len(run)} in input_ids (valid_len={valid_len_in}). "
        f"Expected either 1 or {num_image_tokens}."
    )


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
    if len(attentions) == 0:
        raise RuntimeError("Model did not return attentions. Ensure output_attentions=True.")

    B = input_ids.shape[0]
    N = gaze_probs.shape[1]
    use_layers = attentions[-num_last_layers:]

    attn_img_sum = None
    for attn in use_layers:
        S = attn.shape[-1]
        per_b: List[torch.Tensor] = []
        for b in range(B):
            img_pos, q_idx = infer_image_token_positions_per_sample(
                input_ids[b], attention_mask[b], S, image_token_id, N
            )
            q_idx = min(q_idx, S - 1)

            a_q = attn[b, :, q_idx, :].mean(dim=0)  # [S]
            # a_img = a_q[torch.tensor(img_pos, device=a_q.device, dtype=torch.long)]
            start = img_pos[0]
            a_img = a_q[start : start + N]
            a_img = a_img.clamp_min(0)
            a_img = a_img / (a_img.sum() + eps)
            per_b.append(a_img)

        attn_img = torch.stack(per_b, dim=0)  # [B,N]
        attn_img_sum = attn_img if attn_img_sum is None else (attn_img_sum + attn_img)

    attn_probs = attn_img_sum / float(len(use_layers))

    if loss_type == "mse":
        return F.mse_loss(attn_probs, gaze_probs)
    if loss_type == "kl":
        log_attn = (attn_probs + eps).log()
        return F.kl_div(log_attn, gaze_probs, reduction="batchmean")
    if loss_type == "ce":
        log_attn = (attn_probs + eps).log()
        return -(gaze_probs * log_attn).sum(dim=1).mean()
    raise ValueError(f"Unknown loss_type='{loss_type}'. Choose from: kl, mse, ce")
