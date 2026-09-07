"""Image-token position inference used by the LLaVA 1.5 LGG trainers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from training.attention_positions import _find_longest_run_positions


def infer_image_token_positions_per_sample(
    input_ids_b: torch.Tensor,
    attn_mask_b: torch.Tensor,
    attn_seq_len: int,
    image_token_id: int,
    num_image_tokens: int,
) -> Tuple[List[int], int]:
    """Infer positions of image tokens (length=num_image_tokens) in the *attention* sequence.

    Returns (img_positions, q_idx) where q_idx is the last valid token index (for the query).

        Handles common multimodal tokenization patterns:
            A) Processor already expanded <image> into image tokens in input_ids.
            B) Processor keeps one-or-more contiguous <image> placeholders; model expands internally.
    """

    # Only consider valid (non-pad) part of the processor sequence.
    valid_len_in = int(attn_mask_b.sum().item())
    input_ids_valid = input_ids_b[:valid_len_in]

    img_positions_in = (input_ids_valid == image_token_id).nonzero(as_tuple=False).view(-1).tolist()

    # A) Already expanded in input_ids: use positions directly.
    if len(img_positions_in) == num_image_tokens:
        q_idx = min(valid_len_in, attn_seq_len) - 1
        return img_positions_in, q_idx

    # No image placeholders/tokens found.
    if len(img_positions_in) == 0:
        raise RuntimeError(
            "Could not infer image token positions: no image tokens/placeholders found in valid input_ids. "
            f"valid_len={valid_len_in}, image_token_id={image_token_id}."
        )

    # B) Placeholder expansion path: we support contiguous placeholder runs (length P >= 1)
    # that expand internally to num_image_tokens in the attention sequence.
    run = _find_longest_run_positions(input_ids_valid == image_token_id)
    if len(run) != len(img_positions_in):
        raise RuntimeError(
            "Could not infer image token positions with non-contiguous image placeholders. "
            f"Found {len(img_positions_in)} placeholders but longest contiguous run is {len(run)}. "
            "This function currently expects a single contiguous image-placeholder span per sample."
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
            "This likely means model/processor tokenization differs from assumptions."
        )
    return img_positions, q_idx


def attention_alignment_loss(
    attentions: Tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gaze_probs: torch.Tensor,
    image_token_id: int,
    num_last_layers: int = 1,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    eps: float = 1e-8,
    loss_type: str = "kl",
) -> torch.Tensor:
    """Compute a distribution-matching loss between LLM attention on image tokens and gaze.

    attentions: tuple of length L, each [B, H, S, S]
    gaze_probs: [B, N] (already normalized to sum=1)

    We use attention from the last valid token (q_idx) to image token positions.
    When layer_start or layer_end is set, the corresponding Python slice is used.
    Otherwise, the final num_last_layers layers are used. Attention is averaged
    over heads and the selected layers.
    """
    if len(attentions) == 0:
        raise RuntimeError("Model did not return attentions. Ensure output_attentions=True.")

    B = input_ids.shape[0]
    N = gaze_probs.shape[1]

    if layer_start is not None or layer_end is not None:
        use_layers = attentions[layer_start:layer_end]
        if len(use_layers) == 0:
            raise ValueError(
                f"Empty attention layer selection: start={layer_start}, end={layer_end}, "
                f"num_layers={len(attentions)}."
            )
    else:
        use_layers = attentions[-num_last_layers:]

    # Accumulate per-layer attention distributions over image tokens.
    attn_img_sum = None

    for attn in use_layers:
        # [B,H,S,S]
        S = attn.shape[-1]
        per_b: List[torch.Tensor] = []
        for b in range(B):
            img_pos, q_idx = infer_image_token_positions_per_sample(
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
