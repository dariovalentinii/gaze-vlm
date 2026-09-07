#!/usr/bin/env python3
"""Dual Encoding training for LLaVA 1.5.

  - Keep LGG learnable gaze injection (GazeInjector).
  - Keep LGG optional LoRA on the multimodal projector.
  - Add a *separate* heatmap encoder: a fine-tuned copy of the same ViT used for images.
  - Fuse image-vision features (optionally gaze-weighted) with heatmap-vision features,
    then feed to the LLM as usual.

Training objective remains attention-alignment (KL/MSE/CE) between
LLM attention over image patch tokens and the gaze distribution.

Input JSONL schema is identical to LGG training.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import random
from contextlib import nullcontext
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


# -----------------------
# Imports from repo
# -----------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.data import JsonlGazePromptOnly, set_tokenizer_padding
from training.prompting_llava import collate_fn
from training.dual_encoding.encoder import (
    _heatmap_to_rgb_pil,
    heatmaps_to_rgb_pils,
    forward_heatmap_encoder,
    freeze_all_params,
    clone_vision_tower,
    restrict_vision_lora_to_last_k_layers,
)
from training.dual_encoding.objectives import distill_kl_loss
from training.dual_encoding.attention_llava15 import (
    _find_longest_run_positions,
    attention_alignment_loss,
    infer_image_token_positions_per_sample,
)
from src.data.heatmaps import heatmap_to_patch_weights, DualEncodingGazeInjector
from src.data.prompts import PROMPTS_FT
from src.models.unwrapping import unwrap_to_llava


# -----------------------
# Heatmap preprocessing
# -----------------------
@torch.no_grad()
def heatmaps_to_weights_for_attention(
    processor: Any,
    vision_patch_size: int,
    device: torch.device,
    heatmap_pils: Optional[List[Image.Image]] = None,
) -> torch.Tensor:
    """Returns weights [B,N] aligned to the patch grid after processor.image_processor."""

    hm_out = processor.image_processor(
        images=heatmap_pils,
        return_tensors="pt",
        do_normalize=False,
        do_rescale=True,
    )["pixel_values"]  # [B,3,H',W']

    grid_h = hm_out.shape[-2] // vision_patch_size
    grid_w = hm_out.shape[-1] // vision_patch_size
    hm_gray = hm_out.mean(dim=1, keepdim=True)  # [B,1,H',W']
    w = heatmap_to_patch_weights(hm_gray, grid_h, grid_w)  # [B,N]
    return w.to(device)


def heatmaps_to_pixel_values_for_encoder(
    processor: Any,
    device: torch.device,
    heatmap_pils: Optional[List[Image.Image]] = None,
) -> torch.Tensor:
    """Returns CLIP-normalized pixel_values [B,3,H',W'] for the heatmap encoder."""
    pix = processor.image_processor(images=heatmap_pils, return_tensors="pt")["pixel_values"]
    return pix.to(device)

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

    # Mix objectives
    ap.add_argument("--lambda_attn", type=float, default=0.05)
    ap.add_argument("--lambda_distill", type=float, default=1.0)
    ap.add_argument("--lambda_gate", type=float, default=0.05)
    ap.add_argument("--distill_temp", type=float, default=2.0)
    ap.add_argument("--attn_ramp_updates", type=int, default=200)
    ap.add_argument("--gaze_label_smoothing", type=float, default=0.05)

    ap.add_argument("--p_no_gaze", type=float, default=0.0)

    # Learnable injector
    ap.add_argument("--injector_init_scale", type=float, default=1.0)
    ap.add_argument("--injector_init_bias", type=float, default=0.0)
    ap.add_argument("--injector_min_gate", type=float, default=0.05)

    # Fusion
    ap.add_argument("--fusion_init_alpha", type=float, default=1.0)
    ap.add_argument("--fusion_fuse_cls", action="store_true", help="If set, fuse CLS token too.")

    # Optional: LoRA on projector
    ap.add_argument("--train_projector_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--init_lora_dir", type=str, default=None)

    # Heatmap encoder fine-tuning
    ap.add_argument("--train_heatmap_encoder_lora", action="store_true")
    ap.add_argument("--heatmap_lora_r", type=int, default=8)
    ap.add_argument("--heatmap_lora_alpha", type=int, default=16)
    ap.add_argument("--heatmap_lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--heatmap_lora_last_k",
        type=int,
        default=0,
        help="If >0, only keep LoRA params trainable for the last K ViT blocks (others frozen).",
    )
    ap.add_argument(
        "--init_heatmap_lora_dir",
        type=str,
        default=None,
        help="Path to an existing heatmap-encoder LoRA adapter dir to initialize from.",
    )

    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument(
        "--cuda_empty_cache_every",
        type=int,
        default=0,
        help="If >0, run gc.collect()+torch.cuda.empty_cache() every N optimizer updates.",
    )

    args = ap.parse_args()

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

    # Base model
    base = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=amp_dtype,
        device_map=None,
        attn_implementation="eager",
    ).to(device)
    base.config.use_cache = False
    if args.grad_ckpt and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    # Optional projector LoRA
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

    # Freeze everything; we will re-enable only what we need.
    freeze_all_params(model)

    # Trainable injector
    core_llava = unwrap_to_llava(model)
    injector = DualEncodingGazeInjector(
        d_model=core_llava.config.vision_config.hidden_size,
        init_scale=args.injector_init_scale,
        init_bias=args.injector_init_bias,
        min_gate=args.injector_min_gate,
    ).to(device)

    # Ensure LoRA params are trainable if enabled
    if args.train_projector_lora:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

    # Heatmap encoder
    img_vision = core_llava.model.vision_tower

    heatmap_encoder_base = clone_vision_tower(img_vision).to(device)
    if args.grad_ckpt and hasattr(heatmap_encoder_base, "gradient_checkpointing_enable"):
        heatmap_encoder_base.gradient_checkpointing_enable()

    # Freeze heatmap encoder by default
    freeze_all_params(heatmap_encoder_base)
    heatmap_encoder: nn.Module = heatmap_encoder_base

    if args.train_heatmap_encoder_lora:
        if args.init_heatmap_lora_dir:
            kwargs = {}
            sig = inspect.signature(PeftModel.from_pretrained)
            if "is_trainable" in sig.parameters:
                kwargs["is_trainable"] = True
            heatmap_encoder = PeftModel.from_pretrained(heatmap_encoder_base, args.init_heatmap_lora_dir, **kwargs)
        else:
            hm_lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.heatmap_lora_r,
                lora_alpha=args.heatmap_lora_alpha,
                lora_dropout=args.heatmap_lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            )
            heatmap_encoder = get_peft_model(heatmap_encoder_base, hm_lora_cfg)

        # Make LoRA params trainable
        for n, p in heatmap_encoder.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        # Optionally restrict to last K blocks
        num_layers = int(getattr(img_vision.config, "num_hidden_layers", 0) or 0)
        if num_layers > 0:
            restrict_vision_lora_to_last_k_layers(heatmap_encoder, args.heatmap_lora_last_k, num_layers)

    # Collect trainable params
    trainable_params: List[nn.Parameter] = list(injector.parameters())
    if args.train_projector_lora:
        trainable_params += [p for p in model.parameters() if p.requires_grad]
    if args.train_heatmap_encoder_lora:
        trainable_params += [p for p in heatmap_encoder.parameters() if p.requires_grad]

    # Deduplicate
    seen = set()
    unique_params = []
    for p in trainable_params:
        if id(p) not in seen:
            unique_params.append(p)
            seen.add(id(p))
    trainable_params = unique_params

    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Enable at least one of: projector LoRA / heatmap LoRA.")

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
    patch_size = getattr(core_llava.config.vision_config, "patch_size", 14)
    vision_layer = getattr(core_llava.config, "vision_feature_layer", None)
    num_vision_layers = int(getattr(core_llava.config.vision_config, "num_hidden_layers", 0) or 0)

    def _vision_layer_to_encoder_block_idx(layer_idx: Optional[int], n_layers: int) -> Optional[int]:
        if layer_idx is None or n_layers <= 0:
            return None
        hs_len = n_layers + 1  # CLIP hidden_states includes embeddings at index 0
        hs_idx = layer_idx if layer_idx >= 0 else (hs_len + layer_idx)
        if hs_idx <= 0 or hs_idx > n_layers:
            return None
        return hs_idx - 1

    def _find_encoder_block_for_hook(m: nn.Module, block_idx: int) -> Optional[nn.Module]:
        suffix = f"vision_model.encoder.layers.{block_idx}"
        for name, mod in m.named_modules():
            if name.endswith(suffix):
                return mod
        return None

    def _vision_layer_requires_hidden_states(layer_idx: Optional[int], n_layers: int) -> bool:
        if layer_idx is None:
            return False
        if layer_idx == -1:
            return False
        if n_layers > 0 and layer_idx == n_layers:
            return False
        return True

    image_token_id = getattr(core_llava.config, "image_token_index", None)
    if image_token_id is None:
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            image_token_id = None
    if image_token_id is None or image_token_id < 0:
        raise RuntimeError(
            "Could not resolve image token id (config.image_token_index or '<image>'). "
            "Cannot align attention to image tokens."
        )

    # Hook state
    gaze_state: Dict[str, Optional[torch.Tensor]] = {"weights": None, "gates": None}
    # gaze_state["weights"] contains heatmap patch embeddings to be injected
    # gaze_state["gates"] are the actual gates applied to the patch embeddings (after sigmoid+scaling) --> used to compute gate loss

    heatmap_state: Dict[str, Optional[torch.Tensor]] = {"layer_out": None}
    # heatmap_state["layer_out"] contains the output of the heatmap encoder at the selected layer --> used to compute gaze_state["weights"] (by dropping the cls token))

    def _select_vision_feats(out: Any) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        if hasattr(out, "hidden_states") and vision_layer is not None and out.hidden_states is not None:
            feats = out.hidden_states[vision_layer]
            hs = list(out.hidden_states)
            return feats, hs
        feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return feats, None

    def _vision_hook(_module, _inp, out):
        w = gaze_state["weights"]
        if w is None:
            return out

        feats, hs = _select_vision_feats(out)
        cls = feats[:, :1, :]
        patches = feats[:, 1:, :]

        if patches.shape[1] != w.shape[1]:
            raise RuntimeError(f"Patch count mismatch: {patches.shape[1]} vs weights {w.shape[1]}")

        g = injector(w).to(patches.dtype)  # [B,N,1]
        gaze_state["gates"] = g
        patches_new = patches * g

        new_feats = torch.cat([cls, patches_new], dim=1)

        if hs is not None:
            hs[vision_layer] = new_feats
            out.hidden_states = tuple(hs)
        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = new_feats
        return out

    handle = core_llava.model.vision_tower.register_forward_hook(_vision_hook)

    heatmap_hook_handle = None
    hm_block_idx = _vision_layer_to_encoder_block_idx(vision_layer, num_vision_layers)
    if hm_block_idx is not None:
        hm_block = _find_encoder_block_for_hook(heatmap_encoder, hm_block_idx)
        if hm_block is not None:
            def _heatmap_block_hook(_module, _inp, out):
                heatmap_state["layer_out"] = out[0] if isinstance(out, (tuple, list)) else out
                
            heatmap_hook_handle = hm_block.register_forward_hook(_heatmap_block_hook)

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
                    "train_heatmap_encoder_lora": bool(args.train_heatmap_encoder_lora),
                    "injector": {
                        "type": "affine_sigmoid",
                        "init_scale": args.injector_init_scale,
                        "init_bias": args.injector_init_bias,
                        "min_gate": args.injector_min_gate,
                    },
                    "projector_lora": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "targets": ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"],
                    },
                    "heatmap_lora": {
                        "r": args.heatmap_lora_r,
                        "alpha": args.heatmap_lora_alpha,
                        "dropout": args.heatmap_lora_dropout,
                        "targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                        "last_k": args.heatmap_lora_last_k,
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
    heatmap_encoder.train()

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
        heatmap_pils = heatmaps_to_rgb_pils(b.heatmaps)

        # patch weights from heatmap (ONLY to compute gaze target for attention alignment)
        w = heatmaps_to_weights_for_attention(
            processor,
            patch_size,
            device,
            heatmap_pils=heatmap_pils,
        )

        # Possibly disable gaze for this batch.
        # Always use heatmap patch weights (w) to compute the gaze target for attention alignment, since it is more faithful than embedded heatmap
        use_gaze = (random.random() >= args.p_no_gaze)
        if use_gaze:
            gaze_target = w.clamp_min(0)
            gaze_target = gaze_target / (gaze_target.sum(dim=1, keepdim=True) + 1e-8)
        else:
            gaze_target = torch.full_like(w, 1.0 / w.shape[1])

        # Gaze label smoothing
        if args.gaze_label_smoothing > 0:
            beta = float(args.gaze_label_smoothing)
            gaze_target = (1.0 - beta) * gaze_target + beta * (1.0 / gaze_target.shape[1])
            gaze_target = gaze_target / (gaze_target.sum(dim=1, keepdim=True) + 1e-8)

        # Heatmap encoder feats (only if enabled for this batch)
        if use_gaze and args.train_heatmap_encoder_lora:
            need_hm_hidden_states = (
                _vision_layer_requires_hidden_states(vision_layer, num_vision_layers)
                and heatmap_hook_handle is None
            )
            heatmap_state["layer_out"] = None
            hm_pix = heatmaps_to_pixel_values_for_encoder(
                processor,
                device,
                heatmap_pils=heatmap_pils,
            )
            hm_out = forward_heatmap_encoder(
                heatmap_encoder=heatmap_encoder,
                pixel_values=hm_pix,
                output_hidden_states=need_hm_hidden_states,
                return_dict=True,
            )

            if heatmap_state["layer_out"] is not None:
                hm_feats_all = heatmap_state["layer_out"]
            elif need_hm_hidden_states:
                hm_feats_all = hm_out.hidden_states[vision_layer]
            else:
                hm_feats_all = hm_out.last_hidden_state

            heatmap_state["layer_out"] = None
            
            hm_feats = hm_feats_all[:, 1:, :]
            if hm_feats.shape[1] != w.shape[1]:
                raise RuntimeError(
                    f"Heatmap patch count mismatch: hm_feats={hm_feats.shape[1]} vs gaze_weights={w.shape[1]}"
                )
            gaze_state["weights"] = hm_feats
        else:
            gaze_state["weights"] = None

        inputs = {k: v.to(device) for k, v in b.inputs.items()}
        attn_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))

        out_s = model(**inputs, output_attentions=True, return_dict=True)
        gaze_state["weights"] = None
        if use_gaze and args.train_heatmap_encoder_lora:
            del hm_out, hm_feats_all, hm_feats, hm_pix

        # Gate regularization (keep gates near identity).
        g_actual = gaze_state["gates"]
        if g_actual is not None:
            loss_gate = ((g_actual - 1.0) ** 2).mean()
        else:
            loss_gate = torch.tensor(0.0, device=device)
        gaze_state["gates"] = None

        loss_attn = attention_alignment_loss(
            attentions=out_s.attentions,
            input_ids=inputs["input_ids"],
            attention_mask=attn_mask,
            gaze_probs=gaze_target,
            image_token_id=int(image_token_id),
            num_last_layers=args.attn_last_layers,
            loss_type=args.loss,
        )

        # Logit distillation: teacher = same model with NO gaze injection and adapters disabled.
        loss_distill = torch.tensor(0.0, device=device)
        if args.lambda_distill > 0:
            # prev_w = gaze_state["weights"]
            # gaze_state["weights"] = None
            was_training = model.training
            model.eval()
            ctx_m = model.disable_adapter() if isinstance(model, PeftModel) and hasattr(model, "disable_adapter") else nullcontext()
            with torch.inference_mode():
                with ctx_m:
                    out_t = model(**inputs, output_attentions=False, return_dict=True)
            if was_training:
                model.train()
            # gaze_state["weights"] = prev_w
            loss_distill = distill_kl_loss(
                logits_s=out_s.logits.float(),
                logits_t=out_t.logits.float(),
                attention_mask=attn_mask,
                temperature=args.distill_temp,
            )
            del out_t

        del out_s, inputs, attn_mask, heatmap_pils

        lam_attn = _lambda_attn_now(update_idx)
        loss_total = lam_attn * loss_attn + float(args.lambda_distill) * loss_distill + float(args.lambda_gate) * loss_gate
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
        heatmap_encoder.eval()
        tot = 0.0
        n = 0
        for b in dl:
            loss, _ = _forward_batch(b, update_idx=global_update)
            tot += float(loss.detach().cpu())
            n += 1
        model.train()
        injector.train()
        heatmap_encoder.train()
        return tot / max(n, 1)

    pbar = tqdm(total=total_updates, desc="updates")
    try:
        for epoch in range(args.epochs):
            for b in train_dl:
                if global_update >= total_updates:
                    break

                with torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext():
                    loss, loss_metrics = _forward_batch(b, update_idx=global_update)
                    loss = loss / args.grad_accum

                loss.backward()
                running_loss_total += loss_metrics["loss_total"]
                running_loss_attn += loss_metrics["loss_attn"]
                running_loss_distill += loss_metrics["loss_distill"]
                running_loss_gate += loss_metrics["loss_gate"]
                accum += 1

                if accum >= args.grad_accum:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_update += 1
                    pbar.update(1)

                    if (
                        device.type == "cuda"
                        and args.cuda_empty_cache_every > 0
                        and (global_update % args.cuda_empty_cache_every == 0)
                    ):
                        gc.collect()
                        torch.cuda.empty_cache()

                    if global_update % args.log_every == 0:
                        if device.type == "cuda":
                            alloc = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            max_alloc = torch.cuda.max_memory_allocated() / 1024**3
                            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
                            print(
                                f"[cuda] allocated={alloc:.2f} GB reserved={reserved:.2f} GB "
                                f"peak_alloc={max_alloc:.2f} GB peak_reserved={max_reserved:.2f} GB"
                            )
                            torch.cuda.reset_peak_memory_stats()
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

                    #     if args.train_projector_lora:
                    #         model.save_pretrained(str(ckpt_dir / "projector_lora"))

                    #     if args.train_heatmap_encoder_lora:
                    #         heatmap_encoder.save_pretrained(str(ckpt_dir / "heatmap_encoder_lora"))

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
        if args.train_heatmap_encoder_lora:
            heatmap_encoder.save_pretrained(str(out_dir / "heatmap_encoder_lora"))

    finally:
        if heatmap_hook_handle is not None:
            heatmap_hook_handle.remove()
        handle.remove()
        pbar.close()

    print(f"Saved injector: {out_dir / 'gaze_injector.pt'}")
    if args.train_projector_lora:
        print(f"Saved projector LoRA: {out_dir / 'projector_lora'}")
    if args.train_heatmap_encoder_lora:
        print(f"Saved heatmap encoder LoRA: {out_dir / 'heatmap_encoder_lora'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
