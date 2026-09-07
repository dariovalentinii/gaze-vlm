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
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

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

from training.data import JsonlGazePromptOnly, set_tokenizer_padding
from training.prompting_onevision import collate_fn
from training.modeling import select_model_and_adapter_classes
from training.lgg.attention_onevision import (
    _apply_rotary_pos_emb,
    _apply_rope_qk,
    _find_longest_run_positions,
    _get_llm_layers,
    _repeat_kv,
    _resolve_attn_layout,
    _resolve_rope_cos_sin,
    attention_alignment_loss,
    attention_alignment_loss_from_captured,
    infer_image_token_positions_per_sample,
    install_lastk_attn_slice_capture,
)
from training.lgg.common import (
    build_per_sample_gaze_targets,
    distill_kl_loss,
    freeze_all_params,
    preprocess_heatmaps_to_weights,
)
from src.data.heatmaps import apply_patch_weighting, GazeInjector, pack_gaze_probs_like_llava_next
from src.data.prompts import PROMPTS_FT
from src.models.unwrapping import unwrap_to_llava
from src.models.llava_15 import LlavaHFAdapter
from src.models.llava_next import LlavaNextHFAdapter
from src.models.llava_onevision import LlavaOnevisionHFAdapter


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-chat-hf")
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
