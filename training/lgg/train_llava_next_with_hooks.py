#!/usr/bin/env python3
"""Learnable Gaze Gating training with attention hooks.

Learn how to inject gaze and align model attention to human gaze,
without supervising on the ground-truth text targets.

Implementation:
  1) Gaze injection is learnable via a tiny gating module (default: affine + sigmoid).
  2) Training loss is an attention alignment loss (default: KL) between
     - model attention mass over image patch tokens
     - gaze distribution over patches
  3) Optionally, also train the multimodal projector (via LoRA) using the same attention loss.

JSONL schema per line:
Required:
  - image_path: str
  - heatmap_path: str (npy, (H,W) or (1,1,H,W))
  - prompt: str  OR  cor: str (key into src.data.prompts.PROMPTS_FT)
Optional:
  - target: str  (ignored here; kept for backward compatibility)

Example:
{"image_path":".../img.jpg","heatmap_path":".../hm.npy","cor":"5_ER"}

Run from your project root so `import src...` works.
"""

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
from training.prompting_llava import collate_fn
from training.modeling import select_model_and_adapter_classes
from training.lgg.llava15_attention import (
    _find_longest_run_positions,
    attention_alignment_loss,
    infer_image_token_positions_per_sample,
)
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
from src.models.llava_onevision import LlavaOnevisionHFAdapter
from training.attention_utils import LastKAttnCapture


# -----------------------
# Attention alignment
# -----------------------











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
    ap.add_argument(
        "--attn_ramp_updates",
        type=int,
        default=200,
        help="Linearly ramp lambda_attn from 0 to target over this many optimizer updates.",
    )
    ap.add_argument(
        "--gaze_label_smoothing",
        type=float,
        default=0.05,
        help="Mix gaze target with uniform: (1-b)*gaze + b*uniform.",
    )

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

    # Hook state
    gaze_state: Dict[str, Optional[torch.Tensor]] = {"weights": None, "gates": None}

    def _vision_hook(_module, _inp, out):
        w = gaze_state["weights"]
        if w is None:
            return out

        # Pick which vision features are used later.
        if hasattr(out, "hidden_states") and vision_layer is not None and out.hidden_states is not None:
            feats = out.hidden_states[vision_layer]
            hs = list(out.hidden_states)
        else:
            feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            hs = None

        cls = feats[:, :1, :]
        patches = feats[:, 1:, :]

        if patches.shape[1] != w.shape[1]:
            raise RuntimeError(f"Patch count mismatch: {patches.shape[1]} vs weights {w.shape[1]}")

        g = injector(w).to(patches.dtype)  # [B,N]
        gaze_state["gates"] = g
        patches_new = patches * g
        # patches_new = apply_patch_weighting(patches, w)
        new_feats = torch.cat([cls, patches_new], dim=1)

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


            
        # Gaze label smoothing to avoid peaky collapse
        if args.gaze_label_smoothing > 0:
            beta = float(args.gaze_label_smoothing)
            gaze_target = (1.0 - beta) * gaze_target + beta * (1.0 / gaze_target.shape[1])
            gaze_target = gaze_target / (gaze_target.sum(dim=1, keepdim=True) + 1e-8)



        inputs = {k: v.to(device) for k, v in b.inputs.items()}
        attn_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))

        # Ask for attentions
        print("CUDA memory before forward:")
        print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
        cap = LastKAttnCapture(model, k=args.attn_last_layers, layers_path=("model", "model", "language_model", "layers"))
        with cap:
            out_s = model(**inputs, output_attentions=False, return_dict=True)

        print("CUDA memory after forward:")
        print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
        # Gate regularization
        g_actual = gaze_state["gates"]
        if g_actual is not None:
            loss_gate = ((g_actual - 1.0) ** 2).mean()
        else:
            loss_gate = torch.tensor(0.0, device=device)
        gaze_state["gates"] = None
        
        attentions_last_k = tuple(cap.attns)

        # Compute alignment loss
        loss_attn = attention_alignment_loss(
            attentions=attentions_last_k,
            input_ids=inputs["input_ids"],
            attention_mask=attn_mask,
            gaze_probs=gaze_target,
            image_token_id=int(image_token_id),
            num_last_layers=args.attn_last_layers,
            loss_type=args.loss,
        )
        
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
            if was_training:
                model.train()
            gaze_state["weights"] = prev
            loss_distill = distill_kl_loss(
                logits_s=out_s.logits.float(),
                logits_t=out_t.logits.float(),
                attention_mask=attn_mask,
                temperature=args.distill_temp,
            )

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
        pbar.close()

    print(f"Saved gaze injector to: {out_dir / 'gaze_injector.pt'}")
    if args.train_projector_lora:
        print(f"Saved LoRA adapter to: {out_dir / 'projector_lora'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
