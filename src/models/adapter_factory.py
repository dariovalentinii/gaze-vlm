# src/models/adapter_factory.py
from __future__ import annotations
from typing import Optional, Type, Dict
import torch
from transformers import AutoConfig

from src.inference.runner import InferenceAdapter

# Implemented adapters
from src.models.llava_15 import LlavaHFAdapter  # LLaVA-1.5 HF class
from src.models.llava_next import LlavaNextHFAdapter  # LLaVA-1.6 / LLaVA-NeXT
from src.models.llava_ov import LlavaOnevisionHFAdapter # LLaVA-OV (OneVision) 


def _resolve_key(model_name: str, model_type: Optional[str]) -> str:
    """
    Map (model_name, config.model_type) -> adapter key.
    model_type is best; fall back to name heuristics.
    """
    name = model_name.lower()
    mt = (model_type or "").lower()

    # Prefer model_type when possible
    if mt in {"llava"}:
        return "llava_15"
    if mt in {"llava_next"}:
        return "llava_next"
    if mt in {"llava_onevision", "llava-onevision"}:
        return "llava_onevision"
    if mt in {"qwen2_vl"}:
        return "qwen2_vl"
    if mt in {"instructblip"}:
        return "instructblip"

    # Heuristics fallback (useful when remote code / weird configs)
    if "llava" in name and ("1.5" in name or "v1.5" in name or "llava-1.5" in name):
        return "llava_15"
    if "llava" in name and ("1.6" in name or "next" in name or "llava-next" in name):
        return "llava_next"
    if "qwen2" in name and "vl" in name:
        return "qwen2_vl"
    if "instructblip" in name:
        return "instructblip"

    return "unknown"


def create_inference_adapter(
    model_name: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    lora_dir: Optional[str] = None,
) -> InferenceAdapter:
    """
    Returns an adapter implementing generate_with_weighted_patches for inference.
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    key = _resolve_key(model_name, getattr(cfg, "model_type", None))

    registry: Dict[str, Type] = {
        "llava_15": LlavaHFAdapter,
        "llava_next": LlavaNextHFAdapter,
        "llava_onevision": LlavaOnevisionHFAdapter,
    }

    if key not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"No inference adapter for model='{model_name}' (model_type='{getattr(cfg,'model_type',None)}'). "
            f"Resolved key='{key}'. Supported keys: {supported}."
        )

    AdapterCls = registry[key]
    # if key == "llava_15": 
    return AdapterCls(model_name=model_name, device=device, torch_dtype=torch_dtype, lora_dir=lora_dir)
    # else:
    #     return AdapterCls(model_name=model_name, device=device, torch_dtype=torch_dtype)
