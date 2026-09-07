"""Shared checkpoint-loading helpers for LLaVA inference adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def resolve_checkpoint_layout(
    path: Path,
    adapter_label: str,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Returns:
      (projector_lora_dir, checkpoint_root_dir)

    Accepted inputs:
      - LGG/DE run dir (contains projector_lora/ or lora_adapter/)
      - LGG/DE projector dir itself (.../projector_lora)
      - legacy LLM adapter directory (.../lora_adapter)
      - LGG/DE run dir without projector LoRA (injector-only): no projector_lora/
    """
    if not path.exists():
        raise FileNotFoundError(f"LoRA path not found: {path}")

    if path.is_dir() and path.name == "projector_lora" and (path / "adapter_config.json").exists():
        print(f"[{adapter_label}] Loaded projector LoRA from: {path}")
        return path, path.parent

    if path.is_dir() and path.name == "lora_adapter" and (path / "adapter_config.json").exists():
        print(f"[{adapter_label}] Loaded lora_adapter from: {path}")
        return path, path.parent

    candidate = path / "projector_lora"
    if candidate.exists() and (candidate / "adapter_config.json").exists():
        print(f"[{adapter_label}] Loaded projector LoRA from: {candidate}")
        return candidate, path

    # lora_adapter candidate (LLM [+ proj])
    candidate = path / "lora_adapter"
    if candidate.exists() and (candidate / "adapter_config.json").exists():
        print(f"[{adapter_label}] Loaded lora_adapter from: {candidate}")
        return candidate, path

    # Injector-only run dir (no projector LoRA or adapter)
    print(f"[{adapter_label}] No LoRA adapters found in: {path}. Checking for injector-only checkpoint...")
    if path.is_dir() and ((path / "gaze_injector.pt").exists() or (path / "components.json").exists()):
        return None, path

    raise ValueError(
        f"Unsupported checkpoint layout at: {path}. "
        "Expected a run folder containing projector_lora/ or lora_adapter/ and/or gaze_injector.pt, "
        "or a direct projector_lora/ or lora_adapter/ folder."
    )


def read_injector_min_gate(inj_dir: Path) -> float:
    min_gate = 0.05

    # LGG/DE sidecar
    comp = inj_dir / "components.json"
    if comp.exists():
        try:
            meta = json.loads(comp.read_text(encoding="utf-8"))
            inj = meta.get("injector", {})
            if isinstance(inj, dict) and "min_gate" in inj:
                return float(inj["min_gate"])
        except Exception:
            pass

    return min_gate


def clone_vision_tower(vision_tower) -> Any:
    cls = vision_tower.__class__
    cloned = cls(vision_tower.config)
    cloned.load_state_dict(vision_tower.state_dict(), strict=True)
    return cloned


def vision_layer_to_encoder_block_idx(layer_idx: Optional[int], n_layers: int) -> Optional[int]:
    if layer_idx is None or n_layers <= 0:
        return None
    hs_len = n_layers + 1
    hs_idx = layer_idx if layer_idx >= 0 else (hs_len + layer_idx)
    if hs_idx <= 0 or hs_idx > n_layers:
        return None
    return hs_idx - 1


def find_encoder_block_for_hook(module_root: Any, block_idx: int) -> Optional[Any]:
    suffix = f"vision_model.encoder.layers.{block_idx}"
    for name, mod in module_root.named_modules():
        if name.endswith(suffix):
            return mod
    return None


def vision_layer_requires_hidden_states(layer_idx: Optional[int], n_layers: int) -> bool:
    if layer_idx is None:
        return False
    if layer_idx == -1:
        return False
    if n_layers > 0 and layer_idx == n_layers:
        return False
    return True
