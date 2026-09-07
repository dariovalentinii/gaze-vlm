from __future__ import annotations
import torch


def unwrap_to_llava(model: torch.nn.Module):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model  # type: ignore
    if hasattr(model, "get_base_model"):
        return model.get_base_model()  # type: ignore
    return model  # type: ignore
