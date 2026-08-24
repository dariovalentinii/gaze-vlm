"""Model and adapter selection shared by the training entry points."""

from __future__ import annotations

from typing import Any, Tuple

from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
)

from src.models.llava_15 import LlavaHFAdapter
from src.models.llava_next import LlavaNextHFAdapter
from src.models.llava_ov import LlavaOnevisionHFAdapter


def select_model_and_adapter_classes(model_name: str) -> Tuple[Any, Any]:
    m = model_name.lower()
    if "onevision" in m or "ov-chat" in m:
        return LlavaOnevisionForConditionalGeneration, LlavaOnevisionHFAdapter
    if "v1.6" in m or "next" in m:
        return LlavaNextForConditionalGeneration, LlavaNextHFAdapter
    return LlavaForConditionalGeneration, LlavaHFAdapter
