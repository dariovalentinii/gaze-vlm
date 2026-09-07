"""Shared inference protocol, configuration, and runner."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol
from PIL import Image
import torch

class InferenceAdapter(Protocol):
    device: torch.device

    def generate_with_weighted_patches(
        self,
        images: List[Image.Image],
        heatmaps: Optional[List[torch.Tensor]],
        cfg: "GenerationConfig",
        *,
        cors: List[str],
        use_vision_hook: bool = True,
    ) -> List[str]:
        ...

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    debug: bool = False
    prompt_version: str = "v2"

@dataclass
class InferenceRunner:
    cfg: GenerationConfig

    def run(
        self,
        adapter: InferenceAdapter,
        images: List[Image.Image],
        heatmaps: Optional[List[torch.Tensor]],
        *,
        cors: List[str],
        use_vision_hook: bool = True,
    ) -> List[str]:

        outs = adapter.generate_with_weighted_patches(
            images=images,
            heatmaps=heatmaps,
            cfg=self.cfg,
            cors=cors,
            use_vision_hook=use_vision_hook,
        )
        return outs
