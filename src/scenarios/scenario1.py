"""Shared inference protocol and runner.

The legacy S1 names are retained to keep the existing adapter interface stable.
Baseline, LGG, and Dual Encoding all use this thin dispatch layer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol
from PIL import Image
import torch

class S1Adapter(Protocol):
    device: torch.device

    def generate_with_weighted_patches(
        self,
        images: List[Image.Image],
        heatmaps: List[torch.Tensor],
        cfg: "S1Config",
        *,
        cors: List[str],
        use_vision_hook: bool = True,
    ) -> List[str]:
        ...

@dataclass
class S1Config:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    debug: bool = False
    prompt_version: str = "v2"

@dataclass
class Scenario1:
    cfg: S1Config

    def run(
        self,
        adapter: S1Adapter,
        images: List[Image.Image],
        heatmaps: List[torch.Tensor],
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
