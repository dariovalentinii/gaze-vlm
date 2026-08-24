"""Shared training data structures used by LGG and Dual Encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.prompts import PROMPTS_FT


class JsonlGazePromptOnly(Dataset):
    """Loads JSONL lines with image + heatmap + prompt/cor."""

    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.path}")

        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON on line {ln} of {self.path}: {e}")

                for k in ("image_path", "heatmap_path"):
                    if k not in r:
                        raise ValueError(f"Missing key '{k}' on line {ln} of {self.path}")

                if ("prompt" not in r) and ("cor" not in r):
                    raise ValueError(f"Line {ln} of {self.path} must include either 'prompt' or 'cor'.")
                if "cor" in r and r["cor"] is not None and r["cor"] not in PROMPTS_FT:
                    raise ValueError(
                        f"Unknown cor '{r['cor']}' on line {ln} of {self.path}. "
                        f"Available: {sorted(PROMPTS_FT.keys())}"
                    )

                self.rows.append(r)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_heatmap_npy(path: str) -> torch.Tensor:
        arr = np.load(path)
        # Accept (H,W) or (1,1,H,W)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected heatmap shape {arr.shape} for {path}")
        return torch.from_numpy(arr).float()  # [1,1,H,W]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        img_path = Path(r["image_path"])
        hm_path = Path(r["heatmap_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not hm_path.exists():
            raise FileNotFoundError(f"Heatmap not found: {hm_path}")

        image = Image.open(img_path).convert("RGB")
        heatmap = self._load_heatmap_npy(str(hm_path))

        return {
            "image": image,
            "heatmap": heatmap,
            "prompt": r.get("prompt", None),
            "cor": r.get("cor", None),
            "target": r.get("target", None),  # ignored
        }


def set_tokenizer_padding(processor: Any) -> None:
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"


def resolve_prompt(prompt: Optional[str], cor: Optional[str]) -> str:
    if prompt is not None and isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if cor is None:
        raise ValueError("No 'prompt' provided and 'cor' is None.")
    return PROMPTS_FT[cor]


@dataclass
class Batch:
    inputs: Dict[str, torch.Tensor]
    heatmaps: List[torch.Tensor]
