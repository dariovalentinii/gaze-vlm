"""Data loading and batching helpers for inference."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.constants import (
    COR_LABELS,
    HEATMAP_EXT,
    IMAGE_EXT,
    IMAGE_NUM_END,
    IMAGE_NUM_START,
)
from src.data.heatmaps import load_heatmap_npy


def normalize_inference_inputs(images, heatmaps, cors):
    is_single = isinstance(images, Image.Image)
    images = [images] if is_single else images
    heatmaps = [heatmaps] if is_single else heatmaps
    cors = [cors] if is_single else cors
    return images, heatmaps, cors, is_single


def build_entries(
    images_dir: Path,
    heatmaps_dir: Path,
    img_num: int = None,
    cor: str = None,
) -> list[dict]:
    entries: list[dict] = []

    # Single image mode
    if img_num is not None:
        image_path = images_dir / f"cogbench_v1_{img_num}.{IMAGE_EXT}"
        heatmap_path = heatmaps_dir / f"cogbench_v1_{img_num}_{cor}.{HEATMAP_EXT}"
        entries.append(
            {
                "image_path": str(image_path),
                "heatmap_path": str(heatmap_path),
                "cor": cor,
            }
        )
    # Batch mode
    else:
        for img_num in range(IMAGE_NUM_START, IMAGE_NUM_END + 1):
            for cor in COR_LABELS:
                image_path = images_dir / f"cogbench_v1_{img_num}.{IMAGE_EXT}"
                heatmap_path = heatmaps_dir / f"cogbench_v1_{img_num}_{cor}.{HEATMAP_EXT}"
                entries.append(
                    {
                        "image_path": str(image_path),
                        "heatmap_path": str(heatmap_path),
                        "cor": cor,
                    }
                )
    return entries


def load_entries_from_jsonl(
    jsonl_path: Path,
    require_heatmap: bool = True,
) -> list[dict]:
    """Load inference entries, allowing Baseline records to omit heatmap_path."""
    entries: list[dict] = []
    with open(jsonl_path, "r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_path = row.get("image_path")
            heatmap_path = row.get("heatmap_path")
            cor = row.get("cor")
            if not image_path or not cor or (require_heatmap and not heatmap_path):
                continue

            entry = {
                "image_path": str(image_path),
                "cor": cor,
            }
            if heatmap_path:
                entry["heatmap_path"] = str(heatmap_path)
            entries.append(entry)
    return entries


@dataclass
class InferenceBatch:
    entries: List[Dict[str, Any]]
    images: List[Image.Image]
    heatmaps: Optional[List[torch.Tensor]]
    cors: List[str]
    errors: List[Dict[str, Any]]


class InferenceDataset(Dataset):
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        dtype: torch.dtype,
        load_heatmaps: bool = True,
    ) -> None:
        self.entries = entries
        self.dtype = dtype
        self.load_heatmaps = load_heatmaps

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image_path = Path(entry["image_path"])
        heatmap_path = None
        if self.load_heatmaps and entry.get("heatmap_path"):
            heatmap_path = Path(entry["heatmap_path"])

        missing = []
        if not image_path.exists():
            missing.append(f"image_path not found: {image_path}")
        if self.load_heatmaps and heatmap_path is None:
            missing.append("heatmap_path missing")
        elif heatmap_path is not None and not heatmap_path.exists():
            missing.append(f"heatmap_path not found: {heatmap_path}")

        if missing:
            return {
                "entry": entry,
                "image": None,
                "heatmap": None,
                "cor": entry.get("cor"),
                "error": "; ".join(missing),
            }

        image = Image.open(image_path).convert("RGB")
        heatmap = None
        if heatmap_path is not None:
            heatmap = load_heatmap_npy(str(heatmap_path), device="cpu", dtype=self.dtype)
        return {
            "entry": entry,
            "image": image,
            "heatmap": heatmap,
            "cor": entry["cor"],
            "error": None,
        }


def collate_inference(batch: List[Dict[str, Any]]) -> InferenceBatch:
    entries: List[Dict[str, Any]] = []
    images: List[Image.Image] = []
    heatmaps: Optional[List[torch.Tensor]] = []
    cors: List[str] = []
    errors: List[Dict[str, Any]] = []

    for item in batch:
        if item.get("error"):
            errors.append(item)
            continue
        entries.append(item["entry"])
        images.append(item["image"])
        if item["heatmap"] is None:
            heatmaps = None
        elif heatmaps is not None:
            heatmaps.append(item["heatmap"])
        cors.append(item["cor"])

    return InferenceBatch(
        entries=entries,
        images=images,
        heatmaps=heatmaps,
        cors=cors,
        errors=errors,
    )
