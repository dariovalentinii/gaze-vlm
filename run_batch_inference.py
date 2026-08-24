import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src.data.heatmaps import load_heatmap_npy
from src.data.constants import (
    COR_LABELS,
    HEATMAP_EXT,
    IMAGE_EXT,
    IMAGE_NUM_END,
    IMAGE_NUM_START,
    ROOT,
    STRICT_LLAVA_NEXT_BATCH,
)
from src.models.adapter_factory import create_s1_adapter
from src.models.llava_next import LlavaNextHFAdapter
from src.scenarios.scenario1 import Scenario1, S1Config


def _resolve_lora_run_dir(lora_dir: str) -> Path:
    """
    Resolve a run-level checkpoint directory from --lora_dir.

    Supported inputs:
      - run dir (new): contains projector_lora/ and/or gaze_injector.pt
      - projector LoRA subdir: .../projector_lora
      - lora_adapter subdir: .../lora_adapter
      - legacy adapter dir (old): contains adapter_config.json at root or gaze_injector.pt at root
    """
    p = Path(lora_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"LoRA path not found: {p}")

    if p.is_dir() and (p.name == "projector_lora" or p.name == "lora_adapter") and (p / "adapter_config.json").exists():
        return p.parent

    if (p / "projector_lora" / "adapter_config.json").exists() or (p / "lora_adapter" / "adapter_config.json").exists():
        return p

    if (p / "adapter_config.json").exists():
        return p

    if (p / "gaze_injector.pt").exists() or (p / "components.json").exists() or (p / "gaze_injector.json").exists():
        return p

    raise ValueError(
        f"Unsupported --lora_dir layout: {p}. "
        "Expected one of: run dir with projector_lora/, projector_lora dir, lora_adapter/ or lora_adapter dir."
    )


def normalize_scenario_inputs(images, heatmaps, cors):
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


def load_entries_from_jsonl(jsonl_path: Path) -> list[dict]:
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
            if not image_path or not heatmap_path or not cor:
                continue

            entries.append(
                {
                    "image_path": str(image_path),
                    "heatmap_path": str(heatmap_path),
                    "cor": cor,
                }
            )
    return entries


@dataclass
class InferenceBatch:
    entries: List[Dict[str, Any]]
    images: List[Image.Image]
    heatmaps: List[torch.Tensor]
    cors: List[str]
    errors: List[Dict[str, Any]]


class InferenceDataset(Dataset):
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        dtype: torch.dtype,
    ) -> None:
        self.entries = entries
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image_path = Path(entry["image_path"])
        heatmap_path = Path(entry["heatmap_path"])

        missing = []
        if not image_path.exists():
            missing.append(f"image_path not found: {image_path}")
        if not heatmap_path.exists():
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
    heatmaps: List[torch.Tensor] = []
    cors: List[str] = []
    errors: List[Dict[str, Any]] = []

    for item in batch:
        if item.get("error"):
            errors.append(item)
            continue
        entries.append(item["entry"])
        images.append(item["image"])
        heatmaps.append(item["heatmap"])
        cors.append(item["cor"])

    return InferenceBatch(
        entries=entries,
        images=images,
        heatmaps=heatmaps,
        cors=cors,
        errors=errors,
    )


def group_outputs_by_cor(full_outputs_path: Path) -> None:
    """
    Read full_outputs.jsonl and consolidate COR outputs per image.
    Creates one entry per image with all reasoning outputs as separate fields.
    """
    # Mapping from COR labels to field names
    cor_to_field = {
        "0_E": "entities_output",
        "1_STR": "special_time_reasoning_output",
        "2_LR": "location_reasoning_output",
        "3_CR": "character_reasoning_output",
        "4_CRR": "character_relationship_reasoning_output",
        "5_ER": "event_reasoning_output",
        "6_ERR": "event_relationship_reasoning_output",
        "7_NMER": "next_moment_event_reasoning_output",
        "8_MSR": "mental_state_reasoning_output",
    }
    
    metadata = None
    prompt_version = None
    filename_to_outputs: dict[str, dict] = {}

    with open(full_outputs_path, "r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # First row with metadata (model)
            if "model" in row:
                prompt_version = row.get("prompt_version")
                if metadata is None:
                    metadata = row
                continue

            # Data rows
            filename = row.get("filename")
            cor = row.get("cor")
            model_output = row.get("model_output")
            if model_output == "":
                model_output = "None"
            
            if not filename or not cor:
                continue

            # Initialize entry for this filename if needed
            if filename not in filename_to_outputs:
                filename_to_outputs[filename] = {"filename": filename}
            
            # Map COR to field name and store output
            field_name = cor_to_field.get(cor)
            if field_name:
                filename_to_outputs[filename][field_name] = model_output

    # Write consolidated output (atomic + validated before deleting source)
    if prompt_version:
        output_path = full_outputs_path.parent / f"consolidated_{prompt_version}.jsonl"
    else:
        output_path = full_outputs_path.parent / "consolidated.jsonl"

    expected_lines = (1 if metadata else 0) + len(filename_to_outputs)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(tmp_output_path, "w") as f_out:
        # Write metadata first
        if metadata:
            f_out.write(json.dumps(metadata) + "\n")
        
        # Write consolidated entries (sorted by filename for consistency)
        for filename in sorted(filename_to_outputs.keys()):
            entry = filename_to_outputs[filename]
            f_out.write(json.dumps(entry) + "\n")
        f_out.flush()
        os.fsync(f_out.fileno())

    # Atomically replace target and validate JSONL integrity
    os.replace(tmp_output_path, output_path)

    actual_lines = 0
    with open(output_path, "r") as f_check:
        for line in f_check:
            line = line.strip()
            if not line:
                continue
            json.loads(line)  # raises if malformed
            actual_lines += 1

    if actual_lines != expected_lines:
        raise RuntimeError(
            f"Consolidated output validation failed: expected {expected_lines} lines, found {actual_lines}. "
            f"Source file preserved at {full_outputs_path}."
        )
    
    print(f"Consolidated outputs written to {output_path}")

    # Delete full outputs only after successful write+validation of consolidated file
    if full_outputs_path.exists():
        full_outputs_path.unlink()
        print(f"Deleted source full outputs file: {full_outputs_path}")
    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA/checkpoint artifacts (required for scenario 2/3).")

    ap.add_argument("--images_dir", type=str, default=str(ROOT / "data" / "cogbench_v1-1" / "images"),
                    help="Absolute path to images directory")
    ap.add_argument("--heatmaps_dir", type=str, default=str(ROOT / "data" / "heatmaps" / "avg"),
                    help="Absolute path to heatmaps directory")
    ap.add_argument("--entries_jsonl", type=str, default=None,
                    help="Path to JSONL entries (image_path, heatmap_path, cor). Overrides image/heatmap ranges.")
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "results"),
                    help="Output directory for JSONL files")
    ap.add_argument("--img_num", type=int, default=None,
                    help="(Optional) Process single image. If specified, --cor must also be specified. If omitted, processes all images.")
    ap.add_argument("--cor", type=str, default=None,
                    help="(Optional) COR label for single image. Required if --img_num is specified.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true",
                    help="Enable debug stats for gaze-weighted patch injection")
    ap.add_argument("--no_gaze", action="store_true",
                    help="Disable gaze-weighted vision hook")
    ap.add_argument(
        "--scenario",
        type=int,
        choices=[2, 3],
        default=None,
        help="Gaze method: 2 = Learnable Gaze Gating (LGG), 3 = Dual Encoding (DE).",
    )
    ap.add_argument("--prompt_version", type=str, default="v2", choices=["v1", "v2"],
                    help="Prompt set version to use (v1 or v2)")
    
    ap.add_argument("--batch_size", type=int, default=9,
                    help="Batch size for inference (1 for sequential, >1 for batched)")
    args = ap.parse_args()

    # Validate single vs batch mode
    if (args.img_num is not None) != (args.cor is not None):
        print("Error: --img_num and --cor must be specified together, or both omitted")
        sys.exit(1)

    # Validate scenario selection
    if args.scenario is None and not args.no_gaze:
        print("Error: --no_gaze must be set when --scenario is not specified")
        sys.exit(1)
    if args.scenario is not None and args.no_gaze:
        print("Error: --scenario and --no_gaze are mutually exclusive")
        sys.exit(1)
    if args.scenario in {2, 3} and not args.lora_dir:
        print("Error: --lora_dir is required for --scenario 2 (LGG) or 3 (DE)")
        sys.exit(1)

    if args.entries_jsonl:
        entries = load_entries_from_jsonl(Path(args.entries_jsonl).resolve())
    else:
        images_dir = Path(args.images_dir).resolve()
        heatmaps_dir = Path(args.heatmaps_dir).resolve()
        entries = build_entries(
            images_dir=images_dir,
            heatmaps_dir=heatmaps_dir,
            img_num=args.img_num,
            cor=args.cor,
        )
    
    # Determine processing mode
    if args.entries_jsonl:
        mode = f"jsonl ({len(entries)} entries)"
    elif args.img_num is not None:
        mode = f"single ({args.img_num}, {args.cor})"
    else:
        mode = f"all ({len(entries)} combinations)"
    
    print(f"Generated {len(entries)} entries from ranges - processing {mode}")

    # Setup model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"Loading model {args.model}...")
    print(f"Looking for LoRA adapters in: {args.lora_dir}" if args.lora_dir else "No LoRA adapters")
    adapter = create_s1_adapter(
        model_name=args.model,
        device=args.device,
        torch_dtype=dtype_map[args.dtype],
        lora_dir=args.lora_dir,
    )

    # batch size for llava next models must be 1 or 3 to avoid both no-anyres/anyres images
    # in the same batch, which causes issues with the vision hook
    if STRICT_LLAVA_NEXT_BATCH and isinstance(adapter, LlavaNextHFAdapter) and args.batch_size not in {1, 3}:
        print("Error: for LLaVA-NeXT, --batch_size must be 1 or 3")
        sys.exit(1)

    # Config
    cfg = S1Config(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        debug=args.debug,
        prompt_version=args.prompt_version,
    )

    scenario = Scenario1(cfg=cfg)
    if args.no_gaze:
        scenario_name = "baseline"
        scenario_id = 0
    else:
        scenario_id = args.scenario
        scenario_name = {2: "lgg", 3: "dual_encoding"}[scenario_id]
        
    

    # Prepare output file
    output_root = Path(args.output_dir).resolve()
    model_name = args.model
    model_dir_name = model_name.split("/")[-1] if "/" in model_name else model_name
    output_dir = output_root / scenario_name / model_dir_name
    # safety check on model and scenario
    if args.lora_dir:
        lora_run_dir = _resolve_lora_run_dir(args.lora_dir)
        lora_model = lora_run_dir.parent
        lora_model_name = lora_model.name
        if lora_model_name != model_dir_name:
            print(
                f"Error: LoRA directory '{args.lora_dir}' resolves to run '{lora_run_dir.name}' "
                f"for model '{lora_model_name}', not '{model_dir_name}'"
            )
            sys.exit(1)
        lora_scenario_name = lora_model.parent.name
        accepted_scenario_names = {
            0: {"baseline", "no_gaze"},
            2: {"lgg", "scenario2"},
            3: {"dual_encoding", "scenario3"},
        }[scenario_id]
        if lora_scenario_name not in accepted_scenario_names:
            print(
                f"Error: LoRA directory '{args.lora_dir}' resolves to run '{lora_run_dir.name}' "
                f"for scenario '{lora_scenario_name}', not '{scenario_name}'"
            )
            sys.exit(1)
        output_dir = output_dir / f"{lora_run_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"full_{args.prompt_version}.jsonl"

    # Process entries in batches (DataLoader)
    print(f"Starting inference with batch_size={args.batch_size}...")
    dataset = InferenceDataset(entries, dtype=dtype_map[args.dtype])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_inference,
    )

    with open(output_path, "w") as f_out:
        f_out.write(json.dumps({
            "model": args.model,
            "scenario": scenario_id,
            "prompt_version": args.prompt_version,
        }) + "\n")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            try:
                for error_item in batch.errors:
                    entry = error_item["entry"]
                    result = {
                        "filename": Path(entry.get("image_path", "unknown")).name,
                        "cor": entry.get("cor", "unknown"),
                        "model_output": None,
                        "error": error_item.get("error"),
                    }
                    f_out.write(json.dumps(result) + "\n")

                if not batch.entries:
                    f_out.flush()
                    continue

                # Run inference via Scenario1
                outputs = scenario.run(
                    adapter=adapter,
                    images=batch.images,
                    heatmaps=batch.heatmaps,
                    cors=batch.cors,
                    use_vision_hook=not args.no_gaze,
                )

                # Save results
                for entry, output in zip(batch.entries, outputs):
                    result = {
                        "filename": Path(entry["image_path"]).name,
                        "cor": entry["cor"],
                        "model_output": output,
                    }
                    f_out.write(json.dumps(result) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                # Save error for each entry in the batch
                for entry in batch.entries:
                    result = {
                        "filename": Path(entry.get("image_path", "unknown")).name,
                        "cor": entry.get("cor", "unknown"),
                        "model_output": None,
                        "error": str(e),
                    }
                    f_out.write(json.dumps(result) + "\n")
                f_out.flush()

    print(f"\nInference complete!")
    print(f"Processed {len(entries)} entries")

    group_outputs_by_cor(output_path)
    


if __name__ == "__main__":
    main()
