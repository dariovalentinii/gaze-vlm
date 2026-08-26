import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.constants import (
    ROOT,
    STRICT_LLAVA_NEXT_BATCH,
)
from src.inference.data import (
    InferenceBatch,
    InferenceDataset,
    build_entries,
    collate_inference,
    load_entries_from_jsonl,
    normalize_inference_inputs,
)
from src.inference.results import group_outputs_by_cor
from src.models.adapter_factory import create_inference_adapter
from src.models.llava_next import LlavaNextHFAdapter
from src.inference.runner import GenerationConfig, InferenceRunner


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA/checkpoint artifacts (required for LGG/DE).")

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
        "--method",
        choices=["lgg", "de"],
        default=None,
        help="Gaze method: Learnable Gaze Gating (lgg) or Dual Encoding (de).",
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

    # Select exactly one inference method.
    if args.method is None and not args.no_gaze:
        print("Error: --no_gaze must be set when --method is not specified")
        sys.exit(1)
    if args.method is not None and args.no_gaze:
        print("Error: --method and --no_gaze are mutually exclusive")
        sys.exit(1)
    if args.method is not None and not args.lora_dir:
        print("Error: --lora_dir is required for LGG or DE")
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
    adapter = create_inference_adapter(
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
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        debug=args.debug,
        prompt_version=args.prompt_version,
    )

    runner = InferenceRunner(cfg=cfg)
    if args.no_gaze:
        method_name = "baseline"
    else:
        method_name = {"lgg": "lgg", "de": "dual_encoding"}[args.method]



    # Prepare output file
    output_root = Path(args.output_dir).resolve()
    model_name = args.model
    model_dir_name = model_name.split("/")[-1] if "/" in model_name else model_name
    output_dir = output_root / method_name / model_dir_name
    # Safety check on model and method for the current checkpoint layout.
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
        checkpoint_method_name = lora_model.parent.name
        known_method_names = {"lgg", "dual_encoding"}
        if checkpoint_method_name in known_method_names and checkpoint_method_name != method_name:
            print(
                f"Error: LoRA directory '{args.lora_dir}' resolves to run '{lora_run_dir.name}' "
                f"for method '{checkpoint_method_name}', not '{method_name}'"
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
            "method": method_name,
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

                # Run inference via InferenceRunner
                outputs = runner.run(
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
