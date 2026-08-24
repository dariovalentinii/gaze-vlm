#!/usr/bin/env python3
"""
Run baseline, LGG, or Dual Encoding inference and then evaluate it.

- Calls run_batch_inference (your script) to produce:
    results/<scenario>/<model_dir>/consolidated.jsonl
- Then calls run_all_eval.py using that consolidated.jsonl as --model_output_file_path

Usage:
  python scripts/infer_and_eval.py --model llava-hf/llava-1.5-7b-hf --no_gaze

Optional:
  --output_dir results
  --no_gaze
  --batch_size 9
  --device cuda
  --dtype float16
  --debug
  --scores_output_dir <path>
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import ROOT

def _run(cmd: list[str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _model_dir_name(model: str) -> str:
    return model.split("/")[-1]


def _find_consolidated(output_dir: Path, model: str, scenario_name_hint: str | None = None, prompt_version: str = None) -> Path:
    """
    Find consolidated.jsonl produced by inference.
    If scenario_name_hint is given, prefer that path; otherwise pick the newest match.
    """
    model_dir = _model_dir_name(model)

    output_name = f"consolidated_{prompt_version}.jsonl" if prompt_version else "consolidated.jsonl"

    # Preferred expected location, including a possible checkpoint run subdirectory.
    if scenario_name_hint:
        scenario_root = output_dir / scenario_name_hint / model_dir
        matches = list(scenario_root.glob(f"**/{output_name}"))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

    # Fallback: search
    matches = list(output_dir.glob(f"**/{model_dir}/**/{output_name}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find consolidated.jsonl under output_dir={output_dir} for model={model!r}."
        )

    # Pick newest modified
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name (same as run_batch_inference expects)")

    ap.add_argument(
        "--inference_script",
        type=str,
        default="run_batch_inference.py",
        help="Path to run_batch_inference script (default: run_batch_inference.py)",
    )
    ap.add_argument(
        "--eval_script",
        type=str,
        default="evaluation/run_all_eval.py",
        help="Path to the evaluation script (default: evaluation/run_all_eval.py)",
    )

    # Forwarded inference args (keep aligned with run_batch_inference)
    ap.add_argument("--images_dir", type=str, default=None)
    ap.add_argument("--heatmaps_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="results")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=9)
    ap.add_argument("--no_gaze", action="store_true")
    ap.add_argument("--scenario", type=int, choices=[2, 3], default=None)
    ap.add_argument("--lora_dir", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--prompt_version", type=str, default="v2", choices=["v1", "v2"],
                    help="Prompt set version to use (v1 or v2)")

    # Forwarded eval args (run_all_eval.py)
    ap.add_argument("--scores_output_dir", type=str, default=None, help="Optional output dir for scores.json")

    args = ap.parse_args()

    if args.no_gaze == (args.scenario is not None):
        ap.error("select exactly one of --no_gaze or --scenario")
    if args.scenario is not None and not args.lora_dir:
        ap.error("--lora_dir is required for LGG and Dual Encoding")

    here = _PROJECT_ROOT
    inference_script = (here / args.inference_script).resolve()
    eval_script = (here / args.eval_script).resolve()

    if not inference_script.exists():
        raise FileNotFoundError(f"Inference script not found: {inference_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    output_dir = Path(args.output_dir).resolve()

    # 1) Run inference
    infer_cmd = [sys.executable, str(inference_script), "--model", args.model, "--output_dir", str(output_dir)]
    if args.images_dir:
        infer_cmd += ["--images_dir", args.images_dir]
    if args.heatmaps_dir:
        infer_cmd += ["--heatmaps_dir", args.heatmaps_dir]

    infer_cmd += ["--device", args.device, "--dtype", args.dtype]
    infer_cmd += ["--max_new_tokens", str(args.max_new_tokens)]
    infer_cmd += ["--temperature", str(args.temperature)]
    infer_cmd += ["--batch_size", str(args.batch_size)]

    if args.do_sample:
        infer_cmd += ["--do_sample"]
    if args.no_gaze:
        infer_cmd += ["--no_gaze"]
    if args.scenario is not None:
        infer_cmd += ["--scenario", str(args.scenario), "--lora_dir", args.lora_dir]
    if args.debug:
        infer_cmd += ["--debug"]
    if args.prompt_version:
        infer_cmd += ["--prompt_version", args.prompt_version]

    _run(infer_cmd)

    # Small delay to ensure filesystem timestamps settle on some systems
    time.sleep(0.2)

    scenario_hint = "baseline" if args.no_gaze else {2: "lgg", 3: "dual_encoding"}[args.scenario]

    consolidated = _find_consolidated(output_dir=output_dir, model=args.model, scenario_name_hint=scenario_hint, prompt_version=args.prompt_version)
    print(f"\n[INFO] Using consolidated output: {consolidated}")

    # 2) Run eval on consolidated.jsonl
    eval_cmd = [sys.executable, str(eval_script), "--model_output_file_path", str(consolidated)]
    if args.debug:
        eval_cmd += ["--debug"]
    if args.scores_output_dir:
        eval_cmd += ["--scores_output_dir", args.scores_output_dir]

    _run(eval_cmd)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
