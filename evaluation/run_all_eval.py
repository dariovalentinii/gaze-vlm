"""
Run recognition score, cognition GPT eval, and cognition score in order.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import ROOT
from evaluation.recognition_score import main as recognition_main
from evaluation.cognition_gpt_eval import main as cognition_gpt_eval_main
from evaluation.cognition_score import main as cognition_score_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output_file_path",
        type=str,
        required=True,
        help="Path to model output jsonl",
    )
    parser.add_argument(
        "--cogbench_description_file_path",
        type=str,
        default=str(ROOT / "data" / "cogbench_v1-1" / "cogbench_v1_description.json"),
        help="CogBench description json",
    )
    parser.add_argument(
        "--entries_jsonl",
        type=str,
        default=None,
        help="Optional JSONL entries (image_path, heatmap_path, cor). Filters description json to those images.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--scores_output_dir",
        type=str,
        default=None,
        help="Optional output directory for recognition scores.json",
    )

    args = parser.parse_args()

    gemini_name = "gemini-2.5-flash"

    description_path = args.cogbench_description_file_path
    temp_desc_path = None
    if args.entries_jsonl:
        image_keys = set()
        with open(args.entries_jsonl, "r") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_path = row.get("image_path")
                if not image_path:
                    continue
                image_keys.add(Path(str(image_path)).stem)

        with open(description_path, "r") as f_desc:
            full_desc = json.load(f_desc)

        filtered_desc = {k: v for k, v in full_desc.items() if k in image_keys}

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        with tmp:
            json.dump(filtered_desc, tmp, ensure_ascii=True)
        temp_desc_path = tmp.name
        description_path = temp_desc_path

    try:
        # 1) Recognition score
        scores_file = recognition_main(
            description_path,
            args.model_output_file_path,
            args.scores_output_dir,
        )

        # 2) Cognition GPT eval
        eval_output_file = cognition_gpt_eval_main(
            description_path,
            args.model_output_file_path,
            args.scores_output_dir,
            gemini_name,
            args.debug,
        )

        # 3) Cognition score
        cognition_score_main(eval_output_file, scores_file=str(scores_file) if scores_file else None)
    finally:
        if temp_desc_path:
            try:
                Path(temp_desc_path).unlink()
            except OSError:
                print(f"Warning: failed to remove temp file: {temp_desc_path}")


if __name__ == "__main__":
    main()
