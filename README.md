# Gaze-VLM

Gaze-conditioned inference and fine-tuning for CogBench with LLaVA-family vision-language models.

This repository contains three supported methods:

- **Baseline**: standard inference without gaze injection.
- **Learnable Gaze Gating (LGG)**: a learned affine-sigmoid gate weights visual patch features from gaze heatmaps.
- **Dual Encoding (DE)**: a separate vision encoder embeds the heatmap before gaze gating.

The refactor preserves the existing model computations, training losses, adapter behavior, and checkpoint contents. The public inference entry point selects gaze methods by name: `--method lgg` or `--method de`.

## Which file should I run?

| Goal | Entry point |
| --- | --- |
| Run Baseline, LGG, or DE inference | `run_batch_inference.py` |
| Train LGG | A model-specific script in `training/lgg/` |
| Train DE | A model-specific script in `training/dual_encoding/` |
| Evaluate an existing inference JSONL | `evaluation/run_all_eval.py` |
| Run inference and evaluation together | `scripts/infer_and_eval.py` |

See the [training guide](training/README.md) to choose a trainer and the [evaluation guide](evaluation/README.md) for the complete scoring pipeline.

### Inference and evaluation flow

```text
run_batch_inference.py
    -> src/models/adapter_factory.py
    -> model-family adapter in src/models/
    -> src/inference/runner.py
    -> results/<method>/<model>/.../consolidated_<prompt_version>.jsonl
    -> evaluation/run_all_eval.py
```

LGG and DE training produce a run directory that is passed to inference with `--lora_dir`:

```text
training/lgg/ or training/dual_encoding/
    -> <model>/<run_name>/
    -> run_batch_inference.py --lora_dir <model>/<run_name>/
```

## Repository layout

```text
.
├── run_batch_inference.py       # baseline, LGG, and DE inference
├── src/
│   ├── data/                    # prompts and heatmap processing
│   ├── models/
│   │   ├── llava_*.py           # model-family inference adapters
│   │   ├── checkpoint_utils.py  # shared checkpoint loading helpers
│   │   └── unwrapping.py        # access through optional model wrappers
│   └── inference/
│       ├── data.py              # inference entries, dataset, and batching
│       ├── results.py           # JSONL consolidation
│       └── runner.py            # shared inference protocol and runner
├── training/
│   ├── data.py                  # shared training dataset and input structures
│   ├── lgg/                     # LGG trainers, attention, and capture helpers
│   ├── dual_encoding/           # DE trainers, encoder, and objectives
│   ├── attention_alignment.py   # shared LLaVA-NeXT-style attention logic
│   ├── attention_positions.py   # shared image-token position helper
│   ├── modeling.py              # shared model/adapter selection
│   ├── prompting_llava.py       # LLaVA 1.5/NeXT prompt formatting
│   └── prompting_onevision.py   # OneVision prompt formatting
├── evaluation/                  # CogBench evaluation pipeline
└── scripts/                     # convenience entry points
```

Datasets, experiment outputs, logs, adapters, and checkpoints are intentionally not versioned.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The evaluation pipeline uses Gemini for cognition scoring. Provide the key through the environment rather than source code:

```bash
export GEMINI_API_KEY="..."
```

The project root is detected automatically. `GAZE_VLM_ROOT` can override it when data is stored relative to another root.

## Data

Inference expects CogBench images and NumPy heatmaps. The default layout is:

```text
data/
├── cogbench_v1-1/
│   ├── images/
│   └── cogbench_v1_description.json
└── heatmaps/
    └── avg/
```

Training consumes JSONL records with `image_path`, `heatmap_path`, and either `prompt` or `cor`:

```json
{"image_path":"/path/image.jpg","heatmap_path":"/path/heatmap.npy","cor":"0_E"}
```

Multiple training or validation files can be passed as comma-separated paths.

## Inference

Baseline:

```bash
python run_batch_inference.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --heatmaps_dir /path/to/heatmaps \
  --no_gaze
```

For parity with the original implementation, baseline data loading still expects heatmap files even though the vision hook is disabled.

LGG:

```bash
python run_batch_inference.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --heatmaps_dir /path/to/heatmaps \
  --method lgg \
  --lora_dir training/lgg/llava-1.5-7b-hf/my_run
```

Dual Encoding:

```bash
python run_batch_inference.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --heatmaps_dir /path/to/heatmaps \
  --method de \
  --lora_dir training/dual_encoding/llava-1.5-7b-hf/my_run
```

Checkpoints stored in earlier directory layouts remain accepted.

Outputs are written below `results/baseline`, `results/lgg`, or `results/dual_encoding`. Each run first writes `full_<prompt_version>.jsonl` and then consolidates it into `consolidated_<prompt_version>.jsonl`.

## Training

For trainer selection, inputs, outputs, and minimal commands, see [training/README.md](training/README.md).

Choose the trainer that matches the method and model family:

| Method | Model family | Script |
| --- | --- | --- |
| LGG | LLaVA 1.5 7B | `training/lgg/train_llava_15_7b.py` |
| LGG | LLaVA 1.5 13B | `training/lgg/train_llava_15_13b.py` |
| LGG | LLaVA-NeXT 7B/13B | `training/lgg/train_llava_next.py` |
| LGG | LLaVA-NeXT hook variant | `training/lgg/train_llava_next_with_hooks.py` |
| LGG | LLaVA-OneVision | `training/lgg/train_llava_onevision.py` |
| DE | LLaVA 1.5 7B/13B | `training/dual_encoding/train_llava_15.py` |
| DE | LLaVA-NeXT 7B | `training/dual_encoding/train_llava_next_7b.py` |
| DE | LLaVA-NeXT 13B | `training/dual_encoding/train_llava_next_13b.py` |
| DE | LLaVA-OneVision | `training/dual_encoding/train_llava_onevision.py` |

Example LGG training command:

```bash
python training/lgg/train_llava_15_7b.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --output_dir_name my_run \
  --train_projector_lora \
  --batch_size 1 \
  --grad_accum 16 \
  --epochs 1
```

Example Dual Encoding training command:

```bash
python training/dual_encoding/train_llava_15.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --output_dir_name my_run \
  --train_projector_lora \
  --train_heatmap_encoder_lora \
  --batch_size 1 \
  --grad_accum 16 \
  --epochs 1
```

Training artifacts are stored below the selected method and model, for example `training/lgg/llava-1.5-7b-hf/my_run`.

## Evaluation

For the execution order, input/output files, and Gemini configuration, see [evaluation/README.md](evaluation/README.md).

```bash
python evaluation/run_all_eval.py \
  --model_output_file_path results/lgg/llava-1.5-7b-hf/my_run/consolidated_v2.jsonl \
  --cogbench_description_file_path /path/to/cogbench_v1_description.json
```

No license has been selected yet. Add one before making the repository public.
