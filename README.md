# Gaze-VLM

Code accompanying the paper **“Impact of Architecture and Integration Strategy on Gaze-Augmented Visual Reasoning in VLMs”** by Dario Valentini, Matteo Moro, Vittorio Murino, and Lucia Schiatti.

Gaze-VLM investigates how task-dependent human gaze can be integrated into open-source LLaVA-family vision-language models. The repository supports training, inference, and CogBench evaluation for a no-gaze baseline and two gaze-integration methods. The paper evaluates whether their effect depends on the model architecture and on the distinction between visual recognition and higher-level reasoning.

## Methods

| Method | Gaze integration |
| --- | --- |
| **Baseline** | Runs the original model without gaze injection. |
| **Learnable Gaze Gating (LGG)** | Resizes each heatmap to the visual patch grid and learns an affine-sigmoid gate that modulates the image patch embeddings. |
| **Dual Encoding (DE)** | Encodes the heatmap with a separate vision encoder and maps its patch embeddings to learned gates for the image features. |

The experiments cover LLaVA 1.5 (7B and 13B), LLaVA-NeXT/LLaVA 1.6 (7B and 13B), and LLaVA-OneVision (7B Chat). The current public workflow focuses on Baseline, LGG, and DE.

The paper finds that gaze primarily affects higher-level reasoning rather than entity recognition, and that the most effective integration strategy is architecture-dependent: LGG is better suited to the evaluated LLaVA 1.5 and LLaVA-NeXT models, while DE is more effective for LLaVA-OneVision cognition.

## Which file should I run?

| Goal | Entry point |
| --- | --- |
| Run Baseline, LGG, or DE inference | `run_batch_inference.py` |
| Train LGG | A model-specific script in `training/lgg/` |
| Train DE | A model-specific script in `training/dual_encoding/` |
| Evaluate an inference JSONL | `evaluation/run_all_eval.py` |
| Run inference and evaluation together | `scripts/infer_and_eval.py` |

See the [training guide](training/README.md) to select a trainer and the [evaluation guide](evaluation/README.md) for the complete scoring pipeline.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The cognition evaluation uses Gemini 2.5 Flash. Provide the API key through the environment rather than source code:

```bash
export GEMINI_API_KEY="..."
```

The project root is detected automatically. Set `GAZE_VLM_ROOT` only if the data directories should be resolved relative to a different root.

## Data

The datasets are not redistributed in this repository.

### CogBench images and annotations

Obtain the CogBench images and annotations directly from the [official CogBench repository](https://github.com/X-LANCE/CogBench) by following its data-access instructions. CogBench requires users to accept its Data Use Agreement; this repository does not mirror or replace the original dataset distribution.

### Gaze heatmaps

The task-dependent CogBench gaze heatmaps used by this project are available in this [Google Drive folder](https://drive.google.com/drive/u/0/folders/1HkC8yg4Ev7NnAkOedp8Yk73RHv7hQSi9). They were collected in an eye-tracking study with 30 participants and averaged across three observers for each image-task pair.

Place the downloaded files in the following default layout, or pass custom paths through the command-line arguments:

```text
data/
├── cogbench_v1-1/
│   ├── images/
│   │   ├── cogbench_v1_1.jpg
│   │   └── ...
│   └── cogbench_v1_description.json
└── heatmaps/
    └── avg/
        ├── cogbench_v1_1_0_E.npy
        └── ...
```

Each image has nine task-dependent heatmaps. Their filename labels correspond to the following CogBench dimensions:

| Label | Dimension |
| --- | --- |
| `0_E` | Entity recognition |
| `1_STR` | Special Time |
| `2_LR` | Location |
| `3_CR` | Character |
| `4_CRR` | Character Relationship |
| `5_ER` | Event |
| `6_ERR` | Event Relationship |
| `7_NMER` | Next Moment Event |
| `8_MSR` | Mental State |

The experiments in the paper use CapGaze for training and reserve CogBench for evaluation. Training scripts consume JSONL records with `image_path`, `heatmap_path`, and either `prompt` or `cor`:

```json
{"image_path":"/path/to/image.jpg","heatmap_path":"/path/to/heatmap.npy","cor":"0_E"}
```

Multiple training or validation files can be passed as comma-separated paths. Dataset files, experiment outputs, logs, adapters, and checkpoints are intentionally excluded from version control.

## Inference

Run commands from the repository root. Select exactly one of `--no_gaze` or `--method {lgg,de}`.

Baseline:

```bash
python run_batch_inference.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --no_gaze
```

Baseline inference does not load or require gaze heatmaps. It still runs the nine CogBench prompts for each image because the `cor` labels select the evaluation dimension independently of gaze.

When `--entries_jsonl` is used for Baseline, each record only needs `image_path` and `cor`; `heatmap_path` remains required for LGG and DE.

Learnable Gaze Gating:

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

Earlier compatible checkpoint layouts remain accepted. Outputs are written below `results/baseline`, `results/lgg`, or `results/dual_encoding`. Each run first writes `full_<prompt_version>.jsonl` and then produces `consolidated_<prompt_version>.jsonl`.

## Training

The [training guide](training/README.md) documents input records, generated artifacts, model-specific options, and minimal commands.

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

Minimal LGG example:

```bash
python training/lgg/train_llava_15_7b.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --output_dir_name my_run
```

Minimal DE example:

```bash
python training/dual_encoding/train_llava_15.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --output_dir_name my_run
```

Each run is stored below the selected method and model, for example `training/lgg/llava-1.5-7b-hf/my_run`. Pass that complete directory to inference with `--lora_dir`.

## Evaluation

The evaluation follows CogBench's description task and reports recognition and dimension-specific cognition scores. `evaluation/run_all_eval.py` applies the scoring steps in the required order:

```text
consolidated_<prompt_version>.jsonl
    -> recognition_score.py
    -> cognition_gpt_eval.py (Gemini 2.5 Flash)
    -> cognition_score.py
    -> scores_<prompt_version>.json
```

Run the complete evaluation with:

```bash
python evaluation/run_all_eval.py \
  --model_output_file_path results/lgg/llava-1.5-7b-hf/my_run/consolidated_v2.jsonl \
  --cogbench_description_file_path /path/to/cogbench_v1_description.json
```

See [evaluation/README.md](evaluation/README.md) for the expected input format, generated files, and combined inference/evaluation command.

## Repository layout

```text
.
├── run_batch_inference.py       # Baseline, LGG, and DE inference
├── src/
│   ├── data/                    # prompts and heatmap processing
│   ├── inference/               # datasets, batching, runner, and results
│   └── models/                  # model-family inference adapters
├── training/
│   ├── lgg/                     # LGG trainers and attention helpers
│   ├── dual_encoding/           # DE trainers, encoder, and objectives
│   └── *.py                     # shared training utilities
├── evaluation/                  # CogBench evaluation pipeline
└── scripts/                     # convenience entry points
```

### Execution flow

```text
run_batch_inference.py
    -> src/models/adapter_factory.py
    -> model-family adapter in src/models/
    -> src/inference/runner.py
    -> consolidated inference JSONL
    -> evaluation/run_all_eval.py
    -> recognition and cognition scores
```

## Citation

If this code contributes to your research, please cite:

```bibtex
@misc{valentini2026gazeaugmented,
  title  = {Impact of Architecture and Integration Strategy on Gaze-Augmented Visual Reasoning in VLMs},
  author = {Valentini, Dario and Moro, Matteo and Murino, Vittorio and Schiatti, Lucia},
  year   = {2026}
}
```

The citation will be updated when a public paper record is available.

## Acknowledgements

This project builds on CogBench and the LLaVA model family. Please follow the original projects' licenses, data-use conditions, and citation requirements.

## License

A software license has not yet been added. Until a license is selected, making the source public does not grant permission to reuse, modify, or redistribute it. Add the intended license before announcing the public release.
