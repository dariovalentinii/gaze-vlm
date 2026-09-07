# Training

Training uses direct Python entry points, one for each supported method and model family. Run commands from the repository root.

## Choose a trainer

| Method | Model family | Trainer |
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

The shared helpers in `training/` are imported by these entry points and are not run directly.

## Attention helper layout

| Module | Used for |
| --- | --- |
| `training/attention_alignment.py` | Shared LLaVA-NeXT-style attention capture used by LGG NeXT and DE NeXT/OneVision |
| `training/attention_positions.py` | Shared image-token position helper |
| `training/lgg/attention_capture.py` | Last-layer attention capture used by the LGG NeXT hook variant |
| `training/lgg/llava15_attention.py` | LGG LLaVA 1.5 and hook-variant attention helpers |
| `training/lgg/attention_onevision.py` | LGG OneVision-specific attention and RoPE handling |
| `training/dual_encoding/attention_llava15.py` | DE LLaVA 1.5-specific attention alignment |

The family-specific implementations remain separate even when some functions look similar, because their module-level dependencies and token-layout assumptions can differ.

## Other shared modules

| Module | Contents |
| --- | --- |
| `training/data.py` | Training dataset, batch structure, prompt resolution, and tokenizer padding |
| `src/models/unwrapping.py` | Access to the underlying LLaVA model through optional wrappers |
| `training/dual_encoding/encoder.py` | Heatmap encoder setup and preprocessing |
| `training/dual_encoding/objectives.py` | Dual Encoding distillation loss and gaze targets |

## Input data

Every trainer requires `--train_jsonl` and `--output_dir_name`. Validation data is optional and is passed with `--val_jsonl`.

Each JSONL record must contain `image_path`, `heatmap_path`, and either `prompt` or `cor`:

```json
{"image_path":"/path/image.jpg","heatmap_path":"/path/heatmap.npy","cor":"0_E"}
```

Multiple training or validation files can be supplied as comma-separated paths.

## Minimal commands

LGG with LLaVA 1.5 7B:

```bash
python training/lgg/train_llava_15_7b.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --output_dir_name my_run
```

Dual Encoding with LLaVA 1.5 7B:

```bash
python training/dual_encoding/train_llava_15.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --output_dir_name my_run
```

Add `--val_jsonl /path/to/val.jsonl` when a validation set is available. Use `python <trainer> --help` for model-specific options such as batch size, gradient accumulation, LoRA training, and initialization from an existing adapter.

## Output files

Each run is stored below the selected method and model, for example:

```text
training/lgg/llava-1.5-7b-hf/my_run/
```

The run directory contains:

| Artifact | Contents |
| --- | --- |
| `gaze_injector.pt` | Trained gaze injector weights |
| `components.json` | Components and vision-layer metadata needed at inference time |
| `train_log.jsonl` | Training configuration, losses, and validation events |
| `projector_lora/` | Optional projector LoRA, created with `--train_projector_lora` |
| `heatmap_encoder_lora/` | Optional DE heatmap-encoder LoRA, created with `--train_heatmap_encoder_lora` |

Pass the complete run directory to inference:

```bash
python run_batch_inference.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --heatmaps_dir /path/to/heatmaps \
  --method lgg \
  --lora_dir training/lgg/llava-1.5-7b-hf/my_run
```

Cross-validation and experimental direct-weighting variants are outside the current public scope.
