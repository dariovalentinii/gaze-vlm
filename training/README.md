# Training

This directory contains the model-specific entry points for training Learnable Gaze Gating (LGG) and Dual Encoding (DE). Run every command from the repository root.

## Before you start

Install the project dependencies as described in the [main README](../README.md#installation). Training is intended for a CUDA-capable GPU; the required memory depends on the selected backbone, precision, batch size, gradient accumulation, and whether projector or heatmap-encoder LoRA is enabled.

The experiments in the paper use CapGaze for training and reserve CogBench for evaluation. No dataset is distributed in this repository. See the [data guide](../README.md#data) for the CogBench evaluation data and gaze heatmaps.

## Input data

Every trainer requires `--train_jsonl` and `--output_dir_name`. Validation data is optional and can be supplied with `--val_jsonl`.

Each JSONL record must contain:

- `image_path`: path to the training image;
- `heatmap_path`: path to a NumPy heatmap with shape `(H, W)` or `(1, 1, H, W)`;
- either `cor`, selecting a prompt from `src/data/prompts.py`, or a custom `prompt` string.

Example:

```json
{"image_path":"/path/to/image.jpg","heatmap_path":"/path/to/heatmap.npy","cor":"5_ER"}
```

Valid `cor` labels are `0_E`, `1_STR`, `2_LR`, `3_CR`, `4_CRR`, `5_ER`, `6_ERR`, `7_NMER`, and `8_MSR`. An optional `target` field is accepted for compatibility but is not used by the current training objective.

Paths are interpreted relative to the directory from which the trainer is launched unless they are absolute. Multiple training or validation JSONL files can be supplied as comma-separated paths:

```bash
--train_jsonl /path/to/train_a.jsonl,/path/to/train_b.jsonl
```

## Choose a trainer and backbone

Each family-specific trainer defaults to its 7B backbone. Pass `--model` explicitly to select the 13B variant or to make a run configuration self-documenting.

| Method | Backbone | Hugging Face model | Trainer |
| --- | --- | --- | --- |
| LGG | LLaVA 1.5 7B | `llava-hf/llava-1.5-7b-hf` | `training/lgg/train_llava_15.py` |
| LGG | LLaVA 1.5 13B | `llava-hf/llava-1.5-13b-hf` | `training/lgg/train_llava_15.py` |
| LGG | LLaVA-NeXT 7B | `llava-hf/llava-v1.6-vicuna-7b-hf` | `training/lgg/train_llava_next.py` |
| LGG | LLaVA-NeXT 13B | `llava-hf/llava-v1.6-vicuna-13b-hf` | `training/lgg/train_llava_next.py` |
| LGG | LLaVA-OneVision 7B Chat | `llava-hf/llava-onevision-qwen2-7b-ov-chat-hf` | `training/lgg/train_llava_onevision.py` |
| DE | LLaVA 1.5 7B | `llava-hf/llava-1.5-7b-hf` | `training/dual_encoding/train_llava_15.py` |
| DE | LLaVA 1.5 13B | `llava-hf/llava-1.5-13b-hf` | `training/dual_encoding/train_llava_15.py` |
| DE | LLaVA-NeXT 7B | `llava-hf/llava-v1.6-vicuna-7b-hf` | `training/dual_encoding/train_llava_next.py` |
| DE | LLaVA-NeXT 13B | `llava-hf/llava-v1.6-vicuna-13b-hf` | `training/dual_encoding/train_llava_next.py` |
| DE | LLaVA-OneVision 7B Chat | `llava-hf/llava-onevision-qwen2-7b-ov-chat-hf` | `training/dual_encoding/train_llava_onevision.py` |

The shared LGG LLaVA 1.5 trainer supports both `--attn_last_layers` and the optional `--attn_layer_start`/`--attn_layer_end` slice. The shared DE LLaVA-NeXT trainer writes periodic `step_<update>` checkpoints according to `--save_every` for both model sizes.

## Training commands

LGG trains the gaze injector. Add `--train_projector_lora` to train a LoRA adapter on the multimodal projector, as in the paper configuration:

```bash
python training/lgg/train_llava_15.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --output_dir_name my_run \
  --train_projector_lora
```

DE additionally uses a separate heatmap encoder. Add `--train_heatmap_encoder_lora` to fine-tune that encoder with LoRA:

```bash
python training/dual_encoding/train_llava_15.py \
  --model llava-hf/llava-1.5-7b-hf \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --output_dir_name my_run \
  --train_projector_lora \
  --train_heatmap_encoder_lora
```

To use another backbone, select the corresponding trainer and model identifier from the table above. Omit `--val_jsonl` when no validation split is available.

## Training objective

The loss combines three terms:

| Argument | Purpose |
| --- | --- |
| `--lambda_attn` | Aligns language-model attention over visual tokens with the gaze distribution. |
| `--lambda_distill` | Preserves the original model behavior through logit distillation. |
| `--lambda_gate` | Regularizes learned gates toward identity to limit destructive feature suppression. |

Attention alignment supports `kl`, `mse`, and `ce` through `--loss`. The default values are defined by each trainer and are shown by running:

```bash
python training/lgg/train_llava_15.py --help
```

## Common options

| Option | Description |
| --- | --- |
| `--batch_size` | Per-step batch size. |
| `--grad_accum` | Number of steps accumulated before an optimizer update. |
| `--epochs` / `--max_steps` | Training duration; a positive `--max_steps` limits the total updates. |
| `--dtype` | `float16`, `bfloat16`, or `float32`. |
| `--grad_ckpt` | Enables gradient checkpointing. |
| `--attn_last_layers` | Number of final language-model layers used for attention alignment. |
| `--attn_layer_start` / `--attn_layer_end` | Optional Python-style attention-layer slice in the LGG LLaVA 1.5 trainer. |
| `--save_every` | Number of optimizer updates between periodic DE LLaVA-NeXT checkpoints. |
| `--p_no_gaze` | Probability of disabling gaze for a batch as regularization. |
| `--init_lora_dir` | Initializes the projector LoRA from an existing adapter. |
| `--init_heatmap_lora_dir` | Initializes the DE heatmap-encoder LoRA from an existing adapter. |

Some options are model- or method-specific. Use the selected trainer's `--help` output as the authoritative reference.

## Output files

Each run is stored below its method and model directory. For example:

```text
training/lgg/llava-1.5-7b-hf/my_run/
```

The run directory contains:

| Artifact | Contents |
| --- | --- |
| `gaze_injector.pt` | Trained LGG or DE injector weights. |
| `components.json` | Saved component and vision-layer configuration used during inference. |
| `train_log.jsonl` | Training configuration, losses, and validation events. |
| `projector_lora/` | Optional projector LoRA created by `--train_projector_lora`. |
| `heatmap_encoder_lora/` | Optional DE encoder LoRA created by `--train_heatmap_encoder_lora`. |

Pass the complete run directory to inference:

```bash
python run_batch_inference.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --heatmaps_dir /path/to/heatmaps \
  --method lgg \
  --lora_dir training/lgg/llava-1.5-7b-hf/my_run
```

## Developer reference

The entry points import shared helpers; these modules are not run directly.

| Module | Responsibility |
| --- | --- |
| `training/data.py` | Training dataset, input validation, prompt resolution, and batch structures. |
| `training/modeling.py` | Model and adapter selection. |
| `training/attention_alignment.py` | LLaVA-NeXT-style attention capture used by LGG NeXT and DE NeXT/OneVision. |
| `training/attention_positions.py` | Shared image-token position inference. |
| `training/lgg/common.py` | LGG gaze targets, distillation, freezing, and heatmap preprocessing. |
| `training/lgg/llava15_attention.py` | LLaVA 1.5 attention helpers. |
| `training/lgg/attention_onevision.py` | OneVision-specific attention and RoPE handling. |
| `training/dual_encoding/encoder.py` | Heatmap encoder setup, preprocessing, and vision LoRA selection. |
| `training/dual_encoding/objectives.py` | DE distillation loss and gaze targets. |
| `training/dual_encoding/attention_llava15.py` | DE-specific LLaVA 1.5 attention alignment. |
