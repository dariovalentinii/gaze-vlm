# Evaluation

`evaluation/run_all_eval.py` is the main evaluation entry point. It reads a consolidated inference JSONL, computes entity-recognition recall, evaluates the reasoning outputs with Gemini, and writes the final recognition and cognition scores.

## Before you start

Install the dependencies and spaCy model described in the [main README](../README.md#installation). Evaluation also requires the CogBench description JSON; follow the [data instructions](../README.md#data) to obtain it from CogBench.

On its first use, `recognition_score.py` may download the `sentence-transformers/all-mpnet-base-v2` model. Cognition evaluation requires a Gemini API key:

```bash
export GEMINI_API_KEY="..."
```

The complete pipeline uses `gemini-2.5-flash`.

## Run the complete evaluation

```bash
python evaluation/run_all_eval.py \
  --model_output_file_path results/lgg/llava-1.5-7b-hf/my_run/consolidated_v2.jsonl \
  --cogbench_description_file_path /path/to/cogbench_v1_description.json
```

The three steps run in this order:

```text
consolidated_<prompt_version>.jsonl
    -> recognition_score.py
    -> cognition_gpt_eval.py (Gemini 2.5 Flash)
    -> cognition_score.py
    -> scores_<prompt_version>.json
```

The individual scripts remain available, but `run_all_eval.py` should be used for a complete evaluation because it passes each result to the following step automatically.

## Consolidated input format

The first JSONL row contains run metadata:

```json
{"model":"llava-hf/llava-1.5-7b-hf","method":"lgg","prompt_version":"v2"}
```

Each following row contains one image and the outputs consolidated across the nine CogBench prompts:

```json
{"filename":"cogbench_v1_1.jpg","entities_output":"person; bicycle","special_time_reasoning_output":"None","location_reasoning_output":"[A visible school sign suggests the setting -> The scene is near a school]","character_reasoning_output":"None","character_relationship_reasoning_output":"None","event_reasoning_output":"None","event_relationship_reasoning_output":"None","next_moment_event_reasoning_output":"None","mental_state_reasoning_output":"None"}
```

`run_batch_inference.py` creates this format automatically as `consolidated_<prompt_version>.jsonl`. The expected output fields are:

| CogBench label | JSONL field |
| --- | --- |
| `0_E` | `entities_output` |
| `1_STR` | `special_time_reasoning_output` |
| `2_LR` | `location_reasoning_output` |
| `3_CR` | `character_reasoning_output` |
| `4_CRR` | `character_relationship_reasoning_output` |
| `5_ER` | `event_reasoning_output` |
| `6_ERR` | `event_relationship_reasoning_output` |
| `7_NMER` | `next_moment_event_reasoning_output` |
| `8_MSR` | `mental_state_reasoning_output` |

## Arguments

| Argument | Description |
| --- | --- |
| `--model_output_file_path` | Required consolidated JSONL produced by inference. |
| `--cogbench_description_file_path` | CogBench description JSON; defaults to `data/cogbench_v1-1/cogbench_v1_description.json`. |
| `--entries_jsonl` | Optional JSONL used to restrict evaluation to selected images. Only `image_path` is read from each record. |
| `--scores_output_dir` | Optional directory for all evaluation outputs; defaults to the model-output directory. |
| `--debug` | Prints model outputs, key points, prompts, and Gemini parsing details. |

When `--entries_jsonl` is supplied, the CogBench annotations are filtered using the stem of each `image_path`. Heatmap and `cor` fields are not required for this filtering operation.

## Metrics

### Recognition

Recognition is evaluated against the CogBench entity annotations. Nouns extracted from `entities_output` with spaCy are matched semantically using `all-mpnet-base-v2` and a cosine-similarity threshold of `0.6`.

The output contains:

- `macro_avg_recall`: mean entity recall across evaluated images;
- `micro_avg_recall`: recall aggregated across all annotated entities.

These are recall metrics and therefore do not penalize additional hallucinated entities.

### Cognition

Gemini compares each non-empty reasoning output with the CogBench key points and assigns binary coverage judgments. Scores are averaged separately for:

- Special Time Reasoning;
- Location Reasoning;
- Character Reasoning;
- Character Relationship Reasoning;
- Event Reasoning;
- Event Relationship Reasoning;
- Next Moment Event Reasoning;
- Mental State Reasoning.

The final `overall` score averages the judgments across all eight dimensions. Empty or `None` reasoning outputs receive zero without an API call. Null judgments stop final aggregation instead of being silently included.

## Gemini execution notes

Gemini is called once for each image-dimension pair that has both annotated conclusions and a non-empty model output. Transient API or parsing failures are retried up to five times; non-recoverable API failures are recorded as zero.

`cognition_gpt_eval_<prompt_version>.jsonl` is recreated when cognition evaluation starts. An interrupted run is not resumed automatically, so rerunning the pipeline restarts the Gemini evaluation and may repeat API calls.

The evaluator uses two checked-in system prompts:

| File | Purpose |
| --- | --- |
| `evaluation/system/eval_system_prompt_v2.txt` | General key-point coverage evaluation. |
| `evaluation/system/eval_system_prompt_er_v2.txt` | Event Relationship-specific evaluation. |

## Output files

By default, outputs are written beside the consolidated input:

| Output | Contents |
| --- | --- |
| `cognition_gpt_eval_<prompt_version>.jsonl` | Metadata followed by per-image Gemini judgments. |
| `scores_<prompt_version>.json` | Recognition metrics, cognition scores by dimension, and overall cognition score. |

If the input metadata does not contain `prompt_version`, the filenames are `cognition_gpt_eval.jsonl` and `scores.json`.

A final score file has this structure:

```json
{
  "model": "llava-hf/llava-1.5-7b-hf",
  "method": "lgg",
  "prompt_version": "v2",
  "recognition": {
    "macro_avg_recall": 0.0,
    "micro_avg_recall": 0.0
  },
  "cognition": {
    "Special Time Reasoning": 0.0,
    "Location Reasoning": 0.0,
    "Character Reasoning": 0.0,
    "Character Relationship Reasoning": 0.0,
    "Event Reasoning": 0.0,
    "Event Relationship Reasoning": 0.0,
    "Next Moment Event Reasoning": 0.0,
    "Mental State Reasoning": 0.0,
    "overall": 0.0
  }
}
```

The zero values above illustrate the schema and are not expected results.

## Inference and evaluation in one command

`scripts/infer_and_eval.py` runs inference, locates the resulting consolidated JSONL, and passes it to the evaluation pipeline. Select exactly one of `--no_gaze` or `--method`.

Baseline does not require heatmaps:

```bash
python scripts/infer_and_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --no_gaze
```

LGG and DE require both gaze heatmaps and a trained run directory:

```bash
python scripts/infer_and_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --images_dir /path/to/images \
  --heatmaps_dir /path/to/heatmaps \
  --method lgg \
  --lora_dir training/lgg/llava-1.5-7b-hf/my_run
```
