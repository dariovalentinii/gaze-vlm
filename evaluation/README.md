# Evaluation

`evaluation/run_all_eval.py` is the main evaluation entry point. It evaluates an existing consolidated inference JSONL and runs the three scoring steps in the required order.

## Pipeline

```text
consolidated_<prompt_version>.jsonl
    -> recognition_score.py
    -> cognition_gpt_eval.py (Gemini)
    -> cognition_score.py
    -> scores_<prompt_version>.json
```

The individual scripts remain available, but `run_all_eval.py` should be used for a complete evaluation because it passes their outputs to the next step automatically.

## Inputs

| Argument | Description |
| --- | --- |
| `--model_output_file_path` | Required consolidated JSONL produced by inference |
| `--cogbench_description_file_path` | CogBench description JSON; defaults to `data/cogbench_v1-1/cogbench_v1_description.json` |
| `--entries_jsonl` | Optional training-style JSONL used to evaluate only the listed image stems |
| `--scores_output_dir` | Optional directory for evaluation outputs; defaults to the model output directory |
| `--debug` | Prints additional Gemini evaluation details |

The consolidated JSONL must include its metadata row and the per-image outputs produced by `run_batch_inference.py`.

## Gemini configuration

Cognition evaluation uses `gemini-2.5-flash`. Set the API key in the environment before starting:

```bash
export GEMINI_API_KEY="..."
```

Gemini is called once for each non-empty reasoning output. The recognition step does not use Gemini.

## Run the complete evaluation

```bash
python evaluation/run_all_eval.py \
  --model_output_file_path results/lgg/llava-1.5-7b-hf/my_run/consolidated_v2.jsonl \
  --cogbench_description_file_path /path/to/cogbench_v1_description.json
```

By default, the pipeline writes beside the consolidated input:

| Output | Contents |
| --- | --- |
| `cognition_gpt_eval_<prompt_version>.jsonl` | Per-image Gemini judgments |
| `scores_<prompt_version>.json` | Recognition and cognition scores |

If the input metadata does not contain a prompt version, the filenames are `cognition_gpt_eval.jsonl` and `scores.json`.

## Inference and evaluation in one command

`scripts/infer_and_eval.py` runs inference first, locates its consolidated output, and passes it to the evaluation pipeline. Select exactly one of `--no_gaze` or `--method`.

Baseline:

```bash
python scripts/infer_and_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --no_gaze
```

LGG or DE also requires a trained run directory:

```bash
python scripts/infer_and_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --method lgg \
  --lora_dir training/lgg/llava-1.5-7b-hf/my_run
```
