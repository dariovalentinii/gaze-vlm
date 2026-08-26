"""Consolidation helpers for inference results."""

import json
import os
from pathlib import Path


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
