"""Prompt formatting and batching shared by LLaVA 1.5 and LLaVA-NeXT."""

from __future__ import annotations

from typing import Any, Dict, List

from training.data import Batch, resolve_prompt


def has_chat_template(processor: Any) -> bool:
    tok = getattr(processor, "tokenizer", None)
    return bool(tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None))


def build_prompt_only_text(processor: Any, prompt_text: str) -> str:
    tok = processor.tokenizer
    if has_chat_template(processor):
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        return tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    return f"USER: <image>\n{prompt_text}\nASSISTANT:"


def collate_fn(batch: List[Dict[str, Any]], processor: Any, max_length: int) -> Batch:
    images = [b["image"] for b in batch]
    heatmaps = [b["heatmap"] for b in batch]

    prompt_texts: List[str] = []
    for b in batch:
        prompt = resolve_prompt(b.get("prompt"), b.get("cor"))
        prompt_texts.append(build_prompt_only_text(processor, prompt))

    # For chat-template tokenizers, processor usually expects add_special_tokens=False.
    used_chat = has_chat_template(processor)
    add_special_tokens = False if used_chat else True

    model_inputs = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )

    # No labels: we optimize only attention-alignment loss.
    return Batch(inputs=model_inputs, heatmaps=heatmaps)
