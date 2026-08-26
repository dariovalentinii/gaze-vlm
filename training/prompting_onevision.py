"""Prompt formatting and batching for LLaVA-OneVision."""

from __future__ import annotations

from typing import Any, Dict, List

from training.common import Batch, resolve_prompt


def has_chat_template(processor: Any) -> bool:
    has_proc_template = hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None)
    tok = getattr(processor, "tokenizer", None)
    has_tok_template = bool(
        tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)
    )
    return bool(has_proc_template or has_tok_template)


def build_prompt_only_text(processor: Any, prompt_text: str) -> str:
    has_proc_template = hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None)
    has_tok_template = (
        hasattr(processor, "tokenizer")
        and hasattr(processor.tokenizer, "apply_chat_template")
        and getattr(processor.tokenizer, "chat_template", None)
    )

    if has_proc_template:
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        return processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)

    if has_tok_template:
        # Some tokenizer chat templates (e.g. Qwen) expect content as a plain string.
        user = [{"role": "user", "content": f"<image>\n{prompt_text}"}]
        return processor.tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)

    return f"user: <image>\n{prompt_text}\nassistant\n"


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
