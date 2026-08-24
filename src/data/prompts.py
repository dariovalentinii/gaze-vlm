from __future__ import annotations
from typing import Dict, List

# v1
NONE_NOTE_V1 = " Write None, if the picture does not involve this specific kind of reasoning or it is not important in the picture."

PROMPTS_V1: Dict[str, str] = {
    "0_E": "Please list the entities appearing in the picture, including people, animals, and objects. List the entities you can clearly see. Mention each entity at most once (no duplicates). Stop as soon as you have covered all salient entities. Do not guess.",
    "1_STR": "Please write your reasoning processes about the special time of the story in the picture, e.g. festivals, seasons etc. The special time is usually relevant to the story of the picture. For instance, if it is daytime in a picture, it is easily recognized, requires no reasoning and there is nothing special, you can write None. However, if there is a lamp on or a clock indicating a specific time, you can write down your reasoning about the time." + NONE_NOTE_V1,
    "2_LR": "Please write your reasoning processes about the location of the story in the picture, e.g. near the school." + NONE_NOTE_V1,
    "3_CR": "Please write your reasoning processes about the characters of the subjects in the picture, e.g. a teacher, a doctor etc." + NONE_NOTE_V1,
    "4_CRR": "Please write your reasoning processes about the relationships between the characters in the picture, e.g. mother-son relationship." + NONE_NOTE_V1,
    "5_ER": "Please write your reasoning processes about the events in the current and previous moments in the picture based on the clues provided. Note that you only need to annotate those high-level events and can ignore the low-level ones. For instance, \"the woman is looking at the man\" is a low-level event and you can ignore its reasoning process. Differently, the reasoning process [A mother is busy cooking. + A boy is fetching cookies behind the mom. + A girl is shushing the boy. -> The boy is stealing cookies.] is about a high-level event \"stealing\" and you should write it down." + NONE_NOTE_V1,
    "6_ERR": "Please write your reasoning processes about the relationships between different events in the picture. These events are usually linked through causal and temporal relations. Note that events in this part do not necessarily appears in the [Event Reasoning] part as some events here are low-level events." + NONE_NOTE_V1,
    "7_NMER": "Please write your reasoning processes about the events that will happen in the next moment. Note that you only need to write down events that have a very high probability of happening, instead of guessing what might happen next." + NONE_NOTE_V1,
    "8_MSR": "Please write your reasoning processes about the mental states of the subjects in the picture, e.g. daydreaming, happy, etc. You need to reason as best you can about the mental states of all the subjects in the picture, unless they are not showing obvious emotions." + NONE_NOTE_V1,
}

# v2
FINAL_NOTE = " If you write a conclusion for your reasoning process, there must be at least one premise. Do not write conclusions without premises. Examples mentioned above are illustrative only and should not be treated as categories that must be checked. The picture may not necessarily require this specific kind of reasoning. Write None if the picture does not involve this specific kind of reasoning or if it is not important in the picture."

ER_EXAMPLE_FORMAT = "[A mother is busy cooking. + A boy is fetching cookies behind the mom. + A girl is shushing the boy. -> The boy is stealing cookies.]"

ER_EXAMPLE = "The boy is stealing cookies, since he is fetching them behind the mom, while she is busy cooking, and being shushed by the girl."

PROMPTS_V2: Dict[str, str] = {
    "0_E": "List the entities appearing in the picture, including people, animals, and objects. List only entities that are clearly visible. Mention each entity at most once (no duplicates). Stop as soon as you have covered all salient entities. Do not guess.",
    "1_STR": "Write your reasoning about the special time context of the story depicted in the picture, for example festivals, seasons, or particular times of day. The special time is relevant only if it requires reasoning beyond what is immediately obvious." + FINAL_NOTE,
    "2_LR": "Write your reasoning about the location of the story depicted in the picture, for example near a school, or at home." + FINAL_NOTE,
    "3_CR": "Write your reasoning about the characters in the picture, including their possible roles or identities, for example a teacher, a doctor, or a student." + FINAL_NOTE,
    "4_CRR": "Write your reasoning about the relationships between the characters in the picture, for example a mother-child relationship or friendship." + FINAL_NOTE,
    "5_ER": "Write your reasoning about the events in the current and previous moments of the picture based on the clues provided. You only need to annotate high-level events and can ignore low-level ones. For example, \"the woman is looking at the man\" is a low-level action. A reasoning process like " + ER_EXAMPLE_FORMAT + " describes a high-level event." + FINAL_NOTE,
    "6_ERR": "Write your reasoning about the relationships between different events in the picture. These relationships are typically causal or temporal (e.g., one event causes or precedes another)." + FINAL_NOTE,
    "7_NMER":" Write your reasoning about the events that are most likely to happen in the next moment. Only include events that have a very high probability of occurring based on the current visual scene." + FINAL_NOTE,
    "8_MSR": "Write your reasoning about the mental states of the subjects in the picture, for example being happy, worried, or daydreaming. You need to reason as best you can about the mental states of all the subjects in the picture. Only omit the subjects that are not showing obvious emotions." + FINAL_NOTE,
}

# for fine-tuning
FORMAT = "Output only the chains (no extra text). Format: Each reasoning chain must follow the structure A1 + A2 + ... + Ak -> B where A1 ... Ak are premises (k ≥ 1), separated by \" + \"(space-plus-space). B is the single conclusion. Constraints: A chain must contain at least one premise; do not output [B]. Do not output multi-step chains like [A -> B -> C]. Split into separate chains, e.g. [A -> B]; [B -> C]. Write one conclusion per chain. Separate different chains with \"; \" (semicolon + space). Do not use ; inside any premise or conclusion. If the image does not involve this reasoning type or it is not important, output exactly None."

PROMPTS_FT: Dict[str, str] = {
    "0_E": "List the entities appearing in the picture, including people, animals, and objects. List only entities that are clearly visible. Mention each entity at most once (no duplicates). Stop as soon as you have covered all salient entities. Do not guess. Separate different entities with a semicolon.",
    "1_STR": "Write your reasoning about the special time context of the story depicted in the picture, for example festivals, seasons, or particular times of day. The special time is relevant only if it requires reasoning beyond what is immediately obvious." + FORMAT,
    "2_LR": "Write your reasoning about the location of the story depicted in the picture, for example near a school, or at home." + FORMAT,
    "3_CR": "Write your reasoning about the characters in the picture, including their possible roles or identities, for example a teacher, a doctor, or a student." + FORMAT,
    "4_CRR": "Write your reasoning about the relationships between the characters in the picture, for example a mother-child relationship or friendship." + FORMAT,
    "5_ER": "Write your reasoning about the events in the current and previous moments of the picture based on the clues provided. You only need to annotate high-level events and can ignore low-level ones. For example, \"the woman is looking at the man\" is a low-level action. A reasoning process like " + ER_EXAMPLE_FORMAT + " describes a high-level event." + FORMAT,
    "6_ERR": "Write your reasoning about the relationships between different events in the picture. These relationships are typically causal or temporal (e.g., one event causes or precedes another)." + FORMAT,
    "7_NMER":" Write your reasoning about the events that are most likely to happen in the next moment. Only include events that have a very high probability of occurring based on the current visual scene." + FORMAT,
    "8_MSR": "Write your reasoning about the mental states of the subjects in the picture, for example being happy, worried, or daydreaming. You need to reason as best you can about the mental states of all the subjects in the picture. Only omit the subjects that are not showing obvious emotions." + FORMAT,
}

PROMPTS_BY_VERSION: Dict[str, Dict[str, str]] = {
    "v1": PROMPTS_V1,
    "v2": PROMPTS_V2,
    "ft": PROMPTS_FT,
}

DEFAULT_PROMPT_VERSION = "v2"

# Backward compatible alias (defaults to v2)
PROMPTS: Dict[str, str] = PROMPTS_V2

def build_prompt_texts(processor, cors: List[str], prompt_version: str = DEFAULT_PROMPT_VERSION) -> List[str]:
    prompts = PROMPTS_BY_VERSION.get(prompt_version)
    if prompts is None:
        valid = ", ".join(sorted(PROMPTS_BY_VERSION.keys()))
        raise ValueError(f"Unknown prompt_version='{prompt_version}'. Valid options: {valid}.")

    has_proc_template = hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None)
    has_tok_template = (
        hasattr(processor, "tokenizer")
        and hasattr(processor.tokenizer, "apply_chat_template")
        and getattr(processor.tokenizer, "chat_template", None)
    )

    out = []
    for cor in cors:
        prompt = prompts[cor]

        if has_proc_template:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            prompt_text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        elif has_tok_template:
            # tokenizer template: keep content as STRING
            conversation = [{
                "role": "user",
                "content": f"<image>\n{prompt}",
            }]
            prompt_text = processor.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

        out.append(prompt_text)
    return out


