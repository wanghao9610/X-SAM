from xtuner.utils import PROMPT_TEMPLATE

PROMPT_TEMPLATE.update(
    dict(
        qwen3_thinking=dict(
            SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
            INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n<think>"),
            SUFFIX="<|im_end|>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
        ),
        qwen3_instruct=dict(
            SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
            INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
            SUFFIX="<|im_end|>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
        ),
        qwen3_vl_instruct=dict(
            SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
            INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
            SUFFIX="<|im_end|>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
        ),
    ),
)
