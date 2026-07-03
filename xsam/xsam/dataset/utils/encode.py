import copy
import re

from xsam.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PLACEHOLDER_TOKEN,
    DEFAULT_REGION_TOKEN,
    DEFAULT_VISION_END_TOKEN,
    DEFAULT_VISION_START_TOKEN,
    IGNORE_INDEX,
    TOKEN2INDEX,
)

from ..utils.tokenize import get_bos_eos_token_ids


def encode_fn(
    example,
    tokenizer,
    max_length,
    image_processor,
    input_ids_with_output=True,
    use_placeholder=False,
    use_vision_token=False,
    next_needs_bos_token=True,
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversations'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversations dataset.
        example['conversations'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversations dataset.
        example['conversations'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example["conversations"]) > 1
    image_grid_thw = example.get("image_grid_thw", None)
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    for single_turn_conversation in example["conversations"]:
        input = single_turn_conversation["input"]
        if DEFAULT_IMAGE_TOKEN in input or DEFAULT_REGION_TOKEN in input:
            if image_grid_thw is not None:
                merge_length = getattr(image_processor, "merge_size", 1) ** 2
                index = 0
                while DEFAULT_IMAGE_TOKEN in input and use_placeholder:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    vision_placeholder = ""
                    vision_content = "<|placeholder|>" * num_image_tokens
                    vision_placeholder += (
                        (DEFAULT_VISION_START_TOKEN if use_vision_token else "")
                        + vision_content
                        + (DEFAULT_VISION_END_TOKEN if use_vision_token else "")
                    )
                    if f"{DEFAULT_VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_VISION_END_TOKEN}" in input:
                        input = input.replace(
                            f"{DEFAULT_VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_VISION_END_TOKEN}",
                            vision_placeholder,
                            1,
                        )
                    else:
                        input = input.replace(DEFAULT_IMAGE_TOKEN, vision_placeholder, 1)
                    index += 1
                input = input.replace("<|placeholder|>", DEFAULT_PLACEHOLDER_TOKEN)

            pattern = f"({'|'.join(re.escape(token) for token in TOKEN2INDEX.keys())})"
            chunks = [chunk for chunk in re.split(pattern, input) if chunk.strip() != ""]
            input_encode = [
                [TOKEN2INDEX[chunk]] if chunk in TOKEN2INDEX else tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in chunks
            ]
            input_encode = [item for sublist in input_encode if isinstance(sublist, list) for item in sublist]
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get("output_with_loss", True)
            output = single_turn_conversation["output"]
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get("need_eos_token", True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get("sep", "")
            if sep != "":
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}
