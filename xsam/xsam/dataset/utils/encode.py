import copy
import re

from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

from ...utils.constants import DEFAULT_REGION_TOKEN, TOKEN2INDEX


def encode_fn(
    example, tokenizer, max_length, input_ids_with_output=True, with_image_token=False, next_needs_bos_token=True
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
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
    is_multi_turn_conversation = len(example["conversation"]) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    for single_turn_conversation in example["conversation"]:
        input = single_turn_conversation["input"]
        if (DEFAULT_IMAGE_TOKEN in input or DEFAULT_REGION_TOKEN in input) and with_image_token:
            pattern = f"({'|'.join(TOKEN2INDEX.keys())})"
            chunks = re.split(pattern, input)
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
