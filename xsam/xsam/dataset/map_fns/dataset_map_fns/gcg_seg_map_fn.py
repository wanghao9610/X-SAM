import random

from xtuner.utils import DEFAULT_IMAGE_TOKEN

from ....utils.constants import DEFAULT_CLS_TOKEN, DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN

QUESTION_LIST = [
    "Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    "Can you provide a brief description of the this image? Please output interleaved segmentation masks for the corresponding phrases.",
    "Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    "Could you give a brief explanation of what can be found within this picture? Please output interleaved segmentation masks for the corresponding phrases.",
    "Could you give me an brief explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.",
    "Could you provide me with a briefly analysis of this photo? Please output interleaved segmentation masks for the corresponding parts of the answer.",
]


ANSWER_LIST = [
    f"{DEFAULT_SEG_TOKEN}.",
    f"And {DEFAULT_SEG_TOKEN}.",
    f"The segmentation mask is {DEFAULT_SEG_TOKEN}.",
    f"And the segmentation mask is {DEFAULT_SEG_TOKEN}.",
]

P_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN
P_C_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN + DEFAULT_CLS_TOKEN

FORMAT_DICT = {
    "phrase": P_FORMAT,
    "cls": P_C_FORMAT,
    "all": P_C_FORMAT,
}


def tag_caption(caption, tokens, phrase_format=P_FORMAT):
    for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
        caption = f"{caption[:start]}{phrase_format.format(caption[start:end])} {caption[end:]}"
    return caption


def gcg_seg_conversations(caption, tokens_positive, question=None, output_ids_with_output=True, cond_type="phrase"):
    questions = []
    answers = []
    phrase_format = FORMAT_DICT[cond_type]

    question = random.choice(QUESTION_LIST) if question is None else question
    if output_ids_with_output and tokens_positive is not None:
        answer = tag_caption(caption, tokens_positive, phrase_format).strip() + " " + random.choice(ANSWER_LIST)
    else:
        answer = ""
    questions.append(question)
    answers.append(answer)

    rets = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 0:
            rets.append({"from": "human", "value": DEFAULT_IMAGE_TOKEN + question})
        else:
            rets.append({"from": "human", "value": question})
        rets.append({"from": "gpt", "value": answer})
    return rets


def gcg_seg_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    if "annotations" in example:
        tokens_positive = [ann["tokens_positive"] for ann in example["annotations"]]
        assert all(
            tokens_positive[i][0] < tokens_positive[i][1] for i in range(len(tokens_positive))
        ), f"tokens_positive intervals are not valid: {tokens_positive}"
        assert all(
            tokens_positive[i][1] < tokens_positive[i + 1][0] for i in range(len(tokens_positive) - 1)
        ), f"tokens_positive intervals are not disjoint: {tokens_positive}"
        caption = example.get("caption", None)
        question = example.get("question", None)
    else:
        tokens_positive = None
        caption = None
        question = None
    messages = gcg_seg_conversations(caption, tokens_positive, question, output_ids_with_output, cond_type)
    input = ""
    conversation = []
    while messages and messages[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg["from"] == "human":
            if DEFAULT_IMAGE_TOKEN in msg["value"]:
                msg["value"] = msg["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                msg["value"] = DEFAULT_IMAGE_TOKEN + "\n" + msg["value"]
                msg["value"] = msg["value"].strip()
            input += msg["value"]

        elif msg["from"] == "gpt":
            conversation.append({"input": input, "output": msg["value"]})
            input = ""
        else:
            raise NotImplementedError
    example.update({"conversation": conversation})
    return example
