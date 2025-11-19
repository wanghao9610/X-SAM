import random

from xtuner.utils import DEFAULT_IMAGE_TOKEN

from ....utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_REGION_TOKEN,
    DEFAULT_SEG_TOKEN,
)

QUESTION_LIST = [
    "Can you segment the image based on the following regions: {regions}? Please output the corresponding segmentation mask.",
    "Can you generate segmentation masks for this image based on the specified regions: {regions}? Please generate the segmentation masks.",
    "Can you provide segmentation masks for this image based on these regions: {regions}? Please provide the segmentation masks.",
    "Could you create segmentation masks for this image according to the specified regions: {regions}? Please create the segmentation masks.",
    "Could you output segmentation masks for this image that highlight the following regions: {regions}? Please output the segmentation masks.",
    "Could you provide segmentation masks for this image according to the specified regions: {regions}? Please respond with the segmentation masks.",
]

ANSWER_LIST = [
    f"{DEFAULT_SEG_TOKEN}.",
    f"It is {DEFAULT_SEG_TOKEN}.",
    f"Sure, {DEFAULT_SEG_TOKEN}.",
    f"Sure, it is {DEFAULT_SEG_TOKEN}.",
    f"Sure, the segmentation result is {DEFAULT_SEG_TOKEN}.",
]

P_FORMAT = DEFAULT_PSTART_TOKEN + DEFAULT_REGION_TOKEN + DEFAULT_PEND_TOKEN
C_FORMAT = DEFAULT_REGION_TOKEN + DEFAULT_CLS_TOKEN
P_C_FORMAT = DEFAULT_PSTART_TOKEN + DEFAULT_REGION_TOKEN + DEFAULT_PEND_TOKEN + DEFAULT_CLS_TOKEN

FORMAT_DICT = {
    "phrase": P_FORMAT,
    "cls": C_FORMAT,
    "all": P_C_FORMAT,
}


def tag_region(regions, format=P_FORMAT):
    formatted_regions = []
    for _ in regions:
        formatted_regions.append(format)

    formatted_regions = ", ".join(formatted_regions)
    return formatted_regions


def vgd_seg_conversations(regions, output_ids_with_output=True, cond_type="phrase"):
    questions = []
    answers = []
    region_format = FORMAT_DICT[cond_type]

    question = random.choice(QUESTION_LIST).format(regions=tag_region(regions, format=region_format))
    answer = random.choice(ANSWER_LIST) if output_ids_with_output else ""

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


def vgdseg_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    messages = vgd_seg_conversations(example["sampled_labels"], output_ids_with_output, cond_type)
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
