import random

from xtuner.utils import DEFAULT_IMAGE_TOKEN

from ....utils.constants import DEFAULT_CLS_TOKEN, DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN

SEG_QUESTIONS = [
    "Please identify and segment the {phrase} in this image.",
    "Please segment {phrase} in this image.",
    "What is {phrase} in this image? Please output the corresponding segmentation mask.",
    "Can you segment {phrase} in this image? Please generate the segmentation mask.",
    "Could you provide a segmentation mask for the {phrase} in this image? Please provide the segmentation mask.",
    "Where is the {phrase} in this picture? Please output the segmentation masks.",
    "Can you highlight the {phrase} in this image with a segmentation mask? Please output the segmentation mask.",
    "Could you provide a segmentation mask for the {phrase} in this image? Please respond with the segmentation mask.",
    "Where is the {phrase} in this picture? Please output the corresponding segmentation mask.",
    "Can you highlight the {phrase} in this image with a segmentation mask? Please output the segmentation mask.",
]

ANSWER_LIST = [
    f"{DEFAULT_SEG_TOKEN}.",
    f"It is {DEFAULT_SEG_TOKEN}.",
    f"Sure, {DEFAULT_SEG_TOKEN}.",
    f"Sure, it is {DEFAULT_SEG_TOKEN}.",
    f"Sure, the segmentation mask is {DEFAULT_SEG_TOKEN}.",
]

FORMAT = "{}"
P_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN
C_FORMAT = "{} " + DEFAULT_CLS_TOKEN
P_C_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN + DEFAULT_CLS_TOKEN

FORMAT_DICT = {
    "phrase": P_FORMAT,
    "cls": C_FORMAT,
    "all": P_C_FORMAT,
}


def refer_seg_conversations(labels, output_ids_with_output=True, cond_type="phrase"):
    questions = []
    answers = []
    phrase_format = FORMAT_DICT[cond_type]

    for i, label in enumerate(labels):
        label = label.strip()
        assert len(label.split("||")) == 1
        question_template = random.choice(SEG_QUESTIONS)
        questions.append(question_template.format(phrase=phrase_format.format(label.lower())))
        answers.append(random.choice(ANSWER_LIST) if output_ids_with_output else "")

    rets = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 0:
            rets.append({"from": "human", "value": DEFAULT_IMAGE_TOKEN + question})
        else:
            rets.append({"from": "human", "value": question})
        rets.append({"from": "gpt", "value": answer})
    return rets


def refseg_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    messages = refer_seg_conversations(example["sampled_sents"], output_ids_with_output, cond_type)
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
