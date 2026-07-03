import random

from xsam.utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
)

SEG_SEG_QUESTIONS = [
    "Can you segment the image based on the following categories: {categories}? Please output the segmentation masks.",
    "Can you generate segmentation masks for this image based on the specified categories: {categories}? Please generate the segmentation masks.",
    "Can you provide segmentation masks for this image based on these categories: {categories}? Please provide the segmentation masks.",
    "Could you create segmentation masks for this image according to the specified categories: {categories}? Please create the segmentation masks.",
    "Could you output segmentation masks for this image that highlight the following categories: {categories}? Please output the segmentation masks.",
    "Could you provide segmentation masks for this image according to the specified categories: {categories}? Please respond with the segmentation masks.",
]

SEG_CAPTION_SEG_QUESTIONS = [
    "Can you segment the image based on the following categories: {categories}? Please first describe the image briefly, then respond with segmentation masks.",
    "Can you generate segmentation masks for this image based on the specified categories: {categories}? Please first briefly describe the contents of the image, then respond with segmentation masks.",
    "Can you provide segmentation masks for this image based on these categories: {categories}? Please first give me a brief description of the image, then output segmentation masks.",
    "Could you create segmentation masks for this image according to the specified categories: {categories}? Please first give me a brief description of this picture, then respond with segmentation masks.",
    "Could you output segmentation masks for this image that highlight the following categories: {categories}? Please first provide me with a brief description of this photo, then respond with segmentation masks.",
    "Could you provide segmentation masks for this image according to the specified categories: {categories}. Please first describe the image briefly, then respond with segmentation masks.",
]

SEG_ANSWER_LIST = [
    f"{DEFAULT_SEG_TOKEN}.",
    f"It is {DEFAULT_SEG_TOKEN}.",
    f"Sure, {DEFAULT_SEG_TOKEN}.",
    f"Sure, it is {DEFAULT_SEG_TOKEN}.",
    f"Sure, the segmentation result is {DEFAULT_SEG_TOKEN}.",
]

SEG_CAPTION_ANSWER_LIST = [
    "{caption} {seg_token}.",
    "{caption} And it is {seg_token}.",
    "{caption} And {seg_token}.",
    "{caption} And the segmentation result is {seg_token}.",
]

P_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN
C_FORMAT = "{} " + DEFAULT_CLS_TOKEN
P_C_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN + DEFAULT_CLS_TOKEN

FORMAT_DICT = {
    "phrase": P_FORMAT,
    "cls": C_FORMAT,
    "all": P_C_FORMAT,
}


def tag_categories(categories, format=P_FORMAT):
    formatted_categories = []
    for category in categories:
        category = (
            category.replace("-merged", "").replace("-other", "").replace("-stuff", "").replace("-", " ").lower()
        )
        category = format.format(category)
        formatted_categories.append(category)

    formatted_categories = ", ".join(formatted_categories)
    return formatted_categories


def generic_seg_conversations(categories, caption=None, output_ids_with_output=True, cond_type="phrase"):
    questions = []
    answers = []
    format = FORMAT_DICT[cond_type]

    if caption is None:
        question = random.choice(SEG_SEG_QUESTIONS).format(categories=tag_categories(categories, format=format))
        answer = random.choice(SEG_ANSWER_LIST) if output_ids_with_output else ""
    else:
        question = random.choice(SEG_CAPTION_SEG_QUESTIONS).format(
            categories=tag_categories(categories, format=format)
        )
        answer = (
            random.choice(SEG_CAPTION_ANSWER_LIST).format(caption=caption, seg_token=DEFAULT_SEG_TOKEN)
            if output_ids_with_output
            else ""
        )

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


def img_genseg_map_fn(
    example,
    output_ids_with_output=True,
    cond_type="phrase",
    image_token=None,
):
    messages = generic_seg_conversations(
        example["sampled_cats"], example.get("caption", None), output_ids_with_output, cond_type
    )
    input = ""
    conversations = []
    while messages and messages[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg["from"] == "human":
            if DEFAULT_IMAGE_TOKEN in msg["value"]:
                msg["value"] = msg["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                msg["value"] = (image_token or f"{DEFAULT_IMAGE_TOKEN}\n") + msg["value"]
                msg["value"] = msg["value"].strip()
            input += msg["value"]
        elif msg["from"] == "gpt":
            conversations.append({"input": input, "output": msg["value"]})
            input = ""
        else:
            raise NotImplementedError
    example.update({"conversations": conversations})
    return example
