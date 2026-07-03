from xsam.utils.constants import DEFAULT_IMAGE_TOKEN


def img_chat_image_only_map_fn(
    example,
    output_ids_with_output=True,
    image_token=None,
):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example["conversations"]
    input = ""
    conversations = []
    while messages and messages[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg["from"] == "human":
            assert DEFAULT_IMAGE_TOKEN in msg["value"]
            input += image_token or f"{DEFAULT_IMAGE_TOKEN}\n"
        elif msg["from"] == "gpt":
            conversations.append({"input": input, "output": msg["value"]})
            input = ""
        else:
            raise NotImplementedError
    return {"conversations": conversations}


def img_chat_map_fn(
    example,
    image_token=None,
    **kwargs,
):
    messages = example["conversations"]
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
