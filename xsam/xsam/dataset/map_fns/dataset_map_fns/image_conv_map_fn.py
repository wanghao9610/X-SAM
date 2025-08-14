from xtuner.utils import DEFAULT_IMAGE_TOKEN


def llava_conv_image_only_map_fn(example, output_ids_with_output=True):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example["conversations"]
    input = ""
    conversation = []
    while messages and messages[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg["from"] == "human":
            assert DEFAULT_IMAGE_TOKEN in msg["value"]
            input += DEFAULT_IMAGE_TOKEN
        elif msg["from"] == "gpt":
            conversation.append({"input": input, "output": msg["value"]})
            input = ""
        else:
            raise NotImplementedError
    return {"conversation": conversation}


def image_conv_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    messages = example["conversations"]
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
    return {"conversation": conversation}
