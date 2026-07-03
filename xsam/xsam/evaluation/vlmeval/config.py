from functools import partial
from os import getenv

import vlmeval.config as vlmeval_config

from .xsam_chat import XSamChat

init_dir = getenv("INIT_DIR", "./inits/")

xsam_series = {
    "xsam-phi3-mini-4k-instruct-siglip2-so400m-p14-384-sam-large-m2f": partial(
        XSamChat,
        xsam_path=init_dir + "xsam-phi3-mini-4k-instruct-siglip2-so400m-p14-384-sam-large-m2f",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="phi3_chat",
        expand2square=False,
        use_placeholder=False,
        use_dual_encoder=True,
        image_token="<image>",
    ),
    "xsam-qwen3-vl-4b-sam-large-m2f-lora": partial(
        XSamChat,
        xsam_path=init_dir + "xsam-qwen3-vl-4b-sam-large-m2f-lora",
        vlm_path=init_dir + "Qwen3-VL-4B-Instruct",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="qwen3_instruct",
        expand2square=False,
        use_placeholder=True,
        use_dual_encoder=False,
        image_token="<|vision_start|><image><|vision_end|>",
    ),
}

vlmeval_config.supported_VLM.update(xsam_series)
supported_VLM = vlmeval_config.supported_VLM
