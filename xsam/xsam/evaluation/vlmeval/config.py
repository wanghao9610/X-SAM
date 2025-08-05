from functools import partial
from os import getenv

import vlmeval.config as vlmeval_config

from .xsam_xtuner import XSam_XTuner

init_dir = getenv("INIT_DIR", "./inits/")

xsam_series = {
    "llava-phi3-siglip2-ft": partial(
        XSam_XTuner,
        xsam_path=init_dir + "llava-phi3-siglip2-ft",
        visual_encoder_path=init_dir + "siglip2-so400m-patch14-384",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="phi3_chat",
    ),
    "xsam-phi3-siglip2-sam-l-mft": partial(
        XSam_XTuner,
        xsam_path=init_dir + "xsam-phi3-siglip2-sam-l-mft",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="phi3_chat",
    ),
}

vlmeval_config.supported_VLM.update(xsam_series)
supported_VLM = vlmeval_config.supported_VLM
