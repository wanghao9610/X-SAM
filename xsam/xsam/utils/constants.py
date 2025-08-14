from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

DEFAULT_SEG_TOKEN = "<SEG>"
DEFAULT_CLS_TOKEN = "<CLS>"
DEFAULT_PSTART_TOKEN = "<p>"
DEFAULT_PEND_TOKEN = "</p>"
DEFAULT_REGION_TOKEN = "<region>"
REGION_TOKEN_INDEX = -300

DEFAULT_TASKS = ["imgconv", "genseg", "refseg", "reaseg", "gcgseg", "ovseg", "interseg", "vgdseg"]
TOKEN2INDEX = {
    DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_INDEX,
    DEFAULT_REGION_TOKEN: REGION_TOKEN_INDEX,
}
INDEX2TOKEN = {v: k for k, v in TOKEN2INDEX.items()}
