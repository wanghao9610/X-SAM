from .gcgseg_map_fn import gcgseg_map_fn
from .genseg_map_fn import genseg_map_fn
from .imgconv_map_fn import imgconv_map_fn
from .intseg_map_fn import intseg_map_fn
from .ovseg_map_fn import ovseg_map_fn
from .reaseg_map_fn import reaseg_map_fn
from .refseg_map_fn import refseg_map_fn
from .vgdseg_map_fn import vgdseg_map_fn

__all__ = [
    "genseg_map_fn",
    "refseg_map_fn",
    "imgconv_map_fn",
    "imgconv_map_fn",
    "gcgseg_map_fn",
    "vgdseg_map_fn",
    "reaseg_map_fn",
    "intseg_map_fn",
    "ovseg_map_fn",
]
