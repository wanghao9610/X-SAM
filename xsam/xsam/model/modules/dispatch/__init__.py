import os
import types

import torch
import transformers
from mmengine.config.lazy import LazyObject
from mmengine.utils import digit_version
from transformers.utils.import_utils import is_flash_attn_2_available

TRANSFORMERS_VERSION = digit_version(transformers.__version__)
IS_LOW_VERSION_TRANSFORMERS = TRANSFORMERS_VERSION < digit_version("4.38")
# Transformers requires torch version >= 2.1.1 when using Torch SDPA.
# Refer to https://github.com/huggingface/transformers/blob/caa5c65db1f4db617cdac2ad667ba62edf94dd98/src/transformers/modeling_utils.py#L1611  # noqa: E501
SUPPORT_FLASH1 = digit_version(torch.__version__) >= digit_version("2.1.1")
SUPPORT_FLASH2 = is_flash_attn_2_available()
SUPPORT_FLASH = SUPPORT_FLASH1 or SUPPORT_FLASH2

USE_TRITON_KERNEL = bool(os.getenv("USE_TRITON_KERNEL", default=0))
SUPPORT_TRITON = False
try:
    import triton  # pre-check # noqa: F401
    import triton.language as tl  # pre-check # noqa: F401

    SUPPORT_TRITON = True
except ImportError:
    if USE_TRITON_KERNEL:
        raise RuntimeError(
            "USE_TRITON_KERNEL is set to 1, but triton has not been installed."
            " Run `pip install triton==2.1.0` to install triton."
        )

NO_ATTN_WEIGHTS_MSG = (
    "Due to the implementation of the PyTorch version of flash attention, "
    "even when the `output_attentions` flag is set to True, it is not "
    "possible to return the `attn_weights`."
)

LOWEST_TRANSFORMERS_VERSION = dict(
    LlamaForCausalLM=digit_version("4.48"),
    Phi3ForCausalLM=digit_version("4.39"),
    Qwen2ForCausalLM=digit_version("4.48"),
    Qwen2MoeForCausalLM=digit_version("4.48"),
    # Qwen3ForCausalLM=digit_version("4.52"),
    # Qwen3VLForConditionalGeneration=digit_version("4.57"),
)

ATTN_DISPATCH_MAPPING = dict(
    LlamaAttention=LazyObject("xsam.model.modules.dispatch.llama", "llama_attn_forward"),
    Phi3FlashAttention2=LazyObject("xsam.model.modules.dispatch.phi3", "phi3_attn_forward"),
    Qwen2Attention=LazyObject("xsam.model.modules.dispatch.qwen2", "qwen2_attn_forward"),
    Qwen2MoeAttention=LazyObject("xsam.model.modules.dispatch.qwen2", "qwen2_attn_forward"),
    # Qwen3Attention=LazyObject("xsam.model.modules.dispatch.qwen3", "qwen3_attn_forward"),
    # Qwen3VLTextAttention=LazyObject("xsam.model.modules.dispatch.qwen3vl", "qwen3vl_attn_forward"),
)

VARLEN_ATTN_DISPATCH_MAPPING = dict(
    LlamaAttention=LazyObject("xsam.model.modules.dispatch.llama", "llama_attn_forward"),
    Phi3FlashAttention2=LazyObject("xsam.model.modules.dispatch.phi3", "phi3_varlen_attn_forward"),
    Qwen2Attention=LazyObject("xsam.model.modules.dispatch.qwen2", "qwen2_attn_forward"),
    Qwen2MoeAttention=LazyObject("xsam.model.modules.dispatch.qwen2", "qwen2_attn_forward"),
    # Qwen3Attention=LazyObject("xsam.model.modules.dispatch.qwen3", "qwen3_attn_forward"),
    # Qwen3VLTextAttention=LazyObject("xsam.model.modules.dispatch.qwen3vl", "qwen3vl_attn_forward"),
)

RMS_DISPATCH_MAPPING = dict(
    LlamaRMSNorm=LazyObject("xsam.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
    Phi3RMSNorm=LazyObject("xsam.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
    Qwen2RMSNorm=LazyObject("xsam.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
    Qwen2MoeRMSNorm=LazyObject("xsam.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
    # Qwen3RMSNorm=LazyObject("xsam.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
    # Qwen3VLTextRMSNorm=LazyObject("xsam.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
)


def log_once(func):
    logged = False

    def wrapper(*args, **kwargs):
        nonlocal logged
        if not logged:
            logged = True
            func(*args, **kwargs)
        return

    return wrapper


def dispatch_attn_forward(model):
    if not SUPPORT_FLASH2:
        return

    from xsam.utils.logging import print_log

    print_log = log_once(print_log)

    attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in ATTN_DISPATCH_MAPPING:
            if attn_forward is None:
                attn_forward = ATTN_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            print_log(f"Dispatch {name} forward. {NO_ATTN_WEIGHTS_MSG}", "current")
            module.forward = types.MethodType(attn_forward, module)


def dispatch_varlen_attn_forward(model):
    if not SUPPORT_FLASH2:
        return

    from xsam.utils.logging import print_log

    varlen_attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in VARLEN_ATTN_DISPATCH_MAPPING:
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            print_log(f"Dispatch {name} varlen forward. {NO_ATTN_WEIGHTS_MSG}", "current")
            module.forward = types.MethodType(varlen_attn_forward, module)


def dispatch_rmsnorm_forward(model):
    if (not SUPPORT_TRITON) or (not USE_TRITON_KERNEL):
        return

    from xsam.utils.logging import print_log

    print_log = log_once(print_log)

    rms_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in RMS_DISPATCH_MAPPING:
            if rms_forward is None:
                rms_forward = RMS_DISPATCH_MAPPING[name]
                rms_forward = rms_forward.build()
            print_log(f"Dispatch {name} forward.", "current")
            module.forward = types.MethodType(rms_forward, module)


def dispatch_modules(model, use_varlen_attn=False):
    def check(model_name):
        if "ForCausalLM" not in model_name and model_name.endswith("Model"):
            # a walkaround for reward model
            model_name = model_name[:-5] + "ForCausalLM"
        msg = "{} requires transformers version at least {}, but got {}"
        if model_name in LOWEST_TRANSFORMERS_VERSION:
            assert TRANSFORMERS_VERSION >= LOWEST_TRANSFORMERS_VERSION[model_name], msg.format(
                model_name,
                LOWEST_TRANSFORMERS_VERSION[model_name],
                TRANSFORMERS_VERSION,
            )

    check(type(model).__name__)
    if use_varlen_attn:
        dispatch_varlen_attn_forward(model)
    else:
        dispatch_attn_forward(model)
    dispatch_rmsnorm_forward(model)


__all__ = ["dispatch_modules"]
