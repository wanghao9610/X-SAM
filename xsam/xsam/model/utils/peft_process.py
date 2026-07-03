import torch.nn as nn
from peft import PeftType


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "output_layer" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("output_layer")
    return list(lora_module_names)


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    # Modified from `https://github.com/huggingface/peft/blob/main/src/peft/utils/save_and_load.py`  # noqa: E501

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type == PeftType.LORA:
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`  # noqa: E501
        # to be used directly with the state dict which is necessary
        # when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
    else:
        # Currently we only support lora
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key] = value

    return to_return
