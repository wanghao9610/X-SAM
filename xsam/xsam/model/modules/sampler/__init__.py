from transformers import AutoConfig, AutoModel

from .configuration_sampler import SamplerConfig
from .modeling_sampler import SamplerModel

AutoConfig.register("sampler", SamplerConfig)
AutoModel.register(SamplerConfig, SamplerModel)

__all__ = ["SamplerConfig", "SamplerModel"]
