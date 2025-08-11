from transformers import AutoConfig, AutoModel

from .configuration_projector import DynamicProjectorConfig
from .modeling_projector import DynamicProjectorModel

AutoConfig.register("dynamic_projector", DynamicProjectorConfig)
AutoModel.register(DynamicProjectorConfig, DynamicProjectorModel)

__all__ = ["DynamicProjectorConfig", "DynamicProjectorModel"]
