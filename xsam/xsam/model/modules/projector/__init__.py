from transformers import AutoConfig, AutoModel

from .configuration_projector import DynamicProjectorConfig, ProjectorConfig
from .modeling_projector import DynamicProjectorModel, ProjectorModel

AutoConfig.register("projector", ProjectorConfig)
AutoModel.register(ProjectorConfig, ProjectorModel)

AutoConfig.register("dynamic_projector", DynamicProjectorConfig)
AutoModel.register(DynamicProjectorConfig, DynamicProjectorModel)

__all__ = ["ProjectorConfig", "ProjectorModel", "DynamicProjectorConfig", "DynamicProjectorModel"]
