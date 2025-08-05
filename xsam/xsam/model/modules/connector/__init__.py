from transformers import AutoConfig, AutoModel

from .configuration_connector import ConnectorConfig
from .modeling_connector import ConnectorModel

AutoConfig.register("connector", ConnectorConfig)
AutoModel.register(ConnectorConfig, ConnectorModel)

__all__ = ["ConnectorConfig", "ConnectorModel"]
