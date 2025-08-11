from transformers import PretrainedConfig


class ConnectorConfig(PretrainedConfig):
    model_type = "connector"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        bias=False,
        scale_factor=[4, 2, 1, 0.5],  # [1/4, 1/8, 1/16, 1/32]
        segmentor_encoder_channels=[1280, 1280, 1280, 1280],
        connector_type="conv",
        hidden_channels=256,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bias = bias
        self.scale_factor = scale_factor
        self.segmentor_encoder_channels = segmentor_encoder_channels
        self.connector_type = connector_type
        self.hidden_channels = hidden_channels
        self.initializer_range = initializer_range
