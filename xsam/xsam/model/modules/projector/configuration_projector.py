from transformers import PretrainedConfig


class DynamicProjectorConfig(PretrainedConfig):
    model_type = "dynamic_projector"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        downsample_ratio=1.0,
        depth=2,
        hidden_act="gelu",
        bias=True,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.downsample_ratio = downsample_ratio
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        super().__init__(**kwargs)
