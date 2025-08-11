from transformers import PretrainedConfig


class SamplerConfig(PretrainedConfig):
    model_type = "sampler"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        sampler_type="naive",
        input_dim=2048,
        output_dim=256,
        num_init_point=1024,
        num_sample_point=256,
        num_sub_point=[16, 32, 64, 128],
        num_neighbor=[16, 32, 64, 128],
        pooler_mode="mean",
        initializer_range=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampler_type = sampler_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_init_point = num_init_point
        self.num_sample_point = num_sample_point
        self.num_sub_point = num_sub_point
        self.num_neighbor = num_neighbor
        self.pooler_mode = pooler_mode
        self.initializer_range = initializer_range
