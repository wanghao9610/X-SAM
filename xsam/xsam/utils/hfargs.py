from dataclasses import dataclass, field
from typing import Optional

import transformers

__all__ = ["TrainingArguments"]


@dataclass
class HFArguments(transformers.TrainingArguments):
    # Model arguments
    model_name_or_path: Optional[str] = field(default=None)

    # Data arguments
    data_path: Optional[str] = field(default=None)
    image_folder: Optional[str] = field(default=None)
    pan_segmap_folder: Optional[str] = field(default=None)
    group_by_modality_length: Optional[bool] = field(default=False)

    # Training specific arguments
    llm_lr: Optional[float] = field(default=None)
    vision_encoder_lr: Optional[float] = field(default=None)
    visual_projector_lr: Optional[float] = field(default=None)
    mmsegmention_decoder_lr: Optional[float] = field(default=None)
    segmention_encoder_lr: Optional[float] = field(default=None)
    segmention_decoder_lr: Optional[float] = field(default=None)


hf_parser = transformers.HfArgumentParser(HFArguments)
