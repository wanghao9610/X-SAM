from typing import Optional, Tuple

from mmengine.config import Config
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.utils import StopWordStoppingCriteria


def setup_model_config(model, cfg: Config) -> Tuple[Optional[StoppingCriteriaList], Optional[GenerationConfig]]:
    """Setup model configuration for generation."""
    stop_criteria = None
    generation_config = None

    if model.llm is not None:
        prompt_template = cfg.prompt_template
        stop_words = []
        if isinstance(prompt_template, str):
            prompt_template = get_object_from_string(prompt_template)
        stop_words += prompt_template.get("STOP_WORDS", [])

        stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            stop_criteria.append(StopWordStoppingCriteria(model.tokenizer, word))

        generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            temperature=1,
            top_p=None,
            bos_token_id=model.tokenizer.bos_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=(
                model.tokenizer.pad_token_id
                if model.tokenizer.pad_token_id is not None
                else model.tokenizer.eos_token_id
            ),
        )

    return stop_criteria, generation_config
