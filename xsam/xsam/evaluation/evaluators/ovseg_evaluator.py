from .genseg_evaluator import GenSegEvaluator


class OVSegEvaluator(GenSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
