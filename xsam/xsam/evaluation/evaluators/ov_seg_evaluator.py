from .generic_seg_evaluator import GenericSegEvaluator


class OVSegEvaluator(GenericSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
