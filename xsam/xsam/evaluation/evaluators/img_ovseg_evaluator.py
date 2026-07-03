from .img_genseg_evaluator import ImgGenSegEvaluator


class ImgOVSegEvaluator(ImgGenSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
