import os
import os.path as osp
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
from mmengine._strategy import DeepSpeedStrategy
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.runner import FlexibleRunner

from xsam.utils.logging import print_log

DATA_BATCH = Optional[Union[dict, tuple, list]]


class PTCheckpointHook(Hook):
    priority = "VERY_LOW"

    def __init__(self, out_dir: Optional[Union[str, Path]] = None, clean_pth: bool = False) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.clean_pth = clean_pth

    def after_run(self, runner) -> None:
        assert isinstance(runner, FlexibleRunner), "Runner should be `FlexibleRunner`"
        assert isinstance(runner.strategy, DeepSpeedStrategy), "Strategy should be `DeepSpeedStrategy`"

        if self.out_dir is None:
            self.out_dir = runner.work_dir

        wrapped_model = runner.strategy.model
        if wrapped_model.zero_optimization_partition_weights():
            assert wrapped_model.zero_gather_16bit_weights_on_model_save(), (
                "Please set `gather_16bit_weights_on_model_save=True` " "in your DeepSpeed config."
            )
            state_dict = wrapped_model._zero3_consolidated_16bit_state_dict()
        else:
            state_dict = wrapped_model.module_state_dict(
                exclude_frozen_parameters=runner.strategy.exclude_frozen_parameters
            )

        torch.save(state_dict, osp.join(self.out_dir, "pytorch_model.bin"))
        print_log(f"Model saved to {osp.join(self.out_dir, 'pytorch_model.bin')}", logger="current")
        pth_dirs = [osp.join(self.out_dir, d) for d in os.listdir(self.out_dir) if d.endswith(".pth")]
        if len(pth_dirs) > 1 and is_main_process() and self.clean_pth:
            for pth_dir in pth_dirs:
                print_log(f"Removing {pth_dir}", logger="current")
                shutil.rmtree(pth_dir)
