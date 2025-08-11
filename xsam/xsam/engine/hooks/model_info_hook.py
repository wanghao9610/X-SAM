from typing import List

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from tabulate import tabulate


def split_list(lst, value):
    res = []
    tmp_res = []
    for i in lst:
        if i == value:
            res.append(tmp_res)
            tmp_res = []
        else:
            tmp_res.append(i)
    res.append(tmp_res)
    return res


class ModelInfoHook(Hook):
    def __init__(self, module_names: List[str] = ["llm"], display_params: bool = False):
        self.module_names = module_names
        self.display_params = display_params

    def log(self, runner, model):
        module_params = {"others": {"num_params": 0, "num_trainable_params": 0, "params": []}}
        module_params.update(
            {name: {"num_params": 0, "num_trainable_params": 0, "params": []} for name in self.module_names}
        )

        for name, param in model.named_parameters():
            num_params = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
            matched_module_name = next(
                (module_name for module_name in self.module_names if f"{module_name}." in name),
                "others",
            )
            module_params[matched_module_name]["num_params"] += num_params
            module_params[matched_module_name]["params"].append(
                (name, param.dtype, "trainable" if param.requires_grad else "frozen", num_params)
            )
            if param.requires_grad:
                module_params[matched_module_name]["num_trainable_params"] += num_params

        total_params = sum([module_params[name]["num_params"] for name in self.module_names])
        num_trainable_params = sum([module_params[name]["num_trainable_params"] for name in self.module_names])
        module_params = {k: v for k, v in module_params.items() if v["num_params"] != 0}
        module_params = dict(
            sorted(
                module_params.items(),
                key=lambda x: (x[1]["num_params"], -x[1]["num_trainable_params"] > 0),
                reverse=True,
            )
        )
        param_table = tabulate(
            [
                [name, p[0], p[1], p[2], f"{p[3]:,}"]
                for name in module_params.keys()
                for p in module_params[name]["params"]
            ],
            headers=["Module", "Name", "DataType", "Status", "# Params"],
            tablefmt="outline",
            stralign="left",
            numalign="left",
        )

        headers = ["Module", "# Params", "Params %", "# Trainable", "Trainable %"]
        module_lens = [len(name) for name in module_params.keys()]
        num_table = tabulate(
            [
                *[
                    [
                        name,
                        f"{module_params['num_params']:,}",
                        f"{module_params['num_params'] / total_params * 100:.2f}%",
                        f"{module_params['num_trainable_params']:,}",
                        f"{module_params['num_trainable_params'] / module_params['num_params'] * 100:.2f}%",
                    ]
                    for name, module_params in module_params.items()
                ],
                [
                    "=" * int(len(header) * scale)
                    for header, scale in zip(headers, [max(module_lens) / len(headers[0]), 1.8, 1.6, 1.2, 1.2])
                ],
                [
                    "Total",
                    f"{total_params:,}",
                    f"{total_params / total_params * 100:.2f}%",
                    f"{num_trainable_params:,}",
                    f"{num_trainable_params / total_params * 100:.2f}%",
                ],
            ],
            headers=headers,
            tablefmt="outline",
            colalign=("center", "right", "right", "right", "right"),
        )

        if self.display_params:
            runner.logger.info(f"Model status:\n{param_table}")
        runner.logger.info(f"Model summary:\n{num_table}")

    def before_train(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        self.log(runner, model)

    def before_val(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        self.log(runner, model)

    def before_test(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        self.log(runner, model)
