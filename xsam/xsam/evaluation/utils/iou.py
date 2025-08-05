import numpy as np
from tabulate import tabulate


class IouStat:
    def __init__(self, cat_names=["ignore", "refer"]):
        self.cat_names = cat_names
        self.num_cats = len(cat_names)

        self.intersection = 0.0
        self.union = 0.0
        self.count = 0.0
        self.acc_iou = 0.0
        self.ciou = 0.0
        self.giou = 0.0

    def update(self, intersection, union, n=1):
        """
        Args:
            intersection: array-like, shape (num_cats,)
            union: array-like, shape (num_cats,)
            n: number of samples
        """
        intersection = np.asarray(intersection)
        union = np.asarray(union)

        self.intersection += intersection
        self.union += union
        self.count += n

        iou_per_sample = np.where(union > 0, intersection / union, 1.0)
        self.acc_iou += iou_per_sample

    def average(self):
        # cIoU (cumulative IoU)
        self.ciou = np.where(self.union > 0, self.intersection / self.union * 100, 100.0)

        # gIoU (global mean IoU)
        self.giou = np.where(self.count > 0, self.acc_iou / self.count * 100, 0.0)

    def reset(self):
        self.intersection.fill(0.0)
        self.union.fill(0.0)
        self.count.fill(0.0)
        self.acc_iou.fill(0.0)
        self.ciou.fill(0.0)
        self.giou.fill(0.0)

    def __repr__(self) -> str:
        headers = ["", "cIoU", "gIoU"]
        data = []
        for i, cat_name in enumerate(self.cat_names):
            data.append(
                [
                    cat_name,
                    self.ciou[i],
                    self.giou[i],
                ]
            )

        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        return str(table)
