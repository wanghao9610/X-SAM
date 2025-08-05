import argparse
import os
from typing import Tuple

from mmengine.dist import get_dist_info, init_dist
from mmengine.utils.dl_utils import set_multi_processing

from xsam.utils.logging import print_log


def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Setup distributed training environment."""
    if args.launcher != "none":
        set_multi_processing(distributed=True)
        init_dist(args.launcher)
        rank, world_size = get_dist_info()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = local_rank = 0
        world_size = 1

    print_log(f"Rank: {rank} / Local rank: {local_rank} / World size: {world_size}", logger="current")
    return rank, local_rank, world_size
