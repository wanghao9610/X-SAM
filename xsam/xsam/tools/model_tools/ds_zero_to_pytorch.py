import argparse
import os.path as osp

from xtuner.utils.zero_to_any_dtype import convert_zero_checkpoint_to_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--tag_dir", type=str, required=False, default=None)
    args = parser.parse_args()

    if args.tag_dir is None:
        last_checkpoint = osp.join(args.work_dir, "last_checkpoint")
        if osp.exists(last_checkpoint):
            with open(last_checkpoint, "r") as f:
                tag_dir = f.read().strip()
        else:
            raise ValueError(f"Last checkpoint file {last_checkpoint} does not exist")
    else:
        tag_dir = args.tag_dir

    output_file = osp.join(args.work_dir, "pytorch_model.bin")

    convert_zero_checkpoint_to_state_dict(args.work_dir, output_file, tag=tag_dir)


if __name__ == "__main__":
    main()
