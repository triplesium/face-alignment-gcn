import os
import argparse
from pathlib import Path
import yaml
from pprint import pprint
from fld import FLD
import torchinfo
from calflops import calculate_flops


def parse_args():
    """
    parse args
    :return:args
    """
    new_parser = argparse.ArgumentParser(
        description="PyTorch Facial Landmark Detector parser.."
    )
    new_parser.add_argument("--config", default="")
    new_parser.add_argument("--load_path", type=str, default=None)
    new_parser.add_argument("--resume", action="store_true")
    new_parser.add_argument("--expname", type=str, default=None)
    new_parser.add_argument("--visualize", action="store_true")
    # exclusive arguments
    group = new_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--eval_ckpts", action="store_true")
    group.add_argument("--info", action="store_true")

    return new_parser.parse_args()


def main():
    # parse args and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.expname is None:
        args.expname = Path(args.config).stem

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    agent = FLD(config)

    if args.evaluate:
        agent.evaluate()
    elif args.eval_ckpts:
        agent.eval_ckpts()
    elif args.train:
        agent.train()
    elif args.info:
        torchinfo.summary(
            agent.model,
            input_size=(
                1,
                3,
                config["common"]["crop_size"],
                config["common"]["crop_size"],
            ),
        )
        flops, macs, params = calculate_flops(
            agent.model,
            input_shape=(
                1,
                3,
                config["common"]["crop_size"],
                config["common"]["crop_size"],
            ),
            print_results=False,
        )
        print(f"Flops: {flops}, Macs: {macs}, Params: {params}")
    else:
        raise Warning("Invalid Args")


if __name__ == "__main__":
    main()
