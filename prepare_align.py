import argparse

import yaml

from preprocessor import aishell3, libritts, qi


def main(config):
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
