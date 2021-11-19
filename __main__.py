from .train import train
from .generate_experiment import generate_experiment
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("action", type=str, choices=["train", "generate_experiment"])
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--exp_folder_name", type=str)
    args = parser.parse_args()
    if args.action == "train":
        train(args.train_batch_size)
    if args.action == "generate_experiment":
        generate_experiment(args.exp_folder_name)


if __name__ == "__main__":
    main()
