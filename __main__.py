from .train import train
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("action", type=str, choices=["train"])
    parser.add_argument("--train-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.action == "train":
        train(args.train_batch_size)


if __name__ == "__main__":
    main()
