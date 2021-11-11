from argparse import ArgumentParser
import json
from .extract_feature import preprocess


def main():
    parser = ArgumentParser()
    parser.add_argument("--in-path", type=str, default="./data")
    parser.add_argument("--out-path", type=str, default="./preprocessed")
    parser.add_argument(
        "--select-variables", type=str, default='[["CTTH", "temperature"]]'
    )
    args = parser.parse_args()
    preprocess(
        in_path=args.in_path,
        out_path=args.out_path,
        select_variables=json.loads(args.select_variables),
    )


if __name__ == "__main__":
    main()
