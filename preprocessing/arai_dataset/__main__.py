from argparse import ArgumentParser
import json
from .preprocessing import preprocess


def main():
    parser = ArgumentParser()
    parser.add_argument("--in-path", type=str, default="./data")
    parser.add_argument("--out-path", type=str, default="/mnt/preprocessed3")
    parser.add_argument(
        "--select-variables",
        type=str,
        default='[["ASII", "asii_turb_trop_prob"]]',  # default='[["CRR", "crr"]]'
    )
    args = parser.parse_args()
    select_variables = tuple(
        (str(x[0]), str(x[1])) for x in json.loads(args.select_variables)
    )
    preprocess(
        in_path=args.in_path, out_path=args.out_path, select_variables=select_variables,
    )


if __name__ == "__main__":
    main()
