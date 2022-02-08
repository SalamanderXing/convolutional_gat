import sys
import getopt
import os.path
import pathlib
import json
import ipdb
from .train import train


def generate_experiment(argv: list[str]):

    current_dir = str(pathlib.Path(__file__).parent.resolve())
    exp_path = current_dir + "/experiments/" + argv
    variables = {}

    exec(open(exp_path + "/config.py").read(), variables)
    variables["OUTPUT_PATH"] = exp_path
    print(
        json.dumps(
            {
                k: v
                for k, v in variables.items()
                if k.isupper() and not k.startswith("__")
            },
            indent=4,
            default=str,
        )
    )
    variables = {k.lower(): v for k, v in variables.items() if k.isupper()}
    """
    model_type = variables["MODEL_TYPE"]
    mapping_type = variables["MAPPING_TYPE"]
    epochs = variables["EPOCHS"]
    train_batch_size = variables["TRAIN_BATCH_SIZE"]
    test_batch_size = variables["TEST_BATCH_SIZE"]
    learning_rate = variables["LEARNING_RATE"]
    preprocessed_folder = variables["PREPROCESSED_FOLDER"]
    dataset = variables["DATASET"]
    lr_step = variables["LR_STEP"]
    gamma = variables["GAMMA"]
    plot = variables["PLOT"]
    criterion = variables["CRITERION"]
    optimizer_class = variables["OPTIMIZER"]
    downsample_size = variables["DOWNSAMPLE_SIZE"]
    """
    history = train(**variables)


if __name__ == "__main__":
    print(generate_experiment(sys.argv[1:]))
