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
    model_class = variables["MODEL"]
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
    binarize_thresh = variables["BINARIZE_THRESH"]

    history, test_loss = train(
        model_class=model_class,
        mapping_type=mapping_type,
        optimizer_class=optimizer_class,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        lr=learning_rate,
        lr_step=lr_step,
        gamma=gamma,
        plot=plot,
        criterion=criterion,
        downsample_size=downsample_size,
        output_path=exp_path,
        preprocessed_folder=preprocessed_folder,
        dataset=dataset,
        binarize_thresh=binarize_thresh
    )


if __name__ == "__main__":
    print(generate_experiment(sys.argv[1:]))
