import sys
import getopt
import os.path
import pathlib

from .train import train


def generate_experiment(argv: list[str]):

    current_dir = str(pathlib.Path(__file__).parent.resolve())
    exp_path = current_dir + '/experiments/' + argv
    variables = {}
    exec(open(exp_path + '/config.py').read(), variables)

    model = variables['MODEL']
    epochs = variables['EPOCHS']
    train_batch_size = variables['TRAIN_BATCH_SIZE']
    test_batch_size = variables['TEST_BATCH_SIZE']
    learning_rate = variables['LEARNING_RATE']
    task = variables['TASK']
    lr_step = variables['LR_STEP']
    gamma = variables['GAMMA']
    plot = variables['PLOT']
    criterion = variables['CRITERION']
    optimizer = variables['OPTIMIZER'](model.parameters(), lr=learning_rate)

    history, test_loss = train(
        model=model,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        lr=learning_rate,
        task=task,
        lr_step=lr_step,
        gamma=gamma,
        plot=plot,
        criterion=criterion,
        optimizer=optimizer
    )


if __name__ == "__main__":
    print(generate_experiment(sys.argv[1:]))