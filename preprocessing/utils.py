import os


def listdir(path: str):
    return [
        (subpath, os.path.join(path, subpath)) for subpath in sorted(os.listdir(path))
    ]


def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
