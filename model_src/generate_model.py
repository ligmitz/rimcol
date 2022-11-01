#!/usr/bin/env python3

import argparse
import sys
from os import mkdir, path


def generate(
    name: str,
    test_input: str,
    test_output: str,
    train_input: str,
    batch_size: int,
    epochs: int,
) -> None:
    if path.isdir(f"model_src/models/{name}"):
        response = input(
            "Model file with same name already exists, do you want to continue? Y/N:"
        )
        if response.upper() != "Y":
            print("Exiting the process!")
            sys.exit(1)

    if not path.isdir(test_input):
        print("Test input directory does not exist!")
        sys.exit(1)

    if not path.isdir(train_input):
        print("Train input directory does not exist!")
        sys.exit(1)

    if not path.isdir(test_output):
        mkdir(test_output)

    try:
        from model_definition import ColorizationModel

        model = ColorizationModel(name, test_input, test_output, train_input, batch_size, epochs)
        model.run()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the model")
    parser.add_argument(
        "--test-input-dir",
        help="The directory containing input test images",
        default="model_src/test_input/",
    )
    parser.add_argument(
        "--test-output-dir",
        help="The directory containing test images output",
        default="model_src/test_output/",
    )
    parser.add_argument(
        "--train_dir",
        help="The directory containing training images",
        default="model_src/train/",
    )
    parser.add_argument(
        "--batch_size",
        help="The batch size for training and testing model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        help="Number of times neural network is trained",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    generate(
        args.name,
        args.test_input_dir,
        args.test_output_dir,
        args.train_dir,
        args.batch_size,
        args.epochs,
    )
