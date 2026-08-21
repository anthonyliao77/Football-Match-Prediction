"""
Runs the training and evaluation of a football prediction model.
"""

import argparse

from src.training import train_model


def parse_arguments():
    """
    Parse command-line arguments for the football prediction model
    training script.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate a football prediction model."
    )

    parser.add_argument(
        "--league",
        required=True,
        help="League to train the model for."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train_model(args.league)
