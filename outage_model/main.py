import click

from . import preprocess_cli
from . import train_validate_cli
from . import evaluate_model_cli

cli = click.CommandCollection(sources=[
    preprocess_cli.PREPROCESS,
    train_validate_cli.TRAIN_VALIDATE,
    evaluate_model_cli.EVALUATE_GATRNN
])

if __name__ == "__main__":
    cli()