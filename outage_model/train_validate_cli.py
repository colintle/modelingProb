import click
import torch
import random
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

@click.group(name="train-validate")
def TRAIN_VALIDATE():
    pass

@TRAIN_VALIDATE.command()
@click.option('--pkl-file', required=True, type=click.Path(exists=True), help='Path to the pickle file containing datasets.')
@click.option('--node-regressor-folder', required=True, type=click.Path(exists=True), help='Path to save the trained node damage regressor.')
@click.option('--edge-regressor-folder', required=True, type=click.Path(exists=True), help='Path to save the trained edge damage regressor.')
def train_validate(pkl_file, node_regressor_folder, edge_regressor_folder):
    """Train and validate the GATRNN model with the given parameters."""

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    node_damage_regressor = RandomForestRegressor(random_state=0)
    edge_damage_regressor = RandomForestRegressor(random_state=0)

    with open(pkl_file, "rb") as f:
        datasets = pickle.load(f)

    NDI_train, NDI_test, NDO_train, NDO_test = datasets["NDI_train"], datasets["NDI_test"], datasets["NDO_train"], datasets["NDO_test"]
    EDI_train, EDI_test, EDO_train, EDO_test = datasets["EDI_train"], datasets["EDI_test"], datasets["EDO_train"], datasets["EDO_test"]

    node_damage_regressor.fit(NDI_train, NDO_train)
    edge_damage_regressor.fit(EDI_train, EDO_train)

    node_validation_score = node_damage_regressor.score(NDI_test, NDO_test)
    edge_validation_score = edge_damage_regressor.score(EDI_test, EDO_test)

    print("Node validation score:", node_validation_score)
    print("Edge validation score:", edge_validation_score)

    # Save the regressors
    with open(os.path.join(node_regressor_folder, "node_damage_regressor.pkl"), 'wb') as f:
        pickle.dump(node_damage_regressor, f)

    with open(os.path.join(edge_regressor_folder, "edge_damage_regressor.pkl"), 'wb') as f:
        pickle.dump(edge_damage_regressor, f)

if __name__ == "__main__":
    TRAIN_VALIDATE()
