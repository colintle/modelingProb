import click
import torch
import random
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from .util.functions import normalizeDataset, validateGAT
from .util.GATRNN import GATRNN

@click.group(name="evaluate-gatrnn")
def EVALUATE_GATRNN():
    """A CLI group for evaluating GATRNN models."""
    pass

@EVALUATE_GATRNN.command()
@click.option('--pkl-file', type=click.Path(exists=True), default="dataDict.pkl", help='Path to the pickle file containing datasets.')
@click.option('--model-file', type=click.Path(exists=True), default="gatrnn_model.pth", help='Path to the trained model file.')
def evaluate_model(pkl_file, model_file):
    """Evaluate the GATRNN model to find the best and worst RMSE among validation cases."""

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(pkl_file, "rb") as f:
        datasets = pickle.load(f)

    datasets_v = datasets['validate']
    results = datasets["sF"]

    feature_names = {
        'node_static_features': datasets["node_static_features"],
        'edge_static_features': datasets["edge_static_features"],
        'node_dynamic_features': datasets["node_dynamic_features"],
    }

    nDatasets_v = normalizeDataset(datasets_v, results, feature_names)

    model = GATRNN(
        num_static_node_features=len(datasets["node_static_features"]),
        num_static_edge_features=len(datasets["edge_static_features"]),
        num_weather_features=len(datasets["node_dynamic_features"]),
        hidden_size=40
    )
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    rmse_list = []
    for i, dataset in enumerate(nDatasets_v):
        node_static_feats = dataset['node_static_features'].to(device)
        edge_static_feats = dataset['edge_static_features'].to(device)
        node_dynamic_feats = dataset['node_dynamic_features'].to(device)
        edge_index = dataset['edge_index'].to(device)
        targets = dataset['targets'].to(device)

        with torch.no_grad():
            outputs, _ = validateGAT(
                model,
                node_static_feats,
                edge_static_feats,
                node_dynamic_feats,
                edge_index,
                targets,
                None,
                torch.nn.MSELoss(),
                float(results['rangeProb'])
            )

        rmse = sqrt(mean_squared_error(targets.cpu().numpy(), outputs.cpu().numpy()))
        rmse_list.append((i, rmse))

    sorted_rmse_list = sorted(rmse_list, key=lambda x: x[1])

    click.echo("Sorted RMSE values (from lowest to highest):")
    for case, rmse in sorted_rmse_list:
        click.echo(f"Case {case + 1}: RMSE = {rmse:.4f}")
    
    avg_rmse = sum(rmse for _, rmse in rmse_list) / len(rmse_list)
    click.echo(f"Average RMSE across all validation cases: {avg_rmse:.4f}")

    best_case_index = sorted_rmse_list[0][0]
    worst_case_index = sorted_rmse_list[-1][0]
    def plot_graph(ax, G, pos, probabilities, title, colormap):
        node_colors = [colormap(prob) for prob in probabilities]
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            node_size=80,
            arrowstyle='fancy',
            arrows=False,
            font_size=10,
        )
        ax.set_title(title, fontsize=20, pad=20)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rows = ['Actual Probabilities', 'Predicted Probabilities']
    cols = ['Best Case', 'Worst Case']

    for ax, row_label in zip(axes[:, 0], rows):
        ax.annotate(
            row_label,
            xy=(-0.1, 0.5),
            xycoords="axes fraction",
            fontsize=20,
            ha="center",
            va="center",
            rotation=90,
        )

    green_red_colormap = LinearSegmentedColormap.from_list('GreenRed', ['green', 'red'])
    norm = Normalize(vmin=0, vmax=1)

    for col, (case_index, rmse) in enumerate([(best_case_index, sorted_rmse_list[0][1]), (worst_case_index, sorted_rmse_list[-1][1])]):
        dataset = nDatasets_v[case_index]
        node_static_feats = dataset['node_static_features'].to(device)
        edge_static_feats = dataset['edge_static_features'].to(device)
        node_dynamic_feats = dataset['node_dynamic_features'].to(device)
        edge_index = dataset['edge_index'].to(device)
        targets = dataset['targets'].to(device)

        G = nx.MultiDiGraph()
        G.add_edges_from(edge_index.cpu().detach().numpy())

        with torch.no_grad():
            outputs, _ = validateGAT(
                model,
                node_static_feats,
                edge_static_feats,
                node_dynamic_feats,
                edge_index,
                targets,
                None,
                torch.nn.MSELoss(),
                float(results['rangeProb'])
            )

        pos = {i: tuple(dataset['coordinates'][i]) for i in range(len(outputs))}

        if col == 0:
            plot_graph(
                axes[0, col], G, pos, targets.cpu().numpy(),
                title=f"Best Case", colormap=cm.ScalarMappable(norm=norm, cmap=green_red_colormap).cmap
            )

            plot_graph(
                axes[1, col], G, pos, outputs.cpu().numpy(),
                title=f"RMSE: {rmse:.4f}", colormap=cm.ScalarMappable(norm=norm, cmap=green_red_colormap).cmap
            )
        else:
            plot_graph(
                axes[0, col], G, pos, targets.cpu().numpy(),
                title=f"Worst Case", colormap=cm.ScalarMappable(norm=norm, cmap=green_red_colormap).cmap
            )

            plot_graph(
                axes[1, col], G, pos, outputs.cpu().numpy(),
                title=f"RMSE: {rmse:.4f}", colormap=cm.ScalarMappable(norm=norm, cmap=green_red_colormap).cmap
            )

    scalarmappable = cm.ScalarMappable(norm=norm, cmap=green_red_colormap)
    cbar = fig.colorbar(scalarmappable, ax=axes[:, -1], orientation='vertical', shrink=0.5)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Probability of an Outage', fontsize=12)

    plt.show()


if __name__ == "__main__":
    EVALUATE_GATRNN()
