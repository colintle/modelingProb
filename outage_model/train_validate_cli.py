import click
import torch
import random
import pickle
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

from .util.functions import normalizeDataset, trainValidate, validateGAT
from .util.GATRNN import GATRNN

@click.group(name="train-validate")
def TRAIN_VALIDATE():
    pass

@TRAIN_VALIDATE.command()
@click.option('--pkl-file', type=click.Path(exists=True), default="dataDict.pkl", help='Path to the pickle file containing datasets.')
@click.option('--output-model-file', type=click.Path(), default="gatrnn_model.pth", help='Path to save the trained model as .pth file')
@click.option('--epochs', type=int, default=1500, help='Number of training epochs.')
@click.option('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
@click.option('--optimizer', type=click.Choice(['adam', 'adamax', 'sgd'], case_sensitive=True), multiple=False, required=True, help='Optimizer for modeling')
@click.option('--hidden-size', type=int, default=40, help='Hidden size for the model.')
def train_validate(pkl_file, output_model_file, epochs, learning_rate, optimizer, hidden_size):
    """Train and validate the GATRNN model with the given parameters."""

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_model_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pkl_file, "rb") as f:
        datasets = pickle.load(f)

    datasets_t = datasets['train']
    datasets_v = datasets['validate']
    results = datasets["sF"]

    feature_names = {
        'node_static_features': datasets["node_static_features"],
        'edge_static_features': datasets["edge_static_features"],
        'node_dynamic_features': datasets["node_dynamic_features"],
    }

    nDatasets_t = normalizeDataset(datasets_t, results, feature_names)
    nDatasets_v = normalizeDataset(datasets_v, results, feature_names)

    validation_scale = float(len(datasets['train'])) / float(len(datasets['validate']))

    criterion = torch.nn.MSELoss()

    model = GATRNN(
        num_static_node_features=len(datasets["node_static_features"]), 
        num_static_edge_features=len(datasets["edge_static_features"]),
        num_weather_features=len(datasets["node_dynamic_features"]), 
        hidden_size=hidden_size
    )
    model.to(device)
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_loss, val_loss = trainValidate(
        model, optimizer, criterion, device, nDatasets_t, nDatasets_v, results, 
        epochs=epochs, validation_scale=validation_scale
    )

    torch.save(model.state_dict(), output_model_file)

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20)) = plt.subplots(4, 5, constrained_layout=True)
    figa, ((ax1a, ax2a, ax3a, ax4a, ax5a), (ax6a, ax7a, ax8a, ax9a, ax10a), (ax11a, ax12a, ax13a, ax14a, ax15a), (ax16a, ax17a, ax18a, ax19a, ax20a)) = plt.subplots(4, 5, constrained_layout=True)
 
   
    # Create a figure and an axis with constrained layout for better spacing
   
 
    # Create a custom colormap from green to red
    green_red_colormap = LinearSegmentedColormap.from_list('GreenRed', ['green', 'red'])
 
    for i, dataset in enumerate(nDatasets_v):
       
        node_static_feats = dataset['node_static_features'].to(device)
        edge_static_feats = dataset['edge_static_features'].to(device)
        node_dynamic_feats = dataset['node_dynamic_features'].to(device)
        edge_index = dataset['edge_index'].to(device)
        targets = dataset['targets'].to(device)
        G = nx.MultiDiGraph()
        G.add_edges_from(edge_index.cpu().detach().numpy())
       
        maxWeather, maxWeatherIndices = torch.max(node_dynamic_feats,1)
       
        newNodeStatic = torch.cat((node_static_feats,maxWeather), dim=1)
 
        output, vloss = validateGAT(model, node_static_feats, edge_static_feats, node_dynamic_feats, edge_index, targets, optimizer, criterion, float(results['rangeProb']))
        #output, vloss = validateGAT(model, newNodeStatic, edge_static_feats, node_dynamic_feats, edge_index, targets, optimizer, criterion, float(results['rangeProb']))
 
        pos = {i: tuple(dataset['coordinates'][i]) for i in range(len(output))}
        # Map each probability to a color in the colormap and store these colors
        nodeColorsModel = [green_red_colormap(prob) for prob in output.cpu()]
        nodeColorsActual = [green_red_colormap(prob) for prob in targets.cpu()]
        # Draw the tree graph with customized node colors and styles
        nx.draw(G, pos=pos, ax=locals()['ax'+str(i+1)],with_labels=False, node_color=nodeColorsModel, node_size=80,
                 arrowstyle='fancy', arrows=False, font_size=12)
       
        nx.draw(G, pos=pos, ax=locals()['ax'+str(i+1)+'a'],with_labels=False, node_color=nodeColorsActual, node_size=80,
                 arrowstyle='fancy', arrows=False, font_size=12)
 
        scalarmappaple = plt.cm.ScalarMappable(cmap=green_red_colormap, norm=plt.Normalize(vmin=0, vmax=1))
        # Add a colorbar to the axis using the ScalarMappable, and set its label
        cbar = fig.colorbar(scalarmappaple, ax=locals()['ax'+str(i+1)])
        cbara = figa.colorbar(scalarmappaple, ax=locals()['ax'+str(i+1)+'a'])
 
        cbar.set_label('Probability of an Outage')
        cbara.set_label('Probability of an Outage')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epochs')
    plt.legend()
    plt.show()

        

if __name__ == "__main__":
    TRAIN_VALIDATE()
