import click
import torch
import numpy as np
import os
import csv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from .util.functions import findStats
import h5py
from collections import defaultdict
import re

@click.group(name="preprocess")
def PREPROCESS():
    pass

@PREPROCESS.command()
@click.option('--dataset-file', type=click.Path(exists=True), required=True, help='Path to the datasets.hdf5 containing the model data.')
@click.option('--edge-static-features', type=str, multiple=True, required=True, help='Static features of the edges.')
@click.option('--node-static-features', type=str, multiple=True, required=True, help='Static features of the nodes.')
@click.option('--weather-features', type=click.Choice(['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco'], case_sensitive=True), multiple=True, required=True, help='Weather features to process.')
@click.option('--output', type=click.Path(), required=True, help='Output path to save both the CSV file and pickle file.')
def preprocess(dataset_file, edge_static_features, node_static_features, weather_features, output):
    """Preprocess the weather and static data for model training."""

    if len(edge_static_features) != len(set(edge_static_features)):
        raise click.ClickException("Error: There are duplicate entries in edge_static_features.")
    
    if len(node_static_features) != len(set(node_static_features)):
        raise click.ClickException("Error: There are duplicate entries in node_static_features.")

    # Ensure the output directory exists
    if not os.path.exists(output):
        os.makedirs(output)

    # To find each dataset and its associated weather features
    dataset_names = []
    with h5py.File(dataset_file, 'r') as f:
        for group in f:
            case_study_group = f[group]
            dataframes = list(case_study_group.keys())
            grouped_data = defaultdict(list)

            for df in dataframes:
                match = re.search(r'_(\d+)$', df)
                if match:
                    key = int(match.group(1))
                    grouped_data[key].append(f"{group}/{df}")

            # Now, append the grouped data to dataset_names
            for key, datasets in grouped_data.items():
                dataset_names.append(datasets)

    datasets = []

    for index, dataset in enumerate(dataset_names):

        edges_physical_features = None
        edges_weather_features = []

        node_physical_features = None
        nodes_weather_features = []

        # Iterate through the dataset names
        for name in dataset:
            if 'edges_features' in name:
                edges_physical_features = name
            elif 'node_features' in name:
                node_physical_features = name
            elif 'edges' in name:
                edges_weather_features.append(name)
            elif 'nodes' in name:
                nodes_weather_features.append(name)
        
        nL = pd.read_hdf(dataset_file, key=node_physical_features)
        eL = pd.read_hdf(dataset_file, key=edges_physical_features)

        sourceList = eL['source'].to_numpy()
        targetList = eL['target'].to_numpy()

        edgeList = np.zeros((len(sourceList), 2))
        edgeList[:, 0] = sourceList
        edgeList[:, 1] = targetList

        nodeStaticFeatures = np.zeros((len(nL), len(node_static_features)))
        edgeStaticFeatures = np.zeros((len(eL), len(edge_static_features)))

        for index, feature in enumerate(edge_static_features):
            edgeStaticFeatures[:, index] = eL[feature].to_numpy()

        for index, feature in enumerate(node_static_features):
            nodeStaticFeatures[:, index] = nL[feature].to_numpy()

        targets = nL['Probability'].to_numpy()

        if len(nodes_weather_features) == 0:
            continue
        
        first = pd.read_hdf(dataset_file, key=nodes_weather_features[0])
        nodeDynamicFeatures = np.zeros((len(nL), len(first.columns), len(nodes_weather_features)))

        for index, ts in enumerate(weather_features):
            for node_weather_feature in nodes_weather_features:
                if ts in weather_features:
                    data = pd.read_hdf(dataset_file, key=node_weather_feature)
                    nodeDynamicFeatures[:, :, index] = data
        
        nodeCoords = nL['coords'].to_numpy()
        # Convert the strings to tuples of floats
        coordinates = [tuple(map(float, coord.strip('()').split(', '))) for coord in nodeCoords]
 
        # Convert the list of tuples to a 2D numpy array
        nodeCoords = np.array(coordinates)

        dataset = {
            'scenario': index,
            'edge_index': torch.tensor(edgeList, dtype=torch.long),
            'node_static_features': torch.tensor(nodeStaticFeatures),
            'edge_static_features': torch.tensor(edgeStaticFeatures),
            'node_dynamic_features': torch.tensor(nodeDynamicFeatures),
            'targets': torch.tensor(targets, dtype=torch.float),
            'coordinates': nodeCoords
        }

        datasets.append(dataset)

    # Split datasets into training and validation sets
    train_datasets, validate_datasets = train_test_split(datasets, test_size=0.2, random_state=42)

    # Aggregate data for statistics
    nodeData = {f: [] for f in node_static_features}
    edgeData = {f: [] for f in edge_static_features}
    weatherData = {f: [] for f in weather_features}
    probabilities = []

    for key in datasets:
        for index, feature in enumerate(node_static_features):
            nodeData[feature].extend(key['node_static_features'][:, index].numpy().ravel())

        for index, feature in enumerate(edge_static_features):
            edgeData[feature].extend(key['edge_static_features'][:, index].numpy().ravel())

        for index, feature in enumerate(weather_features):
            weatherData[feature].extend(key["node_dynamic_features"][:, :, index].numpy().ravel())

        probabilities.extend(key['targets'].numpy().ravel())

    results = {}

    for feature, data in nodeData.items():
        mean_val, max_val, min_val = findStats(data)
        results[f"minNode{feature}"] = min_val
        results[f"rangeNode{feature}"] = max_val - min_val

    for feature, data in edgeData.items():
        mean_val, max_val, min_val = findStats(data)
        results[f"minEdge{feature}"] = min_val
        results[f"rangeEdge{feature}"] = max_val - min_val

    for feature, data in weatherData.items():
        mean_val, max_val, min_val = findStats(data)
        results[f"min{feature}"] = min_val
        results[f"range{feature}"] = max_val - min_val

    mean_val, max_val, min_val = findStats(probabilities)
    results["minProb"] = min_val
    results["rangeProb"] = max_val - min_val

    # Save statistics to CSV
    csv_filename = os.path.join(output, "sF_NE_dict.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, val in results.items():
            writer.writerow([key, val])

    # Save datasets to pickle file
    datasetDict = {
        'train': train_datasets,
        'validate': validate_datasets,
        'sF': results,
        'edge_static_features': edge_static_features,
        'node_static_features': node_static_features,
        'node_dynamic_features': weather_features
    }

    pkl_filename = os.path.join(output, "dataDict.pkl")
    with open(pkl_filename, 'wb') as f:
        pickle.dump(datasetDict, f)

if __name__ == "__main__":
    PREPROCESS()
