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

        if len(nodes_weather_features) == 0:
            continue
        
        nodeDynamicFeatures = np.zeros((len(nL), len(nodes_weather_features)))
        edgeDynamicFeatures = np.zeros((len(eL), len(edges_weather_features)))

        for index, ts in enumerate(weather_features):
            for node_weather_feature in nodes_weather_features:
                if ts in node_weather_feature:
                    data = pd.read_hdf(dataset_file, key=node_weather_feature)
                    nodeDynamicFeatures[:, index] = np.max(data,1)
        
        for index, ts in enumerate(weather_features):
            for edge_weather_feature in edges_weather_features:
                if ts in edge_weather_feature:
                    data = pd.read_hdf(dataset_file, key=edge_weather_feature)
                    edgeDynamicFeatures[:, index] = np.max(data,1)
        
        nodeCoords = nL['coords'].to_numpy()
        coordinates = [tuple(map(float, coord.strip('()').split(', '))) for coord in nodeCoords]
        nodeCoords = np.array(coordinates)

        # [Vegetation, Elevation, FloodZone, Max Wind, Max Rain]
        inputNode = np.concatenate([nodeStaticFeatures,nodeDynamicFeatures],1)
        
        # [Vegetation, Length, Max Wind, Max Rain]
        inputEdge = np.concatenate([edgeStaticFeatures,edgeDynamicFeatures],1)

        nodeDamage = nL['Unmodified Probability'].to_numpy()
        edgeDamage = eL['Unmodified Probability'].to_numpy()
        
        dataset = {
            'scenario': index,
            'node_damage_input': inputNode,
            'node_damage_output':nodeDamage,
            'edge_damage_input':inputEdge,
            'edge_damage_output':edgeDamage,
            'nodeList':nL,
            'edgeList':eL
        }

        datasets.append(dataset)

    for i, dataset in enumerate(datasets):
        if i == 0:
            totalNodeInput = dataset['node_damage_input']
            totalEdgeInput = dataset['edge_damage_input']
            totalNodeOutput = dataset['node_damage_output']
            totalEdgeOutput = dataset['edge_damage_output']
        else:
            totalNodeInput = np.concatenate([totalNodeInput,dataset['node_damage_input']],0)
            totalEdgeInput = np.concatenate([totalEdgeInput,dataset['edge_damage_input']],0)
            totalNodeOutput = np.concatenate([totalNodeOutput,dataset['node_damage_output']])
            totalEdgeOutput = np.concatenate([totalEdgeOutput,dataset['edge_damage_output']])
    
    # NDI: Node Damage Input, NDO: Node Damage Output
    NDI_train, NDI_test, NDO_train, NDO_test = train_test_split(totalNodeInput, totalNodeOutput, test_size=0.25, random_state=42)
    EDI_train, EDI_test, EDO_train, EDO_test = train_test_split(totalEdgeInput, totalEdgeOutput, test_size=0.25, random_state=42)

    datasetDict = {
        "NDI_train": NDI_train,
        "NDI_test": NDI_test,
        "NDO_train": NDO_train,
        "NDO_test": NDO_test,
        "EDI_train": EDI_train,
        "EDI_test": EDI_test,
        "EDO_train": EDO_train,
        "EDO_test": EDO_test
    }

    pkl_filename = os.path.join(output, "dataDict.pkl")
    with open(pkl_filename, 'wb') as f:
        pickle.dump(datasetDict, f)

if __name__ == "__main__":
    PREPROCESS()
