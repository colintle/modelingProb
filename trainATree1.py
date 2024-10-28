import click
import torch
import numpy as np
import os
import csv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
from collections import defaultdict
import re
import h5py
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def plotTreeWithProb(tree, probabilities, title, pos):
    """
    Function to visualize the network graph with nodes colored based on their probability of outage, providing a visual tool for assessing network vulnerability.

    Args:
        tree (nx.DiGraph): Directed graph representing the network topology.
        probabilities (List[float]): List of probabilities associated with each node in the graph.
        title (str): Title for the plotted graph.
        pos (Dict[int, Tuple[float, float]]): Dictionary mapping node indices to their positions for plotting.
    Returns:
        Displays a visual representation of the network graph with nodes colored according to their outage probabilities.
    """
        
    # Create a figure and an axis with constrained layout for better spacing
    fig, ax = plt.subplots(constrained_layout=True)

    # Create a custom colormap from green to red
    green_red_colormap = LinearSegmentedColormap.from_list('GreenRed', ['green', 'red'])

    # Map each probability to a color in the colormap and store these colors
    nodeColors = [green_red_colormap(prob) for prob in probabilities]
    # Draw the tree graph with customized node colors and styles
    nx.draw(tree, pos=pos, ax=ax,with_labels=False, node_color=nodeColors, node_size=80, 
            arrowsize=7, arrowstyle='fancy', arrows=False, font_size=12)

    # Create a ScalarMappable to interpret the colormap scale properly
    scalarmappaple = plt.cm.ScalarMappable(cmap=green_red_colormap, norm=plt.Normalize(vmin=0, vmax=1))
    # Add a colorbar to the axis using the ScalarMappable, and set its label
    cbar = fig.colorbar(scalarmappaple, ax=ax)
    cbar.set_label('Probability of an Outage')

    # Set the title of the plot
    plt.title(title)
    # Display the plot
    plt.savefig(f"C:\\Users\\co986387\\Documents\\outage_map_images\\{title}.png")

def prod(iterable):
    """
    Function that computes the product of all elements in a given list, primarily used within the inclusionExclusion function.

    Args:
        iterable (Iterable[float]): Iterable of floats whose product is to be calculated.
    Returns:
        float: The product of the elements.
    """

    result = 1
    for i in iterable:
        result *= i
    return result

# Function to sort the list of features by importance given fromt the random forest
def sort_by_importance(labels, importances):
    # Combine labels and importances into a list of tuples
    combined = list(zip(labels, importances))
    
    # Sort the combined list by the importance values in descending order
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    
    # Unpack the sorted labels and importances
    sorted_labels = [x[0] for x in combined_sorted]
    sorted_importances = [x[1] for x in combined_sorted]
    
    return sorted_labels, sorted_importances


def probOfNodeAndParent(probN, probE, graph):
    """
    Function that aggregates the probabilities of outages for nodes and their parent nodes in the network graph, taking into account the dependencies due to network topology.

    Args:
        probN (List[List[float]]): List containing probabilities of outage for nodes.
        probE (List[List[float]]): List containing probabilities of outage for edges.
        graph (defaultdict[list]): Graph structure representing the network, where keys are parent node indices and values are lists of child node and edge index pairs.
    Returns:
        List[List[float]]: Aggregated probabilities of outages for nodes considering their parent node dependencies.
    """

    # Create a new list of probability ranges by copying from the provided probN list
    #newProb = [[low, high] for low, high in probN]
    newProb = [prob for prob in probN]

    # Initialize a deque for breadth-first search traversal of the graph
    queue = deque()
    # Start with the root node (assumed to be node 0)
    queue.append(0)
    # Continue until there are no more nodes to process
    while queue:
        # Retrieve the next node to process from the queue
        parent = queue.popleft()
        # print(f"Parent {parent}")
        # print(f"Parent {parent}")
        # Iterate over all children connected to the current parent node
        for child, edge in graph[parent]:
            # print(f"Child {child}")
            # print(f"Connected by edge {edge}")
            # Apply the inclusion-exclusion principle to update probability ranges
            newProb[child] = inclusionExclusion([newProb[parent], newProb[child], probE[edge]])
            # Add the child to the queue to process its own children later
            queue.append(child)

    # Return the updated probability ranges for all nodes
    return newProb
def inclusionExclusion(pr):
    """
    Function to calculate the probability of the union of events using the inclusion-exclusion principle based on provided probabilities of individual events.

    Args:
        pr (List[float]): List of probabilities for individual events.
    Returns:
        float: Probability of the union of the events after applying the inclusion-exclusion principle.
    """

    # Initialize the variable `union` to accumulate the union probability
    union = 0

    # Loop over all possible non-empty subsets of probabilities
    for i in range(1, len(pr) + 1):
        # Generate all combinations of `i` probabilities from the list `pr`
        comb = combinations(pr, i)
        # Calculate the sum of products of probabilities for each combination
        sum_of_probs = sum([prod(combination) for combination in comb])
        # Add to or subtract from the union based on whether the number of elements in the combination is odd or even
        # This is based on the inclusion-exclusion principle
        union += sum_of_probs if i % 2 != 0 else -sum_of_probs

    # Return the calculated union probability
    return union
# Dataset file name
dataset_file= r"C:\Users\co986387\Documents\test_for_outage\datasets.hdf5"

# Feature Labels
weather_features = ['wspd','prcp']
edge_static_features = ['vegetation','length']
node_static_features = ['vegetation','elevation','flood_zone_num']

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

file_names = [
    "weatherEvent108",
    "weatherEvent109",
    "weatherEvent110",
    "weatherEvent180",
    "weatherEvent330",
    "weatherEvent331",
    "weatherEvent343",
    "weatherEvent344",
    "weatherEvent419",
    "weatherEvent420",
    "weatherEvent421",
    "weatherEvent422",
    "weatherEvent423"
]

maxi = 0
for index, dataset in enumerate(dataset_names):
    case_study = dataset[0].split("/")[0]
    index_event = dataset[0].split("_")[-1]

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

    nodeCoords = nL['coords'].to_numpy()
    # Convert the strings to tuples of floats
    coordinates = [tuple(map(float, coord.strip('()').split(', '))) for coord in nodeCoords]

    # Convert the list of tuples to a 2D numpy array
    nodeCoords = np.array(coordinates);
    
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

    # Probability of Damages from Fragility Curves
    nodeDamage = nL['Unmodified Probability'].to_numpy()
    edgeDamage = eL['Unmodified Probability'].to_numpy()

    if len(nodes_weather_features) == 0:
        continue
    
    first = pd.read_hdf(dataset_file, key=nodes_weather_features[0])
    #nodeDynamicFeatures = np.zeros((len(nL), len(first.columns), len(nodes_weather_features)))
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

    
    # [Vegetation, Elevation, FloodZone, Max Wind, Max Rain]
    inputNode = np.concatenate([nodeStaticFeatures,nodeDynamicFeatures],1)
    
    # [Vegetation, Length, Max Wind, Max Rain]
    inputEdge = np.concatenate([edgeStaticFeatures,edgeDynamicFeatures],1)

    
    dataset = {
        'scenario': index,
        'name': f"{case_study} {file_names[int(index_event)]}",
        'edge_index': torch.tensor(edgeList, dtype=torch.long),
        'node_static_features': torch.tensor(nodeStaticFeatures),
        'edge_static_features': torch.tensor(edgeStaticFeatures),
        'node_dynamic_features': torch.tensor(nodeDynamicFeatures),
        'targets': torch.tensor(targets, dtype=torch.float),
        'coordinates': nodeCoords,
        'node_damage_input': inputNode,
        'node_damage_output':nodeDamage,
        'edge_damage_input':inputEdge,
        'edge_damage_output':edgeDamage,
        'nodeList':nL,
        'edgeList':eL
    }

    datasets.append(dataset)

# Loop through all datasets to create a input and output array consisting of all test cases
# print(datasets[0]["node_damage_output"])
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
    

# Initialize the random forest regrossor models
node_damage_regressor = RandomForestRegressor(random_state=0)
edge_damage_regressor = RandomForestRegressor(random_state=0)

# Split the data into training and testing sets
# NDI: Node Damage Input, NDO: Node Damage Output
NDI_train, NDI_test, NDO_train, NDO_test = train_test_split(totalNodeInput, totalNodeOutput, test_size=0.25, random_state=42)
EDI_train, EDI_test, EDO_train, EDO_test = train_test_split(totalEdgeInput, totalEdgeOutput, test_size=0.25, random_state=42)

print('Starting to Train Node Model')

# Train the node model
node_damage_regressor.fit(NDI_train,NDO_train)

print('Finished Node model, now starting to train Edge model')

# Train the edge model
edge_damage_regressor.fit(EDI_train,EDO_train)


plot_tree(node_damage_regressor.estimators_[0],fontsize=10,feature_names=['Vegetation', 'Elevation', 'FloodZone', 'Max Wind', 'Max Rain'])
plt.show()
print('Finished Edge model')
node_validation_score = node_damage_regressor.score(NDI_test,NDO_test)
edge_validation_score = edge_damage_regressor.score(EDI_test,EDO_test)
# Print the validation score for each model
print(node_validation_score)
print(edge_validation_score)

# Create an array to store the output from the measured data and the models output
Noutputs = np.zeros((2,len(NDO_test)))
Eoutputs = np.zeros((2,len(EDO_test)))

# Assign the measured data to it's respective row
Noutputs[0,:] = NDO_test
Eoutputs[0,:] = EDO_test

# # Predict each node test case in the validation set
# for i in range(len(NDI_test)):
#     Noutputs[1,i] = node_damage_regressor.predict(NDI_test[i,:].reshape(1, -1))[0]

# # Predict each edge test case in the validation set
# for i in range(len(EDI_test)):
#     Eoutputs[1,i] = edge_damage_regressor.predict(EDI_test[i,:].reshape(1, -1))[0]

# Create string arrays with the feature names for plotting
node_feats = ['Vegetation', 'Elevation', 'FloodZone', 'Max Wind', 'Max Rain']
edge_feats = ['Vegetation', 'Length', 'Max Wind', 'Max Rain']

# Grab the importance values for each feature from the regressors
node_damage_feat_importance = node_damage_regressor.feature_importances_
edge_damage_feat_importance = edge_damage_regressor.feature_importances_

# Grab the sorted labels and importances for each model to plot
node_xvals, node_yvals = sort_by_importance(node_feats, node_damage_feat_importance)
edge_xvals, edge_yvals = sort_by_importance(edge_feats, edge_damage_feat_importance)



left, width = .25, .5
bottom, height = .25, .7
right = left + width
top = bottom + height
# Plot the results


# Loop through all datasets to create a input and output array consisting of all test cases
for i, dataset in enumerate(datasets):
        edgeList = []
        # ELD = dataset['eL']
        # ELD['num'] = ELD.astype(int)
        # dataset['eL']['num'] = dataset['eL']['num'].astype(int)
        graph = defaultdict(list)
        for j in range(len(dataset['edgeList'])):
            index = dataset['edgeList'].iloc[j]['num']
            dataset['edgeList'].at[j, 'num'] = index
            edgeList.append((int(dataset['edgeList'].iloc[j]["source"]), int(dataset['edgeList'].iloc[j]["target"])))
            # print((int(dataset['edgeList'].iloc[j]["source"]), int(dataset['edgeList'].iloc[j]["target"])))
            graph[int(dataset['edgeList'].iloc[j]["source"])].append([int(dataset['edgeList'].iloc[j]["target"]), index])
        
        node_damage_input = dataset['node_damage_input']
        edge_damage_input = dataset['edge_damage_input']

        node_damage_output = np.zeros((len(node_damage_input),1))
        edge_damage_output = np.zeros((len(edge_damage_input),1))

        node_list = dataset["nodeList"]
        edge_list = dataset["edgeList"]

        for j in range(len(node_damage_input)):
            index = node_list.iloc[j]["num"]
            node_damage_output[index] = node_damage_regressor.predict(node_damage_input[j,:].reshape(1, -1))[0]
            # print(node_list.iloc[j]["name"], node_damage_output[index])
            # print(node_list['Unmodified Probability'].iloc[j])

        for j in range(len(edge_damage_input)):
            index = edge_list.iloc[j]["num"]
            edge_damage_output[index] = edge_damage_regressor.predict(edge_damage_input[j,:].reshape(1, -1))[0]

        totalNodeInput = dataset['node_damage_input']
        totalEdgeInput = dataset['edge_damage_input']
        totalNodeOutput = dataset['node_damage_output']
        totalEdgeOutput = dataset['edge_damage_output']
        totalNodeInput = np.concatenate([totalNodeInput,dataset['node_damage_input']],0)
        totalEdgeInput = np.concatenate([totalEdgeInput,dataset['edge_damage_input']],0)
        totalNodeOutput = np.concatenate([totalNodeOutput,dataset['node_damage_output']])
        totalEdgeOutput = np.concatenate([totalEdgeOutput,dataset['edge_damage_output']])
        pos = {dataset['nodeList'].iloc[i]["num"]: eval(dataset['nodeList'].iloc[[i]]["coords"][i]) for i in range(len(dataset['nodeList']))}

        prob = probOfNodeAndParent(node_damage_output, edge_damage_output, graph)

        # for p in prob:
        #     print(p[0])

        # for i in range(len(node_damage_output)):
        #     print(f"Node Index {i}")
        #     print(node_list['Unmodified Probability'].iloc[i])
        #     print(node_damage_output[i])

        # print(dataset["targets"])
        # print(prob)
        G = nx.DiGraph()
        G.add_nodes_from(range(len(node_list)))
        G.add_edges_from(sorted(edgeList))
        plotTreeWithProb(G, prob, dataset["name"], pos)
        
# fig, axs = plt.subplots(2,3,constrained_layout=True)
# axs[0,0].scatter(np.linspace(1,len(NDO_test),len(NDO_test)),Noutputs[0,:])
# axs[0,0].scatter(np.linspace(1,len(NDO_test),len(NDO_test)),Noutputs[1,:])
# axs[0,0].set_title('Node Damage Probability Predictions')
# axs[0,0].set_xlabel('Case Number')
# axs[0,0].set_ylabel('Damage Probability')
# axs[0,0].text(0.5 * (left + right), top, 'Accuracy: '+str(node_validation_score),
#         horizontalalignment='center',
#         verticalalignment='top',
#         transform=axs[0,0].transAxes)
# axs[0,0].legend('Actual','Model')

# axs[0,1].scatter(np.linspace(1,len(NDO_test),len(NDO_test)),abs(Noutputs[0,:]-Noutputs[1,:]))
# axs[0,1].set_title('Node Damage Model Errors')
# axs[0,1].set_xlabel('Case Number')
# axs[0,1].set_ylabel('Abs Difference Between Measured and Model')

# axs[0,2].barh(node_xvals, node_yvals)
# axs[0,2].set_title('Node Damage Model Feature Importance')
# axs[0,2].set_xlabel('Importance')

# axs[1,0].scatter(np.linspace(1,len(EDO_test),len(EDO_test)),Eoutputs[0,:])
# axs[1,0].scatter(np.linspace(1,len(EDO_test),len(EDO_test)),Eoutputs[1,:])
# axs[1,0].set_title('Edge Damage Probability Predictions')
# axs[1,0].set_xlabel('Case Number')
# axs[1,0].set_ylabel('Damage Probability')
# axs[1,0].text(0.5 * (left + right), top, 'Accuracy: '+str(edge_validation_score),
#         horizontalalignment='center',
#         verticalalignment='top',
#         transform=axs[1,0].transAxes)
# axs[1,0].legend('Actual','Model')

# axs[1,1].scatter(np.linspace(1,len(EDO_test),len(EDO_test)),abs(Eoutputs[0,:]-Eoutputs[1,:]))
# axs[1,1].set_title('Edge Damage Model Errors')
# axs[1,1].set_xlabel('Case Number')
# axs[1,1].set_ylabel('Abs Difference Between Measured and Model')

# axs[1,2].barh(edge_xvals, edge_yvals)
# axs[1,2].set_title('Edge Damage Model Feature Importance')
# axs[1,2].set_xlabel('Importance')
# plt.show()