import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

def normalize(val, mini, range):
    """Normalizes input feature data using the following transformation
    u_normalized = (u_original - u_mean) / u_range

    NORMALIZATION MUST BE DONE BY FEATURE 
    I.E. if original input u is (2x100): 
        denormalize u(0,:) and u(1,:) seperately 

    Args:
        val (float): Original feature data
        mean (float): Predetermined mean of feature data
        range (float): Predetermined range of feature data

    Returns:
        float: Normalized feature data
    """
    y = (val - mini) / range 
    return y

# Function to find the max, mean, and minimum value of feature data passed
def findStats(feature):

    meanFeat = np.mean(feature)
    maxFeat = np.max(feature)
    minFeat = np.min(feature)

    return meanFeat,maxFeat,minFeat

def normalizeDataset(datasets, sF, featureNames):
    transformed = []

    for dataset in datasets:
        processed = {}

        for key in ['scenario', 'edge_index']:
            processed[key] = dataset[key]

        for category, features in featureNames.items():
            shape = dataset[category].size()
            normalized_features = np.zeros(shape)

            for index, feature in enumerate(features):
                if category == 'node_static_features' or category == 'edge_static_features':
                    raw_data = np.copy(dataset[category][:, index].detach().numpy())
                else:
                    raw_data = np.copy(dataset[category][:,:,index].detach().numpy())
                
                if category == "node_static_features":
                    min_key = f'minNode{feature}'
                    range_key = f'rangeNode{feature}'
                elif category == "edge_static_features":
                    min_key = f'minEdge{feature}'
                    range_key = f'rangeEdge{feature}'
                else:
                    min_key = f'min{feature}'
                    range_key = f'range{feature}'

                normalize_data = normalize(raw_data, float(sF[min_key]), float(sF[range_key]))

                if category == 'node_static_features' or category == 'edge_static_features':
                    normalized_features[:, index] = normalize_data
                else:
                    normalized_features[:,:, index] = normalize_data
            
            normalized_features = torch.tensor(normalized_features, dtype=torch.float)
            processed[category] = normalized_features

        ##Target
        raw_data = np.copy(dataset["targets"].detach().numpy())
        normalized_data = normalize(raw_data, float(sF["minProb"]), float(sF["rangeProb"]))
        processed["targets"] = torch.tensor(normalized_data, dtype=torch.float)
        processed["coordinates"] = np.copy(dataset["coordinates"])

        transformed.append(processed)
    
    return transformed

# GAT Model Training Function
def trainGAT(model, node_static_features, edge_static_features, node_dynamic_features, edge_index, targets, optimizer, criterion, rangeImpact):
    # Set model into training model
    model.train()
    # Initialize the optimizer's gradient
    optimizer.zero_grad()
    
    # Pass data to model to get estimated output
    outputs = model(node_static_features, edge_static_features, node_dynamic_features, edge_index)
    
    # Calculate the loss of the measured vs estimated output
    loss = criterion(outputs.view(-1, 1), targets.view(-1, 1))
    
    # Compute the gradient of the loss with respect to all the model parameters
    loss.backward()
    
    # Perform a parameter update
    optimizer.step()
    
    # Return the loss
    return loss.item()

# Model Validation Function
def validGAT(model, node_static_features,edge_static_features, node_dynamic_features, edge_index, targets, optimizer, criterion, rangeImpact):
    # Set the model into evaluation mode
    model.eval()

    # Make sure the gradients are not considered  
    with torch.no_grad():

        # Pass the input data to the model to get estimated output
        outputs = model(node_static_features, edge_static_features, node_dynamic_features, edge_index)

        # Calculate the loss of the measured vs estimated output
        loss = criterion(outputs.view(-1, 1), targets.view(-1, 1))
    
    # Return the loss
    return loss.item()

def trainValidate(model, optimizer, criterion, device, nDatasets_t, nDatasets_v, sF, epochs=1500,validation_scale=1.0):
    tLOSS = []
    vLOSS = []

    for epoch in range(epochs):
        training_loss = []
        validation_loss = []

        for dataset in nDatasets_t:
            node_static_feats = dataset['node_static_features'].to(device)
            edge_static_feats = dataset['edge_static_features'].to(device)
            node_dynamic_feats = dataset['node_dynamic_features'].to(device)
            edge_index = dataset['edge_index'].to(device)
            targets = dataset['targets'].to(device)

            tloss = trainGAT(model, node_static_feats, edge_static_feats, node_dynamic_feats, edge_index, targets, optimizer, criterion, float(sF['rangeProb']))
            training_loss.append(tloss)

        epochTloss = sum(training_loss) / len(training_loss)
        tLOSS.append(epochTloss)

        for dataset in nDatasets_v:
            node_static_feats = dataset['node_static_features'].to(device)
            edge_static_feats = dataset['edge_static_features'].to(device)
            node_dynamic_feats = dataset['node_dynamic_features'].to(device)
            edge_index = dataset['edge_index'].to(device)
            targets = dataset['targets'].to(device)

            vloss = validGAT(model, node_static_feats, edge_static_feats, node_dynamic_feats, edge_index, targets, optimizer, criterion, float(sF['rangeProb']))
            validation_loss.append(vloss)

        epochVloss = (sum(validation_loss) / len(validation_loss)) * validation_scale
        vLOSS.append(epochVloss)

        print('.', end ="", flush=True)
        if (epoch % 100 == 0) or (epoch == epochs - 1):
            print(f'\nEpoch {epoch} | Training Loss: {epochTloss:.5f} | Validation Loss: {epochVloss:.5f} | Train RMSE: {np.sqrt(epochTloss):.5f} | Valid RMSE: {np.sqrt(epochVloss):.5f}')

    return tLOSS, vLOSS

def validateGAT(model, node_static_features,edge_static_features, node_dynamic_features, edge_index, targets, optimizer, criterion, rangeImpact):
    # Set the model into evaluation mode
    model.eval()
 
    # Make sure the gradients are not considered  
    with torch.no_grad():
 
        # Pass the input data to the model to get estimated output
        outputs = model(node_static_features, edge_static_features, node_dynamic_features, edge_index)
 
        # Calculate the loss of the measured vs estimated output
        #loss = criterion(outputs.view(-1, 1)/rangeImpact, targets.view(-1, 1)/rangeImpact)
        loss = criterion(outputs.view(-1, 1), targets.view(-1, 1))
    # Return the loss
    return outputs, loss.item()

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
    plt.show()


