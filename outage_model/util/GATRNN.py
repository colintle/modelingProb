import torch.nn as nn
import torch
from torch_geometric.nn import GATConv

class GATRNN(nn.Module):
    def __init__(self, num_static_node_features, num_static_edge_features, hidden_size):
        super().__init__()
        self.gat_conv = GATConv(num_static_node_features, hidden_size, edge_dim=num_static_edge_features)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, node_static_features, edge_static_features, edge_index):
        node_static_features = node_static_features.float()
        edge_static_features = edge_static_features.float()

        gat_output = self.gat_conv(node_static_features, edge_index.t(), edge_static_features)
        gat_output = torch.relu(gat_output)

        mean_pooled_output = self.linear(gat_output)

        return mean_pooled_output
