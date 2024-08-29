import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GATRNN(nn.module):
    def __init__(self, num_static_node_features, num_static_edge_features, num_weather_features, hidden_size):
        super().__init__()
        self.gat_conv = GATv2Conv(num_static_node_features, hidden_size, edge_dim=num_static_edge_features)
        self.lstm = nn.LSTM(hidden_size + num_weather_features, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, node_static_features, edge_static_features, node_dynamic_features, edge_index):
        # Implement Later
        return
