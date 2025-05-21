import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv


class GNNActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_shape):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output_dim = int(action_shape.shape[0])
        self.additional_feature_vector_dim = 1

        self.fc_mu = nn.Linear(hidden_dim + self.additional_feature_vector_dim, self.output_dim)
        self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.max_action = 1.0

    def forward(self, observations, state=None, info={}):
        batch_size = observations["graph_nodes"].shape[0]
        data_list = [Data(x=observations["graph_nodes"][index],
                          edge_index=observations.graph_edge_links[0][:, observations.graph_edges[0]])
                     for index in range(batch_size)]

        batch_data = Batch.from_data_list(data_list)
        additional_features = observations["additional_features"].clone().detach() #torch.tensor(observations["additional_features"])


        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch

        x = F.relu(self.conv1(x, edge_index, edge_weight=None))
        x = F.relu(self.conv2(x, edge_index, edge_weight=None))
        x = global_mean_pool(x, batch=batch)

        x = torch.cat([x, additional_features], dim=1)
        mu = self.fc_mu(x)

        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()

        return (mu, sigma), state


class GNNCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.additional_feature_vector_dim = 1
        self.fc = nn.Linear(hidden_dim + self.additional_feature_vector_dim, 1)

    def forward(self, observations: nx.Graph, state=None, info={}):
        batch_size = observations["graph_nodes"].shape[0]
        data_list = [Data(x=observations["graph_nodes"][index],
                          edge_index=observations.graph_edge_links[0][:, observations.graph_edges[0]])
                     for index in range(batch_size)]
        batch_data = Batch.from_data_list(data_list)
        additional_features = observations["additional_features"].clone().detach()

        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch=batch)
        x = torch.cat([x, additional_features], dim=1)

        x = self.fc(x)

        return x
