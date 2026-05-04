import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[2000, 500, 250, 50], output_size=1, drop_out=0.0):
        super(MLP, self).__init__()
        
        # Feature extractor
        feature_layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            feature_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            ])
            in_features = hidden_size
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Classifier
        self.classifier = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class MultiTaskMLP(nn.Module):
    def __init__(self,input_size, config, hidden_sizes=[2000, 500, 250, 50], drop_out=0.0, ):
        super().__init__()
        self.config =config
        self.models = nn.ModuleDict()
        
        
        # Couches partagées (backbone)
        feature_layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            feature_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            ])
            in_features = hidden_size
        
        self.feature_extractor= nn.Sequential(*feature_layers)
        
        # Initialisation des têtes
        for model_name, params in config.items():
            self.models[model_name] = nn.Linear(hidden_sizes[-1], params["params_model"]["output_size"])

    def forward(self, x):
        features = self.feature_extractor(x)
        return {model_name: head(features) for model_name, head in self.models.items()}



import torch.nn.functional as F
from torch_geometric.nn import SGConv, global_mean_pool, GCNConv

class SGConvNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_sizes=[1064, 512, 256, 64]):
        self.num_features = num_features
        print(self.num_features)
        super(SGConvNet, self).__init__()
        self.conv0 = GCNConv(num_features, 1064 )
        self.conv1 = GCNConv(1064, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv0(x, edge_index))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        #breakpoint()
        x = torch.cat([x], dim=0)
        x = x.flattent()
        x = x.view( -1, 19887)
        #x = global_mean_pool(x, batch)
        print(x.size(),x[0].size())
        x = self.fc(x)
        return x

from torch_geometric.nn import GCNConv, BatchNorm
class GNN(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[128, 128, 64], output_size=1, drop_out=0.0, num_nodes=19887):
        super(GNN, self).__init__()
        feature_extractor = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            feature_extractor.extend([
                GCNConv(in_features, hidden_size),
                BatchNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            ])
            in_features = hidden_size
        self.feature_extractor = nn.ModuleList(feature_extractor)
        self.num_nodes = num_nodes
        self.hidden_size = hidden_sizes[-1]
        self.classifier = nn.Linear(num_nodes * self.hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(0, len(self.feature_extractor), 4):
            conv = self.feature_extractor[i]
            bn = self.feature_extractor[i+1]
            relu = self.feature_extractor[i+2]
            dropout = self.feature_extractor[i+3]
            x = conv(x, edge_index)
            x = bn(x)
            x = relu(x)
            x = dropout(x)
        batch_size = data.num_graphs
        assert x.size(0) == batch_size * self.num_nodes, f"Nombre de noeuds inattendu : {x.size(0)}"
        x = x.view(batch_size, self.num_nodes, self.hidden_size)
        x = x.reshape(batch_size, -1)
        output = self.classifier(x)
        return output

    def get_features(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(0, len(self.feature_extractor), 4):
            conv = self.feature_extractor[i]
            bn = self.feature_extractor[i+1]
            relu = self.feature_extractor[i+2]
            dropout = self.feature_extractor[i+3]
            x = conv(x, edge_index)
            x = bn(x)
            x = relu(x)
            x = dropout(x)
        return x
    
class MultiTaskGNN(nn.Module):
    def __init__(self,input_size, config, hidden_sizes=[128, 128, 64], drop_out=0.0, num_nodes=19887):
        super().__init__()
        self.config =config
        self.models = nn.ModuleDict()
        
        
        # Couches partagées (backbone)
        feature_extractor = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            feature_extractor.extend([
                GCNConv(in_features, hidden_size),
                BatchNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            ])
            in_features = hidden_size
        self.feature_extractor = nn.ModuleList(feature_extractor)
        self.num_nodes = num_nodes
        self.hidden_size = hidden_sizes[-1]
        
        # Initialisation des têtes
        for model_name, params in config.items():
            self.models[model_name] = nn.Linear(hidden_sizes[-1] * self.num_nodes, params["params_model"]["output_size"])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(data)
        for i in range(0, len(self.feature_extractor), 4):
            conv = self.feature_extractor[i]
            bn = self.feature_extractor[i+1]
            relu = self.feature_extractor[i+2]
            dropout = self.feature_extractor[i+3]
            x = conv(x, edge_index)
            x = bn(x)
            x = relu(x)
            x = dropout(x)
        batch_size = data.num_graphs
        assert x.size(0) == batch_size * self.num_nodes, f"Nombre de noeuds inattendu : {x.size(0)}"
        x = x.view(batch_size, self.num_nodes, self.hidden_size)
        x = x.reshape(batch_size, -1)
        return {model_name: head(x) for model_name, head in self.models.items()}