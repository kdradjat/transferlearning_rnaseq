import pdb
import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv, global_mean_pool, GCNConv

# Définition du modèle SGConvNet
class SGConvNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        print(self.num_features)
        super(SGConvNet, self).__init__()
        self.conv0 = GCNConv(num_features, 1064 )
        self.conv1 = GCNConv(1064, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, 64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv0(x, edge_index))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        #breakpoint()
        #x = torch.cat([x], dim=0)
        # x = x.flattent()
       # x = x.view( -1, 19887)
        x = global_mean_pool(x, batch)
        #print(x.size(),x[0].size())
        
        x = self.fc(x)
        return x

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, drop_out=0.5, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(250, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(50, output_size)
        )
    def forward(self, data):
        return self.layers(data)
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #breakpoint()
        output = model(data)
        #breakpoint()
        loss = criterion(output, data.y.view(-1,1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data.to(device)
            outputs = model(data)
            predicted = (outputs > 0).float()
            total += data.y.size(0)
            correct += (predicted == data.y.view(-1,1)).sum().item()
            accuracy = correct / total
    return correct / len(loader.dataset)
