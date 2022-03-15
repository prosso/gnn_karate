import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



"""
34 students divided in two factions (0 and 1). The links represent 78 different interactions between pairs of members outside the club
"""


# load graph from networkx library
G = nx.karate_club_graph()

# retrieve the labels for each node
labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64) # [0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1] with len 34

# create edge index from
adj = nx.to_scipy_sparse_matrix(G).tocoo() # sparse matrix shape (34, 34)
"""
(0, 1)	1
(0, 2)	1
(0, 3)	1
(0, 4)	1
(0, 5)	1
...

For example, (0, 2)	1 -> interaction between member 0 and 2
"""
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long) # tensor([ 0,  0,  0,  0,  0,  0, ...])
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long) # tensor([ 1,  2,  3,  4,  5, ...])
edge_index = torch.stack([row, col], dim=0) # stacking row and col. torch.Size([2, 156])
"""
tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,
          1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,
          3,  3,  3,  3,  3,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,
          7,  7,  8,  8,  8,  8,  8,  9,  9, 10, 10, 10, 11, 12, 12, 13, 13, 13,
         13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20, 20, 21,
         21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 27, 27,
         27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 31,
         31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33,
         33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33],
        [ 1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 17, 19, 21, 31,  0,  2,
          3,  7, 13, 17, 19, 21, 30,  0,  1,  3,  7,  8,  9, 13, 27, 28, 32,  0,
          1,  2,  7, 12, 13,  0,  6, 10,  0,  6, 10, 16,  0,  4,  5, 16,  0,  1,
          2,  3,  0,  2, 30, 32, 33,  2, 33,  0,  4,  5,  0,  0,  3,  0,  1,  2,
          3, 33, 32, 33, 32, 33,  5,  6,  0,  1, 32, 33,  0,  1, 33, 32, 33,  0,
          1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33,  2, 23,
         24, 33,  2, 31, 33, 23, 26, 32, 33,  1,  8, 32, 33,  0, 24, 25, 28, 32,
         33,  2,  8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33,  8,  9, 13, 14, 15,
         18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32]])

"""
# using degree as embedding
embeddings = np.array(list(dict(G.degree()).values()))
"""
G.degree() -> [(0, 16), (1, 9), (2, 10), ...] means that member 0 interacts with 16 other members, member 1 interacts with 9 other members, etc.

embeddings only takes the values, so it will be a list like: [16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3, 2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4, 6, 12, 17]
"""



# normalizing degree values
# Many machine learning algorithms perform better when numerical input variables are scaled to a standard range, and Standardization is popular techniques for scaling numerical data prior to modeling. Specifically, standardization scales each input variable separately by subtracting the mean (called centering) and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one.
scale = StandardScaler()
embeddings = scale.fit_transform(embeddings.reshape(-1,1)) # (34, 1)
"""
[[ 2.98709092]
 [ 1.15480319]
 [ 1.41655858]
 [ 0.36953702]
 [-0.41572915]
 [-0.15397376]
 [-0.15397376]
 [-0.15397376]
 [ 0.10778163]
 [-0.67748454]
 [-0.41572915]
 [-0.93923993]
 [-0.67748454]
 [ 0.10778163]
 [-0.67748454]
 [-0.67748454]
 [-0.67748454]
 [-0.67748454]
 [-0.67748454]
 [-0.41572915]
 [-0.67748454]
 [-0.67748454]
 [-0.67748454]
 [ 0.10778163]
 [-0.41572915]
 [-0.41572915]
 [-0.67748454]
 [-0.15397376]
 [-0.41572915]
 [-0.15397376]
 [-0.15397376]
 [ 0.36953702]
 [ 1.94006936]
 [ 3.24884631]]
"""




















# custom dataset
"""
The KarateDataset class inherits from the InMemoryDataset class and use a Data object to collate all information relating to the karate club dataset. PyG provides an InMemoryDataset class which can be used to create the custom dataset (Note: InMemoryDataset should be used for datasets small enough to load in the memory).
"""
class KarateDataset(InMemoryDataset):
    def __init__(self):
        super(KarateDataset, self).__init__()

        data = Data(edge_index=edge_index) # A data object describing a homogeneous graph. The object includes several attributes to build graphs like node funture, graph connectivity in COO, etc.

        data.num_nodes = G.number_of_nodes() #34

        # embedding
        data.x = torch.from_numpy(embeddings).type(torch.float32)

        # labels
        y = torch.from_numpy(labels).type(torch.long) # labels: [0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1] with len 34
        data.y = y.clone().detach()

        data.num_classes = 2

        # splitting the data into train, validation and test
        # G.nodes(): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33] (len 34)
        # labels: [0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1] (len 34)
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())),
                                                            pd.Series(labels),
                                                            test_size=0.30,
                                                            random_state=42)

        n_nodes = G.number_of_nodes() # 34

        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask

        self.data, self.slices = self.collate([data])


dataset = KarateDataset()
data = dataset[0] # Data(edge_index=[2, 156], num_nodes=34, x=[34, 1], y=[34], num_classes=2, train_mask=[34], test_mask=[34])


















# GCN model with 2 layers
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # GCNConv defines the aggregate and transform messages functions

        # data.num_features: 1
        self.conv1 = GCNConv(data.num_features, 16) # The graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper

        # data.num_classes: 2
        self.conv2 = GCNConv(16, int(data.num_classes)) # The graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper

    def forward(self):
        x, edge_index = data.x, data.edge_index
        # x.shape: torch.Size([34, 1]) --> degree of each node standardized using Standardization (embeddings). The higher is the value, the higher is the number of interactions of the node. For example, the node 33 has the highest value because it has 17 interactions.
        # edge_index.shape: torch.Size([2, 156]) --> actual interactions between members

        x = F.relu(self.conv1(x, edge_index)) # torch.Size([34, 16])
        x = F.dropout(x, training=self.training) # self.training==True when training, self.training==False when evaluating
        x = self.conv2(x, edge_index) # torch.Size([34, 2])
        return F.log_softmax(x, dim=1) # torch.Size([34, 2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data =  data.to(device)

model = Net().to(device)





















torch.manual_seed(42)

optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 200

def train():
  model.train()
  optimizer.zero_grad()
  out = model()
  F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward() # NLLLoss is a loss function commonly used in multi-classes classification tasks. Its meaning is to take log the probability value after softmax and add the probability value of the correct answer to the average
  # how NLLLoss works (from my understanding it is used with batch): https://clay-atlas.com/us/blog/2021/05/25/nllloss-en-pytorch-loss-function/
  optimizer.step()

@torch.no_grad()
def test():
  model.eval()
  logits = model()
  mask1 = data['train_mask']
  pred1 = logits[mask1].max(1)[1]
  acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
  mask = data['test_mask']
  pred = logits[mask].max(1)[1]
  acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
  return acc1,acc

for epoch in range(1, epochs):
  train()

train_acc,test_acc = test()

print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)
