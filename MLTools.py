import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, SELU
from torch_geometric.nn import knn_graph, global_mean_pool
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, InMemoryDataset
from Scripts.DataFormat import Particle
from Scripts.DataFormat import get_leptons, get_jets


def select_loosen(channel, evt, muons, electrons, jets):
    # 1E2Mu
    # 1. Should pass triggers and safe cuts
    # 2. 1E2Mu
    # 3. Exist OS muon pair with M(OSSF) > 12 GeV
    # 4. Nj >= 2, Nb >= 1
    if channel == "1E2Mu":
        if not (evt.passDblMuTrigs or evt.passEMuTrigs):
            return False
        pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.)
                        or (muons[0].Pt() > 25. and electrons[0].Pt() > 15.)
                        or (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
        if not pass_safecut:
            return False

        if not len(jets) >= 2:
            return False

        return True

    elif channel == "3Mu":
        raise (NotImplementedError)
    else:
        raise (KeyError)


def get_prompt_leptons(muons, electrons):
    muons_prompt = []
    electrons_prompt = []
    for muon in muons:
        if muon.LepType() < 0: continue
        muons_prompt.append(muon)
    for electron in electrons:
        if electron.LepType() < 0.: continue
        electrons_prompt.append(electron)

    return muons_prompt, electrons_prompt


def get_edge_indices(node_list, k=4):
    edge_index = []
    edge_attribute = []
    for i, node in enumerate(node_list):
        distances = {}  # apply K-NN edge with deltaR
        for j, neighbor in enumerate(node_list):
            if i == j:  # same node
                continue
            deta = node[1] - neighbor[1]
            dphi = np.remainder(node[2] - neighbor[2], np.pi)
            distances[j] = np.sqrt(np.power(deta, 2) + np.power(dphi, 2))
            sorted_dRs = dict(
                sorted(distances.items(), key=lambda item: item[1]))
            for n in list(sorted_dRs.keys())[:k]:
                edge_index.append([i, n])
                edge_attribute.append([distances[n]])

    return torch.tensor(edge_index,
                        dtype=torch.long), torch.tensor(edge_attribute,
                                                        dtype=torch.float)


def rtfile_to_datalist(rtfile, is_signal=False):
    data_list = []
    for evt in rtfile.Events:
        muons, electrons = get_leptons(evt)
        jets, bjets = get_jets(evt)
        METv = Particle(evt.METv_pt, 0., evt.METv_phi, 0.)

        if not select_loosen("1E2Mu", evt, muons, electrons, jets):
            continue

        muons_prompt, electrons_prompt = get_prompt_leptons(muons, electrons)
        is_prompt_evt = (len(muons) == len(muons_prompt)
                         and len(electrons) == len(electrons_prompt))
        if not is_signal and not is_prompt_evt: pass  # fake events
        elif is_signal and is_prompt_evt: pass  # signal events
        else: continue  # to reduce noise

        # Now let's convert the event to a single graph
        node_list = []
        for particle in muons + electrons + jets:
            node_list.append([
                particle.Pt(),
                particle.Eta(),
                particle.Phi(),
                particle.M(),
                particle.Charge(),
                particle.IsMuon(),
                particle.IsElectron(),
                particle.IsJet(),
                particle.BtagScore()
            ])

        x = torch.tensor(node_list, dtype=torch.float)

        # make edges
        # NOTE: It is directed graph!
        # for each node, find 3 nearest particles and connect them
        edge_index, edge_attribute = get_edge_indices(node_list, k=4)
        data = Data(x=x,
                    y=int(is_signal),
                    edge_index=edge_index.t().contiguous(),
                    edge_attribute=edge_attribute)
        data_list.append(data)
        if len(data_list) == 15000:
            break

    return data_list


class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__("./tmp/MyDataset")
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)


# Modules
class GCN(torch.nn.Module):
		def __init__(self, num_features, num_classes, hidden_channels):
				super(GCN, self).__init__()
				self.gn0 = GraphNorm(num_features)
				self.conv1 = GCNConv(num_features, hidden_channels)
				self.gn1 = GraphNorm(hidden_channels)
				self.conv2 = GCNConv(hidden_channels, hidden_channels)
				self.gn2 = GraphNorm(hidden_channels)
				self.conv3 = GCNConv(hidden_channels, hidden_channels)
				self.dense = Linear(hidden_channels, hidden_channels)
				self.output = Linear(hidden_channels, num_classes)

		def forward(self, x, edge_index, batch):
				# Convolution layers
				x = self.gn0(x)
				x = F.relu(self.conv1(x, edge_index))
				x = self.gn1(x)
				x = F.relu(self.conv2(x, edge_index))
				x = self.gn2(x)
				x = F.relu(self.conv3(x, edge_index))

				# readout layers
				x = global_mean_pool(x, batch)

				# dense layers
				x = F.dropout(x, p=0.5, training=self.training)
				x = F.relu(self.dense(x))
				x = self.output(x)

				return F.softmax(x, dim=1)

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GNN, self).__init__()
        self.gn0 = GraphNorm(num_features)
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.gn1 = GraphNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.gn2 = GraphNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.dense = Linear(hidden_channels, hidden_channels)
        self.output = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Convolution layers
        x = self.gn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.gn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.gn2(x)
        x = F.relu(self.conv3(x, edge_index))

        # readout layers
        x = global_mean_pool(x, batch)

        # dense layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense(x))
        x = self.output(x)

        return F.softmax(x, dim=1)


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = Sequential(Linear(2 * in_channels, out_channels), SELU(),
                              Linear(out_channels, out_channels), SELU(),
                              Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class DynamicEdgeConv(EdgeConv):
		def __init__(self, in_channels, out_channels, k=4):
				super().__init__(in_channels, out_channels)
				self.k = k

		def forward(self, x, edge_index=None, batch=None):
				if edge_index is None:
						edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
				return super().forward(x, edge_index)

class ParticleNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dynamic=False):
        super(ParticleNet, self).__init__()
        self.dynamic = dynamic
        self.gn0 = GraphNorm(num_features)
        self.conv1 = DynamicEdgeConv(num_features, hidden_channels)
        self.gn1 = GraphNorm(hidden_channels)
        self.conv2 = DynamicEdgeConv(hidden_channels, hidden_channels)
        self.gn2 = GraphNorm(hidden_channels)
        self.conv3 = DynamicEdgeConv(hidden_channels, hidden_channels)
        self.dense = Linear(hidden_channels, hidden_channels)
        self.output = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Convolution layers
        x = self.gn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.gn1(x)
        x = F.relu(self.conv2(x)) if self.dynamic else F.relu(self.conv2(x, edge_index))
        x = self.gn2(x)
        x = F.relu(self.conv3(x)) if self.dynamic else F.relu(self.conv3(x, edge_index))

        # readout layers
        x = global_mean_pool(x, batch)

        # dense layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense(x))
        x = self.output(x)

        return F.softmax(x, dim=1)

class EarlyStopping():
		def __init__(self, patience=7, delta=0, path="./checkpoint.pt"):
				self.patience = patience
				self.counter = 0
				self.best_score = None
				self.early_stop = False
				self.val_loss_min = np.Inf
				self.delta = delta
				self.path = path
				if not os.path.exists(os.path.dirname(path)): 
						os.makedirs(os.path.dirname(path))

		def update(self, val_loss, model):
				score = -val_loss
				if self.best_score is None:
						self.best_score = score
						self.save_checkpt(val_loss, model)
				elif score < self.best_score + self.delta:
						self.counter += 1
						print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
						if self.counter >= self.patience:
								self.early_stop = True
				else:
						self.best_score = score
						self.save_checkpt(val_loss, model)
						self.counter = 0

		def save_checkpt(self, val_loss, model):
				torch.save(model.state_dict(), self.path)
				self.val_loss_min = val_loss

class History():
		def __init__(self, name):
				self.name = name
				self.train_loss = []
				self.train_acc = []
				self.val_loss = []
				self.val_acc = []

		def update(self, train_loss, train_acc, val_loss, val_acc):
				self.train_loss.append(train_loss)
				self.train_acc.append(train_acc)
				self.val_loss.append(val_loss)
				self.val_acc.append(val_acc)

		def get_train_loss(self):
				return np.array(self.train_loss)

		def get_train_acc(self):
				return np.array(self.train_acc)

		def get_val_loss(self):
				return np.array(self.val_loss)

		def get_val_acc(self):
				return np.array(self.val_acc)


def visualize_training_steps(history, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    epochs = np.arange(1, len(history.get_train_loss())+1)
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get_train_loss(), label="Train loss")
    plt.plot(epochs, history.get_val_loss(), label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get_train_acc(), label="Train Accuracy")
    plt.plot(epochs, history.get_val_acc(), label="Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.savefig(path)
