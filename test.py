import warnings; warnings.filterwarnings(action='ignore')

from ROOT import TFile
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.utils import shuffle

import torch
from MLTools import GCN, GNN, ParticleNet

### get the dataset and trained models
print("@@@@@ Loading datasets...")
ig = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTToHcToWA_AToMuMu_MHc130_MA90.root")
f_bkg = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTLL_powheg.root")

# convert root file to particle cloud
sig_datalist = rtfile_to_datalist(f_sig, is_signal=True)
bkg_datalist = rtfile_to_datalist(f_bkg, is_signal=False)
f_sig.Close()
f_bkg.Close()
datalist = shuffle(sig_datalist+bkg_datalist, random_state=953)

train_dataset = MyDataset(datalist[:9000])
val_dataset = MyDataset(datalist[9000:11000])
test_datatset = MyDataset(datalist[11000:])

num_features = train_dataset[0].num_node_features
num_classes = train_dataset.num_classes
hidden_channels = 128

###
gcn = GCN(num_features, num_classes, hidden_channels)
gcn.load_state_dict(
				torch.load("/data6/Users/choij/GraphNeuralNet/.models/GCN-Adadelta/checkpoint.pt"))
gnn = GNN(num_features, num_classes, hidden_channels)
gnn.load_state_dict(
				torch.load("/data6/Users/choij/GraphNeuralNet/.models/GNN-Adadelta/checkpoint.pt"))
pnet = ParticleNet(num_features, num_classes, hidden_channels)
pnet.load_state_dict(
				torch.load("/data6/Users/choij/GraphNeuralNet/.models/ParticleNet-Adadelta/checkpoint.pt"))


