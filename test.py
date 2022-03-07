import warnings; warnings.filterwarnings(action='ignore')

from ROOT import TFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.utils import shuffle

import torch
from torch_geometric.loader import DataLoader
from MLTools import rtfile_to_datalist, MyDataset
from MLTools import GCN, GNN, ParticleNet

### get the dataset and trained models
print("@@@@@ Loading datasets...")
f_sig = TFile.Open("SelectorOutput/2017/Skim1E2Mu__/Selector_TTToHcToWA_AToMuMu_MHc130_MA90.root")
f_bkg = TFile.Open("SelectorOutput/2017/Skim1E2Mu__/Selector_TTLL_powheg.root")

# convert root file to particle cloud
sig_datalist = rtfile_to_datalist(f_sig, is_signal=True)
bkg_datalist = rtfile_to_datalist(f_bkg, is_signal=False)
f_sig.Close()
f_bkg.Close()
datalist = shuffle(sig_datalist+bkg_datalist, random_state=953)

train_dataset = MyDataset(datalist[:9000])
val_dataset = MyDataset(datalist[9000:11000])
test_dataset = MyDataset(datalist[11000:])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

num_features = train_dataset[0].num_node_features
num_classes = train_dataset.num_classes
hidden_channels = 256

###
print("@@@@@ Loading models...")
gcn = GCN(num_features, num_classes, hidden_channels)
gcn.load_state_dict(torch.load(".models/GCN-Adadelta/checkpoint.pt"))
gnn = GNN(num_features, num_classes, hidden_channels)
gnn.load_state_dict(torch.load(".models/GNN-Adadelta/checkpoint.pt"))
pnet = ParticleNet(num_features, num_classes, hidden_channels)
pnet.load_state_dict(torch.load(".models/ParticleNet-Adadelta/checkpoint.pt"))

def predict(model, loader):
    model.eval()
    predictions = []
    answers = []
    with torch.no_grad():
        for data in loader:
            pred = model(data.x, data.edge_index, data.batch)
            for p in pred:
                predictions.append(p[1].numpy())
            for a in data.y:
                answers.append(a.numpy())

    return np.array(predictions), np.array(answers)

pred_gcn, answers = predict(gcn, test_loader)
pred_gnn, _ = predict(gnn, test_loader)
pred_pnet, _ = predict(pnet, test_loader)

fpr_gcn, tpr_gcn, _ = metrics.roc_curve(answers, pred_gcn, pos_label=1)
auc_gcn = metrics.auc(fpr_gcn, tpr_gcn)
fpr_gnn, tpr_gnn, _ = metrics.roc_curve(answers, pred_gnn, pos_label=1)
auc_gnn = metrics.auc(fpr_gnn, tpr_gnn)
fpr_pnet, tpr_pnet, _ = metrics.roc_curve(answers, pred_pnet, pos_label=1)
auc_pnet = metrics.auc(fpr_pnet, tpr_pnet)

plt.figure(figsize=(12, 12))
plt.title("test ROC curve")
plt.plot(tpr_gcn, 1-fpr_gcn, 'r--', label=f"GCN, auc={auc_gcn:.3}")
plt.plot(tpr_gnn, 1-fpr_gnn, 'b--', label=f"GNN, auc={auc_gnn:.3}")
plt.plot(tpr_pnet, 1-fpr_pnet, 'g--', label=f"ParticleNet, auc={auc_pnet:.3}")
plt.legend(loc='best')
plt.xlabel('sig. eff.')
plt.ylabel('bkg. rej.')
plt.savefig("results/ROC.png")
