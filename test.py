#!/usr/bin/env python
import os, sys
sys.path.append("/data6/Users/choij/GraphNeuralNet")

import torch
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle
from MLTools import rtfile_to_datalist, MyDataset
from MLTools import ParticleNet
from ROOT import TFile

# get root files
f_sig = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTToHcToWA_AToMuMu_MHc130_MA90.root")
f_bkg = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTLL_powheg.root")

print("Loading datasets...")
# convert root file to graphs
sig_datalist = rtfile_to_datalist(f_sig, is_signal=True)
bkg_datalist = rtfile_to_datalist(f_bkg, is_signal=False)
f_sig.Close()
f_bkg.Close()
datalist = shuffle(sig_datalist+bkg_datalist)
train_dataset = MyDataset(datalist[:11000])
test_dataset = MyDataset(datalist[11000:])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {DEVICE}")
num_features = train_dataset[0].num_node_features
num_classes = train_dataset.num_classes
hidden_channels = 256
learning_rate = 0.2

# Define the model, optimizer, etc...
model = ParticleNet(num_features, num_classes, hidden_channels).to(DEVICE)
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
criterion = torch.nn.CrossEntropyLoss()

def train():
		model.train()

		for data in train_loader:
				out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.batch.to(DEVICE))
				loss = criterion(out, data.y.to(DEVICE))
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
		scheduler.step()

def test(loader):
		model.eval()

		loss = 0.
		correct = 0
		for data in loader:
				out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.batch.to(DEVICE))
				pred = out.argmax(dim=1)
				loss += float(criterion(out, data.y.to(DEVICE)).sum())
				correct += int((pred == data.y.to(DEVICE)).sum())
		loss /= len(loader.dataset)
		correct /= len(loader.dataset)
		return (loss, correct)

print("Start training...")
for epoch in range(120):
		train()
		train_loss, train_acc = test(train_loader)
		test_loss, test_acc = test(test_loader)
		print(f"[EPOCH {epoch}]\tTrain Acc: {train_acc:.4f}\tTrain Loss: {train_loss:.4f}")
		print(f"[EPOCH {epoch}]\tTest Acc: {test_acc:.4f}\tTest Loss: {test_loss:.4f}\n")
