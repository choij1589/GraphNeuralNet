import os, sys
sys.path.append("/data6/Users/choij/GraphNeuralNet")
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from ROOT import TFile
from torch_geometric.loader import DataLoader

from MLTools import rtfile_to_datalist
from MLTools import MyDataset
from MLTools import GCN, GNN, ParticleNet
from MLTools import EarlyStopping
from MLTools import History, visualize_training_steps

##### get arguments #####
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", required=True, type=str, help="model type")
parser.add_argument("--optim", "-o", required=True, type=str, help="optimizer type")
parser.add_argument("--hidden_channels", "-c", required=True, type=int, help="no. of hidden channels for each model")
parser.add_argument("--initial_lr", "-l", required=True, type=float, help="initial learning rate")
parser.add_argument("--pilot", "-p", action="store_true", help="pilot run")
args = parser.parse_args()

model_name = args.model
optim_name = args.optim
hidden_channels = args.hidden_channels
initial_lr = args.initial_lr
is_pilot = args.pilot

##### GPU settings #####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

##### Dataset #####
print("@@@@@ Loading datasets...")
f_sig = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTToHcToWA_AToMuMu_MHc130_MA90.root")
f_bkg = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTLL_powheg.root")

# convert root file to particle cloud
sig_datalist = rtfile_to_datalist(f_sig, is_signal=True)
bkg_datalist = rtfile_to_datalist(f_bkg, is_signal=False)
f_sig.Close()
f_bkg.Close()
datalist = shuffle(sig_datalist+bkg_datalist, random_state=953)

if is_pilot:
    train_dataset = MyDataset(datalist[:100])
    val_dataset = MyDataset(datalist[100:200])
    test_dataset = MyDataset(datalist[200:300])
else:
    train_dataset = MyDataset(datalist[:9000])
    val_dataset = MyDataset(datalist[9000:11000])
    test_dataset = MyDataset(datalist[11000:])

num_features = train_dataset[0].num_node_features
num_classes = train_dataset.num_classes

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

##### helper functions #####
def train(model, criterion, optimizer, scheduler):
    model.train()
    
    for data in train_loader:
        out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.batch.to(DEVICE))
        loss = criterion(out, data.y.to(DEVICE))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    
def test(model, criterion, loader):
    model.eval()
    
    loss = 0.
    correct = 0. 
    for data in loader:
        out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.batch.to(DEVICE))
        pred = out.argmax(dim=1)
        loss += float(criterion(out, data.y.to(DEVICE)).sum())
        correct += int((pred == data.y.to(DEVICE)).sum())
    loss /= len(loader.dataset)
    correct /= len(loader.dataset)
    return (loss, correct)

def optimize(model_name, optim_name):
    ##### Initialize model and optimizer #####
    if model_name == "GCN":
        model = GCN(num_features, num_classes, hidden_channels).to(DEVICE)
    elif model_name == "GNN":
        model = GNN(num_features, num_classes, hidden_channels).to(DEVICE)
    elif model_name == "ParticleNet":
        model = ParticleNet(num_features, num_classes, hidden_channels).to(DEVICE)
    else:
        print(f"Wrong model name {model_name}!")
        raise NameError

    if optim_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=initial_lr)
    elif optim_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif optim_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif optim_name == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=initial_lr)
    else:
        print(f"Wrong optimizer name {optim_name}!")
        raise NameError

    title = f"{model_name}-{optim_name}"
    ##### Optimize step #####
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    early_stopping = EarlyStopping(patience=11, path=f"/data6/Users/choij/GraphNeuralNet/.models/{title}/checkpoint.pt")
    history = History(name=title)
    
    f = open(f"/data6/Users/choij/GraphNeuralNet/results/training_stage_{title}.txt", "w")
    f.write(f"@@@@@ Start training {title}...\n")
    
    if is_pilot: final_epoch = 10
    else: final_epoch = 200
    
    for epoch in range(final_epoch):
        train(model, criterion, optimizer, scheduler)
        train_loss, train_acc = test(model, criterion, train_loader)
        val_loss, val_acc = test(model, criterion, val_loader)
        f.write(f"[EPOCH {epoch}]\tTrain Acc: {train_acc:.4f}\tTrain Loss: {train_loss:.4f}\t")
        f.write(f"Valid Acc: {val_acc:.4f}\tValid Loss: {val_loss:.4f}\n")
        
        history.update(train_loss, train_acc, val_loss, val_acc)
        early_stopping.update(val_loss, model)
        if early_stopping.early_stop:
            f.write(f"Early stopping in epoch {epoch}\n")
            break

        if epoch == final_epoch-1:
            torch.save(model.state_dict(), 
										  f"/data6/Users/choij/GraphNeuralNet/.models/{title}/checkpoint.pt")
    f.close()
    visualize_training_steps(history, path=f"/data6/Users/choij/GraphNeuralNet/results/training_stage_{title}.png")
    
if __name__ == "__main__":
		optimize(model_name, optim_name)
    
