import os, sys; sys.path.append("/data6/Users/choij/GraphNeuralNet")
import argparse
from time import time
from itertools import product

import torch
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle
from sklearn import metrics
from MLTools import rtfile_to_datalist, MyDataset
from MLTools import GCN, GNN, ParticleNet
from MLTools import EarlyStopping, History
from ROOT import TFile

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", default=None, required=True, type=str, help="model type")
parser.add_argument("--optim", "-o", default=None, required=True, type=str, help="optimizer type")
args = parser.parse_args()

# get root files
f_sig = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTToHcToWA_AToMuMu_MHc130_MA90.root")
f_bkg = TFile.Open("/data6/Users/choij/GraphNeuralNet/SelectorOutput/2017/Skim1E2Mu__/Selector_TTLL_powheg.root")

print("Loading datasets...")
# convert root file to graphs
sig_datalist = rtfile_to_datalist(f_sig, is_signal=True)
bkg_datalist = rtfile_to_datalist(f_bkg, is_signal=False)
f_sig.Close()
f_bkg.Close()
datalist = shuffle(sig_datalist+bkg_datalist, random_state=953)
train_dataset = MyDataset(datalist[:10000])
val_dataset = MyDataset(datalist[10000:11000])
test_dataset = MyDataset(datalist[11000:])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Constants
num_features = train_dataset[0].num_node_features
num_classes = train_dataset.num_classes
hidden_channels = 256
learning_rate = 0.2

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
		correct = 0
		for data in loader:
				out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.batch.to(DEVICE))
				pred = out.argmax(dim=1)
				loss += float(criterion(out, data.y.to(DEVICE)).sum())
				correct += int((pred == data.y.to(DEVICE)).sum())
		loss /= len(loader.dataset)
		correct /= len(loader.dataset)
		return (loss, correct)

def predict(model, node_features, edge_index, prob=False):
		model.eval()
		model.to('cpu')
		predictions = list()
		with torch.no_grad():
				prediction = model(node_features, edge_index)
		
		return np.array(predictions)

def visualize_training_steps(history, path):
		if not os.path.exists(os.path.dirname(path)):
				os.makedirs(os.path.dirname(path))

		epochs = np.arange(1, len(history.train_loss())+1)
		plt.figure(figsize=(24, 8))
		plt.subplot(1, 2, 1)
		plt.plot(epochs, history.train_loss(), label="Train Loss")
		plt.plot(epochs, history.val_loss(), label="Validation Loss")
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.legend(loc="best")
		plt.grid(True)

		plt.subplot(1, 2, 2)
		plt.plot(epochs, history.train_acc(), label="Train Accuracy")
		plt.plot(epochs, history.val_acc(), label="Validation Accuracy")
		plt.xlabel("epoch")
		plt.ylabel("accuracy")
		plt.legend(loc="best")
		plt.grid(True)

		plt.savefig(path)


def optimize(model, optimizer, title):
		criterion = torch.nn.CrossEntropyLoss()
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
		early_stopping = EarlyStopping(patience=7, path=f"/data6/Users/choij/GraphNeuralNet/.models/{title}/checkpoint.pt")
		history = History(name=title)

		f = open(f"/data6/Users/choij/GraphNeuralNet/results/training_stage_{title}.txt", "w")
		
		f.write(f"Start training {title}...")

		for epoch in range(200):
				train(model, criterion, optimizer, scheduler)
				train_loss, train_acc = test(model, criterion, train_loader)
				val_loss, val_acc = test(model, criterion, val_loader)
				f.write(f"[EPOCH {epoch}]\tTrain Acc: {train_acc:.4f}\tTrain Loss: {train_loss:.4f}")
				f.write(f"[EPOCH {epoch}]\tValid Acc: {val_acc:.4f}\tValid Loss: {val_loss:.4f}\n")
				
				history.update(train_loss, train_acc, val_loss, val_acc)
				early_stopping.update(val_loss, model)
				if early_stopping.early_stop:
						f.write(f"Early stopping in epoch {epoch}")
						break

		visualize_training_steps(history, path=f"/data6/Users/choij/GraphNeuralNet/results/training_stage_{title}.png")
		f.close()

def main():
		### Initialize model and optimizer
    if args.model == "GCN":
        model = GCN(num_features, num_classes, hidden_channels).to(DEVICE)
    elif args.model == "GNN":
        model = GNN(num_features, num_classes, hidden_channels).to(DEVICE)
    elif args.model == "ParticleNet":
        model = ParticleNet(num_features, num_classes, hidden_channels).to(DEVICE)
    else:
        print(f"Wrong model name {args.model}!")
        raise NameError

    if args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(model_parameters(), lr=learning_rate)
    else:
        print(f"Wrong optimizer name {args.optim}!")
        raise NameError

    optimize(model, optimizer, title=f"{args.model}-{args.optim}")

    print(f"{model_name}-{optimizer_name} done!")

if __name__ == "__main__":
		main()
