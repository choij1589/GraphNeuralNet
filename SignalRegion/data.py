from torch_geometric.datasets import TUDataset

dataset = TUDataset(root="./ENZYMES", name="ENZYMES")
print(type(dataset))
