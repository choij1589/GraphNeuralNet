#!/usr/bin/env python
import os
import torch
import torch_geometric
os.system("nvidia-smi")

print(torch.__version__)
cuda = torch.device('cuda')

a = torch.tensor([1., 2.], device=cuda)
b = torch.tensor([1., 2.]).cuda()

print(f"cuda avail: {torch.cuda.is_available()}")
print(f"current_device: {torch.cuda.current_device()}")
print(f"device count: {torch.cuda.device_count()}")

for idx in range(torch.cuda.device_count()):
		print(f"device: {torch.cuda.device(idx)}")
		print(f"device name: {torch.cuda.get_device_name(idx)}")
