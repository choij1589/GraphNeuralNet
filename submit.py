#!/usr/bin/env python
import subprocess
from itertools import product

models = ["GCN", "GNN", "ParticleNet"]
optims = ["AdamW", "Adam", "Adadelta"]

def run_optimize(model, optim):
		proc = subprocess.Popen(f"python /data6/Users/choij/GraphNeuralNet/optimize.py --model {model} --optim {optim} --hidden_channels 256 --initial_lr 0.2".split(),
														stdin=subprocess.PIPE,
														stdout=subprocess.PIPE)
		return proc


procs = []
for model, optim in product(models, optims):
		procs.append(run_optimize(model, optim))

for proc in procs:
		proc.communicate()
		assert proc.returncode == 0

