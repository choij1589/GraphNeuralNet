#!/usr/bin/env python
import os
import subprocess
import shlex
from time import sleep

models = ["GCN", "GNN", "ParticleNet"]
optims = ["RMSprop", "Adam", "Adadelta"]

print(os.getcwd())
for model in models:
		print(f"Optimizing {model}...")
		procs = []
		for optim in optims:
				proc = subprocess.Popen(
								f"python /data6/Users/choij/GraphNeuralNet/optimize.py --model {model} --optim {optim}".split(),
								stdout=subprocess.PIPE,
								stderr=subprocess.STDOUT,
								preexec_fn=os.setpgrp
								)
				procs.append(proc)

		while True:
				done_flag = True
				for proc in procs:
						if not proc.poll():
								done_flag = False

				if not done_flag:
						print(f"Running processes {[proc.pid for proc in procs]}")
						sleep(300)
				else:
						print(f"processes done for optimizing {model}")
						break

