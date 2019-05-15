import os
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from Dataset import Dataset

# CUDA for pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Parameters
params = {"batch_size": 3,
	"shuffle": True,
	"num_workers": 20
	}
max_epochs = 100

partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
label = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}


training_set = Dataset(partition["train"], labels)
trainning_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition["validation"], labels)
validation_generator = data.DataLoader(validation_set, **params)


for epoch in range(max_epochs):
	for local_batch, local_labels in training_genertor:
		# Transfer to GPU
		local_batch, local_labels = local_batch.to(device), local_labels.to(device)

		print(local_batch)

		# Model:

	with torch.set_grad_enabled(False):
		for local_batch, local_labels in validation_generator:
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			# Model computations
