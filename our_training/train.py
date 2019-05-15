import os
import torch
import numpy as np
import pandas as pd
from torch.utils import data
import torch.nn as nn
from Dataset import Dataset
from  Models import SegNet
import matplotlib
matplotlib.use('GTK')

# CUDA for pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#device = "cpu"

#cudnn.benchmark = True

# Parameters
params = {"batch_size": 3,
	"shuffle": True,
	"num_workers": 20
	}

max_epochs = 2 
learning_rate = 0.001

partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
#labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}


training_set = Dataset(partition["train"])
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition["validation"])
validation_generator = data.DataLoader(validation_set, **params)

#unique_values = list(set(labels.values()))
#num_classes = len(unique_values)

num_classes = 1 # For regression

model = SegNet(num_classes, 1, 1.).to(device)

model = model.double()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(training_generator)

for epoch in range(max_epochs):
	for local_batch, local_labels in training_generator:
		# Transfer to GPU
		local_batch, local_labels = local_batch.to(device), local_labels.to(device)
		
		# Forward pass
		local_batch = local_batch.unsqueeze(1).double()
		local_labels = local_labels.unsqueeze(1).double()
		outputs = model(local_batch)
		loss = criterion(outputs, local_labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch+1) % 1 == 0:
		print ('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch+1, max_epochs,  loss.item()))
	

	with torch.set_grad_enabled(False):
		for local_batch, local_labels in validation_generator:
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			# Model computations
