import os
import torch
import numpy as np
from torch.utils import data
import torch.nn as nn
from data.Dataset import Dataset
from networks.SegNet import SegNet
from torchsummary import summary


# *************************************************** INPUT PARAMETERS ******************************************************#

# TODO: Change to parse args
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
BATCH_SIZE = 2
LEARNING_RATE = 0.001
MAX_EPOCHS = 2


# ************************************************* INITIALIZE NETWORK ******************************************************#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


training_set = Dataset(partition["train"])
training_generator = data.DataLoader(
    training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=20
)

validation_set = Dataset(partition["validation"])
validation_generator = data.DataLoader(
    validation_set, batch_size=BATCH_SIZE, shuffle=False
)

# initialize model
model = SegNet().to(device)
model = model.double()

# set loss function and optimizer
# TODO: weigths
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# *************************************************** TRAINING LOOP ******************************************************#

TRAIN_LOSS, VAL_LOSS = [], []

for epoch in range(MAX_EPOCHS):
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        local_batch = local_batch.unsqueeze(1).double()
        local_labels = local_labels.unsqueeze(1).double()

        # Forward pass
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        TRAIN_LOSS.append(loss)
        # TODO: save loss as function of number of dark matter particles per voxel

    if (epoch + 1) % 1 == 0:
        print("Epoch [{}/{}],  Loss: {:.4f}".format(epoch + 1, MAX_EPOCHS, loss.item()))

    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch = local_batch.unsqueeze(1).double()
            local_labels = local_labels.unsqueeze(1).double()

            outputs = model(local_batch)
            val_loss = criterion(outputs, local_labels)
            VAL_LOSS.append(val_loss)


# Save training summary
# Save train/val losses
