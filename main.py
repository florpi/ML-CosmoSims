import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F


from torch.utils import data
import random
import numpy as np
from itertools import product

import train
import validate
import initial_loss
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/Dark2Light/training/')
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/Dark2Light/data/')
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/Dark2Light/main/')
from train_f import *
from Dataset import Dataset
from Models import *
from args import args
from parse_args import parse_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# the following variables are global variables that record the statistics
# for each epoch so that the plot can be produced
TRAIN_LOSS, VAL_LOSS, VAL_ACC, VAL_RECALL, VAL_PRECISION = [], [], [], [], []
BEST_VAL_LOSS = 999999999
BEST_RECALL = 0
BEST_PRECISION = 0
BEST_F1SCORE = 0
BEST_ACC = 0
EPSILON = 1e-5



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # input flags
    args = parse_args()
    lr = args.lr
    model_idx = args.model_idx
    epochs = args.epochs
    batch_size = args.batch_size
    loss_weight = args.loss_weight
    weight_decay = args.weight_decay
    print_freq = args.print_freq
    target_cat = args.target_cat
    target_class = args.target_class
    plot_label = args.plot_label
    load_model = args.load_model
    save_name = args.save_name
    record_results = args.record_results
    vel = args.vel  # not considered for now
    normalize = args.normalize

    data_range = 1024
    random_idx = 0

    # Initialize datasets
    training_set = Dataset(
            train_data,
            cat=target_cat,
            reg=target_class,
            vel=vel
    )
    validation_set = Dataset(
            val_data,
            cat=target_cat,
            reg=target_class,
            vel=vel,
            normalize=normalize
    )
    testing_set = Dataset(
            test_data,
            cat=target_cat,
            reg=target_class,
            vel=vel,
            normalize=normalize
    )

    # Define which cubes go into which dataset (train, test, validate)
    pos = list(np.arange(0, data_range, 32))
    ranges = list(product(pos, repeat=3))
    random.seed(7)
    if random_idx == 1:
        random.shuffle(ranges)
        train_data = ranges[: int(np.round(len(ranges) * 0.6))]
        val_data = ranges[
            int(np.round(len(ranges) * 0.6)) : int(np.round(len(ranges) * 0.8))
        ]
        test_data = ranges[int(np.round(len(ranges) * 0.8)) :]
    else:
        train_data, val_data, test_data = [], [], []

        for i in range(0, data_range, 32):
            for j in range(0, data_range, 32):
                for k in range(0, data_range, 32):
                    idx = (i, j, k)
                    if i <= 416 and j <= 416:
                        val_data.append(idx)
                    elif i >= 484 and j >= 448 and k >= 448:
                        test_data.append(idx)
                    else:
                        train_data.append(idx)

    # Load dataset
    params = {"batch_size": batch_size, "shuffle": True, "num_workers": 20}
    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)

    # #set up device

    # #build model
    dim_out = 1
    model = SimpleUnet(dim, target_class).to(device)

    model = torch.load("pretrained/mytraining.pt")
    criterion = nn.CrossEntropyLoss(
        weight=get_loss_weight(loss_weight, num_class=2)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    initial_loss(
        training_generator, validation_generator, model, criterion, target_class
    )

    for epoch in range(epochs):
        adjust_learning_rate(lr, optimizer, epoch)
        train(
            training_generator,
            model,
            criterion,
            optimizer,
            epoch,
            print_freq,
            target_class=target_class,
        )
        # evaluate on validation set
        validate(
            validation_generator,
            model,
            criterion,
            epoch,
            target_class=target_class,
            save_name=save_name,
        )

    if len(plot_label) == 0:
        plot_label = "_" + str(target_class) + "_" + str(model_idx) + "_"

    train_plot(
        TRAIN_LOSS,
        VAL_LOSS,
        VAL_ACC,
        VAL_RECALL,
        VAL_PRECISION,
        target_class,
        plot_label=plot_label,
    )

    if target_class == 0:
        if record_results:
            args = parse_args()
            f = open("all_results", "a+")
            f.write("arguments: %s" % (args) + "\n")
            f.write(
                "Test Loss {BEST_VAL_LOSS:.5f},\
                 Test Accuracy {BEST_ACC:.4f},\
                 Test Recall {BEST_RECALL:.4f},\
                 Precision {BEST_PRECISION:.4f},\
                 F1 score  {BEST_F1SCORE:.4f}\n".format(
                    BEST_VAL_LOSS=BEST_VAL_LOSS,
                    BEST_ACC=BEST_ACC,
                    BEST_RECALL=BEST_RECALL,
                    BEST_PRECISION=BEST_PRECISION,
                    BEST_F1SCORE=BEST_F1SCORE,
                )
            )
            f.close()


if __name__ == "__main__":
    main()
