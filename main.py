import argparse
import os, sys, glob, time
import random
import shutil
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

from parse_args import parse_args
import train
import validate
import initial_loss

sys.path.insert(0, "/cosma7/data/dp004/dc-beck3/Dark2Light/data/")
from Dataset import Dataset

sys.path.insert(0, "/cosma7/data/dp004/dc-beck3/Dark2Light/networks/")
from Models import *

# sys.path.insert(0, '/cosma7/data/dp004/dc-beck3/Dark2Light/training/')
# from train_f import *
    
# Find out if GPUs or CPUs are available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("This computer will use: %s" % device)

# the following variables are global variables that record the statistics
# for each epoch so that the plot can be produced
TRAIN_LOSS, VAL_LOSS, VAL_ACC, VAL_RECALL, VAL_PRECISION = [], [], [], [], []
BEST_VAL_LOSS = 999999999
BEST_RECALL = 0
BEST_PRECISION = 0
BEST_F1SCORE = 0
BEST_ACC = 0
EPSILON = 1e-5

def initial_loss(train_loader, val_loader, model, criterion, target_class):
    # AverageMeter is a object that record the sum,
    # avg, count and val of the target stats
    train_losses = AverageMeter()
    val_losses = AverageMeter()
    correct = 0
    # ptotal = 0  #count of all positive predictions
    # tp = 0    #true positive
    total = 0  # total count of data
    TPRs = AverageMeter()
    FPRs = AverageMeter()
    # switch to train mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
            input = input.to(device).float()
            if target_class == 0:
                target = target.to(device).long()
            elif target_class == 1:
                target = target.to(device).float()
            # compute output
            output = model(input)
            # print("target1: ", target.size())
            # print("output: ", output.size())
            loss = criterion(output, target)
            # measure accuracy and record loss
            train_losses.update(loss.item(), input.size(0))

        for i, (input, target) in enumerate(val_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
            input = input.to(device).float()
            if target_class == 0:
                target = target.to(device).long()
            elif target_class == 1:
                target = target.to(device).float()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))
            if target_class == 0:
                outputs = F.softmax(output, dim=1)
                predicted = outputs.max(1)[1]
                total += np.prod(target.shape)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                TPR, gp, FPR, gf = confusion_matrix_calc(predicted, target)
                TPRs.update(TPR, gp)
                FPRs.update(FPR, gf)
            loss = criterion(output, target)
            val_losses.update(loss.item(), input.size(0))

    if target_class == 0:
        acc = correct / total * 100
        recall = TPRs.avg * 100
        precision = TPRs.sum / (TPRs.sum + FPRs.sum + EPSILON) * 100
        VAL_RECALL.append(recall)
        VAL_ACC.append(acc)
        VAL_PRECISION.append(precision)

    TRAIN_LOSS.append(train_losses.avg)
    VAL_LOSS.append(val_losses.avg)
    if target_class == 0:
        print(
            "Epoch Train Loss {train_losses.avg:.4f},\
             Test Loss {val_losses.avg:.4f},\
             Test Accuracy {acc:.4f},\
             Test Recall {recall:.4f}\t Precision {precision:.4f}\t".format(
                train_losses=train_losses,
                val_losses=val_losses,
                acc=acc,
                recall=recall,
                precision=precision,
            )
        )
    else:
        print(
            "Epoch Train Loss {train_losses.avg:.4f},\
             Test Loss {val_losses.avg:.4f}".format(
                train_losses=train_losses, val_losses=val_losses
            )
        )


def main():
    # Input flags
    ## Parameters for datasets
    args = parse_args()
    data_path = args.features_path
    label_path = args.targets_path
    snapshot_nr = args.snapshot_nr
    voxle_nr = args.voxle_nr
    ## Parameters for ML-algorithm
    lr = args.lr
    model_idx = args.model_idx
    epochs = args.epochs
    batch_size = args.batch_size
    loss_weight = args.loss_weight
    weight_decay = args.weight_decay
    print_freq = args.print_freq
    label_type = args.label_type
    target_class = args.target_class
    load_model = args.load_model
    save_name = args.save_name
    record_results = args.record_results
    vel = args.vel  # not considered for now
    normalize = args.normalize

    # Define which cubes go into which dataset (train, test, validate)
    random_idx = 1
    pos = list(np.arange(0, voxle_nr, 32))  # sub-divide each voxel by 32^3
    ranges = list(product(pos, repeat=3))  # all combinations of 3 elements
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

    # print('train_data', train_data[:10], len(train_data))
    # print('test_data', test_data[:10], len(test_data))

    # Initialize datasets
    training_set = Dataset(
        train_data,
        "%sdark_matter_only_s%d_v%d_dm.npy" % (data_path, snapshot_nr, voxle_nr),
        "%sfull_physics_s%d_v%d_dm.npy" % (label_path, snapshot_nr, voxle_nr),
        cat=label_type,
        reg=target_class,
    )
    validation_set = Dataset(
        val_data,
        "%sdark_matter_only_s%d_v%d_dm.npy" % (data_path, snapshot_nr, voxle_nr),
        "%sfull_physics_s%d_v%d_dm.npy" % (label_path, snapshot_nr, voxle_nr),
        cat=label_type,
        reg=target_class,
        normalize=normalize,
    )
    testing_set = Dataset(
        test_data,
        "%sdark_matter_only_s%d_v%d_dm.npy" % (data_path, snapshot_nr, voxle_nr),
        "%sfull_physics_s%d_v%d_dm.npy" % (label_path, snapshot_nr, voxle_nr),
        cat=label_type,
        reg=target_class,
        normalize=normalize,
    )
    # Load dataset
    params = {"batch_size": batch_size, "shuffle": True, "num_workers": 20}
    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)

    # Build model
    dim_out = 1
    if vel == 1:
        dim_in = 4
    else:
        dim_in = 1
        dim = 1  # need to be changed later

    model = SimpleUnet(dim, target_class).to(device)

    if target_class == 0:
        # Classifier
        criterion = nn.CrossEntropyLoss(
            weight=get_loss_weight(loss_weight, num_class=2)
        ).to(device)
    else:
        # Regresssion
        criterion = weighted_nn_loss(loss_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    initial_loss(
        training_generator, validation_generator, model, criterion, target_class
    )

    # for epoch in range(epochs):
    #    adjust_learning_rate(lr, optimizer, epoch)
    #    train(
    #        training_generator,
    #        model,
    #        criterion,
    #        optimizer,
    #        epoch,
    #        print_freq,
    #        target_class=target_class,
    #    )
    #    # evaluate on validation set
    #    validate(
    #        validation_generator,
    #        model,
    #        criterion,
    #        epoch,
    #        target_class=target_class,
    #        save_name=save_name,
    #    )

    # if len(plot_label) == 0:
    #    plot_label = "_" + str(target_class) + "_" + str(model_idx) + "_"

    # train_plot(
    #    TRAIN_LOSS,
    #    VAL_LOSS,
    #    VAL_ACC,
    #    VAL_RECALL,
    #    VAL_PRECISION,
    #    target_class,
    #    plot_label=plot_label,
    # )

    # if target_class == 0:
    #    if record_results:
    #        args = parse_args()
    #        f = open("all_results", "a+")
    #        f.write("arguments: %s" % (args) + "\n")
    #        f.write(
    #            "Test Loss {BEST_VAL_LOSS:.5f},\
    #             Test Accuracy {BEST_ACC:.4f},\
    #             Test Recall {BEST_RECALL:.4f},\
    #             Precision {BEST_PRECISION:.4f},\
    #             F1 score  {BEST_F1SCORE:.4f}\n".format(
    #                BEST_VAL_LOSS=BEST_VAL_LOSS,
    #                BEST_ACC=BEST_ACC,
    #                BEST_RECALL=BEST_RECALL,
    #                BEST_PRECISION=BEST_PRECISION,
    #                BEST_F1SCORE=BEST_F1SCORE,
    #            )
    #        )
    #        f.close()


if __name__ == "__main__":
    main()
