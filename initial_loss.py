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

