def validate(val_loader, model, criterion, epoch, target_class, save_name):
    global BEST_VAL_LOSS
    global BEST_RECALL
    global BEST_PRECISION
    global BEST_F1SCORE
    global BEST_ACC
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    TPRs = AverageMeter()
    FPRs = AverageMeter()
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device).float()
            if target_class == 0:
                target = target.to(device).long()
            elif target_class == 1:
                target = target.to(device).float()

            # compute output
            output = model(input)
            if target_class == 0:
                outputs = F.softmax(output, dim=1)
                predicted = outputs.max(1)[1]
                total += np.prod(target.shape)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                TPR, gp, FPR, gf = confusion_matrix_calc(predicted, target)
                TPRs.update(TPR, gp)
                FPRs.update(FPR, gf)
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))

    if target_class == 0:
        recall = TPRs.avg * 100
        precision = TPRs.sum / (TPRs.sum + FPRs.sum + EPSILON) * 100
        F1score = 2 * ((precision * recall) / (precision + recall + EPSILON))
        acc = correct / total * 100
        VAL_RECALL.append(recall)
        VAL_ACC.append(acc)
        VAL_PRECISION.append(precision)
        if val_losses.avg < BEST_VAL_LOSS:
            BEST_RECALL = recall
            BEST_PRECISION = precision
            BEST_F1SCORE = F1score
            BEST_ACC = acc

    if val_losses.avg < BEST_VAL_LOSS:
        if len(save_name) > 0:
            # torch.save(model, 'pretrained/' + str(save_name) + '.pt')
            torch.save(model.state_dict(), "pretrained/" + str(save_name) + ".pth")
        BEST_VAL_LOSS = val_losses.avg
    VAL_LOSS.append(val_losses.avg)
    if target_class == 0:
        print(
            "Epoch {0} :Val Loss {val_losses.avg:.4f},\
         Val Accuracy {acc:.4f},  Val Recall {recall:.4f}\t Precision {precision:.4f} F1 score  {F1score:.4f}\t".format(
                epoch,
                val_losses=val_losses,
                acc=acc,
                recall=recall,
                precision=precision,
                F1score=F1score,
            )
        )
    else:
        print(
            "Epoch {0} : Val Loss {val_losses.avg:.4f}".format(
                epoch, val_losses=val_losses
            )
        )
