def train(train_loader, model, criterion, optimizer, epoch, print_freq, target_class):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
        input = input.to(device).float()
        if target_class == 0:
            target = target.to(device).long()
        elif target_class == 1:
            target = target.to(device).float()
        # compute output
        output = model(input)

        # print(torch.nonzero(target).size())
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
    TRAIN_LOSS.append(losses.avg)
    print("Epoch {0} : Train: Loss {loss.avg:.4f}\t".format(epoch, loss=losses))
