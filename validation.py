import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import sys

from utils import AverageMeter, calculate_accuracy

def per_class_stats_writer(predictions, targets, class_map, epoch):

    epoch_report = []
    # Apply Softmax to the predictions
    predictions = F.softmax(predictions, dim=1)
    predictions = torch.argmax(predictions, 1)

    predictions = predictions.cpu()
    targets = targets.cpu()

    for index, class_name in class_map.items():
        idx = np.where(targets == index)[0]
        epoch_report.append({"class": class_map[index],
                         "total": idx.shape[0]})

        # find correct ones
        epoch_report[index]["correct"] = np.count_nonzero(predictions[idx] == index)
        epoch_report[index]["accuracy"] = round(epoch_report[index]["correct"] / epoch_report[index]["total"], 3)

        bincounts = np.bincount(predictions[idx])
        class_list_descending = np.argsort(bincounts)[::-1]

        epoch_report[index]["predict1"] = class_map[class_list_descending[0]]

        if(class_list_descending.shape[0] > 1):
            epoch_report[index]["predict2"] = class_map[class_list_descending[1]]

        if(class_list_descending.shape[0] > 2):
            epoch_report[index]["predict3"] = class_map[class_list_descending[2]]

    if not os.path.isdir("per_class_reports"):
        os.mkdir("per_class_reports")

    filename = "per_class_reports/epoch_%s.csv" % (epoch)

    with open(filename, 'a') as f:
        writer = csv.DictWriter(f, ['class', 'correct', 'total',
                                    'accuracy', 'predict1', 'predict2',
                                    'predict3'])

        writer.writeheader()
        writer.writerows(epoch_report)

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    class_map = {k:v for k,v in data_loader.dataset.class_names.items()}

    output_all = []
    target_all = []

    with torch.no_grad():
        end_time = time.time()
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)

            if opt.per_class_reports:
                output_all.append(outputs)
                target_all.append(targets)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))

        if opt.per_class_reports:
            per_class_stats_writer(output_all, target_all, class_map, epoch)            

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
