from __future__ import print_function

import torch
import torch.nn as nn

import os
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import traceback


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, optimizer, lr_decay_epochs, lr, lr_decay_rate):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = lr * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) # updated by Li,  replacing the .view() method with the .reshape()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion, print_freq):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # for idx, (input, target, _) in enumerate(val_loader):
        for idx, data in enumerate(val_loader):
            try:
                input, target, index = data
            except Exception as e:
                # 捕获异常信息并记录到日志文件
                with open('error_log_valid.txt', 'a') as error_file:
                    error_file.write(f"Error occurred at index {idx}: {str(e)}\n")
                    error_file.write(traceback.format_exc())

                # 继续处理下一个数据
                continue 

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg




def test(test_loader, model, label_map_file, output_dir='./'):
    with open(label_map_file, 'r') as file:
        label_map = {int(line.split()[0]): line.split()[1] for line in file}
    true_labels = [label_map[i] for i in range(len(label_map))]
    predicted_labels = [label_map[i] for i in range(len(label_map))]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    criterion_cls = nn.CrossEntropyLoss()

    # Move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_metrics = os.path.join(output_dir,'metrics.txt')
    with torch.no_grad(), open(output_metrics, 'w') as file:
        end = time.time()
        all_predictions = []
        all_targets = []

        for idx, (input, target, _) in enumerate(test_loader):
            input = input.to(device=device, dtype=torch.float)
            target = target.to(device=device)

            # compute output
            output = model(input)
            loss = criterion_cls(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # get predictions and targets
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 50 == 0:
                log_str = (f'Test: [{idx}/{len(test_loader)}]\t'
                           f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                           f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
                print(log_str)
                file.write(log_str + '\n')

        # Compute Precision, Recall, and F1-Score
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        # Compute Confusion Matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)

        # Write metrics to the file
        file.write(f'Top1 Accuracy: {top1.avg:.3f}\n')
        file.write(f'Top5 Accuracy: {top5.avg:.3f}\n')
        file.write(f'Losses Average: {losses.avg:.4f}\n')
        file.write(f'Precision: {precision:.4f}\n')
        file.write(f'Recall: {recall:.4f}\n')
        file.write(f'F1 Score: {f1:.4f}\n')

        # Save Confusion Matrix
        # np.savetxt(os.path.join(output_dir, 'confusion_matrix.csv'), conf_matrix.astype(int), fmt='%d', delimiter=',')

        # Compute classification report
        class_report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=1)

        # Extract metrics for each class
        class_metrics = {}
        for class_idx, metrics in class_report.items():
            if class_idx.isdigit():  # Ensure it's a class index
                class_name = class_idx  # Change this line if class labels are strings
                class_metrics[class_name] = {
                    'Top1 Accuracy': metrics['precision'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1-score']
                }

        # Save class-wise metrics to confusion_matrix.csv
        csv_file_path = os.path.join(output_dir, 'confusion_matrix.csv')
        with open(csv_file_path, 'w') as csv_file:
            # Write header for metrics
            csv_file.write('Class,Top1 Accuracy,Precision,Recall,F1 Score\n')

            # Write metrics for each class
            for class_name, metrics in class_metrics.items():
                csv_file.write(f'{class_name},{metrics["Top1 Accuracy"]},{metrics["Precision"]},'
                            f'{metrics["Recall"]},{metrics["F1 Score"]}\n')

            # Write confusion matrix
            csv_file.write('\nConfusion Matrix:\n')
            np.savetxt(csv_file, conf_matrix.astype(int), fmt='%d', delimiter=',')


        # Create a heatmap for Confusion Matrix
        plt.figure(figsize=(8, 6))
        # plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=predicted_labels, yticklabels=true_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix Heatmap')

        # Save the heatmap as an image file
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_heatmap.png'))
        plt.close()