"""
This is a boilerplate pipeline 'offline_kd'
generated using Kedro 0.18.11
"""
import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger

from .util import adjust_learning_rate, AverageMeter, accuracy, validate

class Pre_Trainer(object):
    def __init__(self, model, train_loader, val_loader, epochs, save_folder, class_weights,  
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs, print_freq, num_train, **kwargs):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs

        self.save_folder = save_folder
        self.print_freq = print_freq
        self.num_train = num_train

        self.momentum = momentum
        self.weight_decay = weight_decay        
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs

        self.best_acc = 0
        self.best_loss = 10000
        
        save_name = "pretrained_models"

        self.model_folder = os.path.join(self.save_folder, '06_models', 'model_'+save_name)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        self.tb_folder = os.path.join(self.save_folder, '07_model_output', 'tb_'+save_name)
        if not os.path.exists(self.tb_folder):
            os.makedirs(self.tb_folder)
        self.logger = tb_logger.Logger(logdir=self.tb_folder, flush_secs=2)

        self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                            lr=self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion_cls = self.criterion_cls.cuda()
            cudnn.benchmark = True

    def train(self, **kwargs):
        training_process = os.path.join(self.tb_folder,'training_process.txt')
        with open(training_process, 'w') as self.file:
            for epoch in range(1, self.epochs + 1):
                adjust_learning_rate(epoch, self.optimizer, self.lr_decay_epochs, self.lr, self.lr_decay_rate)

                print("epoch {}==> training...".format(epoch))
                tick = time.time()
                train_acc, train_loss = self.train_one_epoch(epoch)
                tock = time.time()
                log_str = ('epoch {}, total training time {:.2f}'.format(epoch, tock - tick))
                print(log_str)
                self.file.write(log_str + '\n')

                self.logger.log_value('train_acc', train_acc, epoch)
                self.logger.log_value('train_loss', train_loss, epoch)

                print("epoch {}==> validating...".format(epoch))
                valid_acc, valid_acc_top5, valid_loss = validate(self.val_loader, self.model, 
                                                                self.criterion_cls, self.print_freq)
                # print('complete validating')

                self.logger.log_value('valid_acc', valid_acc, epoch)
                self.logger.log_value('valid_acc_top5', valid_acc_top5, epoch)
                self.logger.log_value('valid_loss', valid_loss, epoch)

                is_best_acc = valid_acc > self.best_acc
                is_best_loss = valid_loss < self.best_loss
                msg1 = "model: train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f} "

                if is_best_acc:
                    msg2 += " [*]"               
                    self.best_acc = valid_acc
                if is_best_loss:
                    msg2 += " [#]"
                    self.best_loss = valid_loss   

                msg = msg1 + msg2
                print(msg.format(train_loss, train_acc, valid_loss, valid_acc))
                self.file.write(msg.format(train_loss, train_acc, valid_loss, valid_acc) + '\n')

                if is_best_loss:
                    state = {
                        'epoch': epoch,
                        'model': self.model.state_dict(),
                    }
                    save_file = os.path.join(self.model_folder, f'best_loss.pth')
                    # print('saving the best_acc model!')
                    torch.save(state, save_file)

                # regular saving
                if epoch % 10 == 0:
                    print('==> Saving...')
                    state = {
                        'epoch': epoch,
                        'model': self.model.state_dict(),
                    }
                    save_file = os.path.join(self.model_folder, f'ckpt_epoch_{epoch}.pth')
                    torch.save(state, save_file)

            # print('best accuracy:', self.best_acc)
            # print('best loss gap:', self.best_loss)


    def train_one_epoch(self, epoch, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for idx, data in enumerate(self.train_loader):
                input, target, index = data

                data_time.update(time.time() - tic)

                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                # ===================forward=====================
                logit = self.model(input, is_feat=False) 

                loss = self.criterion_cls(logit, target)
                
                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # ===================backward=====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ===================meters=====================
                toc = time.time()
                batch_time.update(toc - tic)
                tic = time.time()

                pbar.set_description(
                    ("{batch_time.avg:.3f}s".format(batch_time=batch_time))
                )
                self.batch_size = input.shape[0]
                pbar.update(self.batch_size)

                # print info
                if idx % self.print_freq == 0:
                    logstr = ('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, idx, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
                    self.file.write(logstr + '\n')
                #     sys.stdout.flush()
            max_memory = torch.cuda.max_memory_allocated()
            print(f' * Max_GPU_Mem {max_memory / 1024**3:.2f}GB')
            self.file.write(f' * Max_GPU_Mem {max_memory / 1024**3:.2f}GB'+ '\n')
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            return top1.avg, losses.avg







