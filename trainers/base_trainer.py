import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger

from abc import ABC, abstractmethod

from .util import adjust_learning_rate, AverageMeter, accuracy, validate
from node_model import create_models
import traceback

class Off_ST_Trainer(ABC):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, **kwargs):

        self.model_t = model_t
        self.model_s = model_s

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_train = num_train

        self.epochs = epochs

        self.save_folder = save_folder
        self.print_freq = print_freq

        self.alpha = alpha
        self.beta = beta

        self.momentum = momentum
        self.weight_decay = weight_decay        
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs

        self.best_acc = 0
        self.best_loss = 10000
        self.n_gpu = torch.cuda.device_count()

        save_name = "step_"+str(step)
        self.tb_folder = os.path.join(self.save_folder, '07_model_output', 'tb_'+save_name)
        if not os.path.exists(self.tb_folder):
            os.makedirs(self.tb_folder)
        self.logger = tb_logger.Logger(logdir=self.tb_folder, flush_secs=2)

        self.model_folder = os.path.join(self.save_folder, '06_models', 'model_'+save_name)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        self.model_t.eval()
        self.model_s.eval()

        data = torch.randn(2, img_shape[0], img_shape[1], img_shape[2])
        self.feat_t_init, self.logit_t_init = model_t(data, is_feat=True)
        self.feat_s_init, self.logit_s_init = model_s(data, is_feat=True)

        self.module_list = nn.ModuleList([self.model_s])
        self.trainable_list = nn.ModuleList([self.model_s])
        self._add_module_components(**kwargs)

        
        self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion_kd = self._add_kd_components(**kwargs)
        
        # optimizer
        self.optimizer = optim.SGD(self.trainable_list.parameters(),
                            lr=self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay)
        
        # append teacher after optimizer to avoid weight_decay
        self.module_list.append(self.model_t)
        
        if torch.cuda.is_available():
            self._move_to_gpu()

        # validate teacher accuracy
        # teacher_acc, _, _ = validate(self.val_loader, self.model_t, self.criterion_cls, self.print_freq)
        # print('teacher accuracy: {:.3f}'.format(teacher_acc))

    def _move_to_gpu(self):
        for i in range(len(self.module_list)):
            self.module_list[i] = DataParallel(self.module_list[i]) if self.n_gpu > 1 else self.module_list[i]
            self.module_list[i] = self.module_list[i].cuda()
        self.criterion_cls = self.criterion_cls.cuda()
        self.criterion_kd = self.criterion_kd.cuda()
        cudnn.benchmark = True


    @abstractmethod
    def _add_kd_components(self, **kwargs):
        pass
    
    @abstractmethod
    def _add_module_components(self, **kwargs):
        pass

    def train(self, **kwargs):
        training_process = os.path.join(self.tb_folder,'training_process.txt')
        with open(training_process, 'w') as self.file:
            for epoch in range(1, self.epochs + 1):
                adjust_learning_rate(epoch, self.optimizer, self.lr_decay_epochs, self.lr, self.lr_decay_rate)

                print(f"epoch {epoch}==> training...")

                tick = time.time()
                train_acc, train_loss = self.train_one_epoch(epoch, **kwargs)
                tock = time.time()
                log_str = ('epoch {}, total time {:.2f}'.format(epoch, tock - tick))
                print(log_str)
                self.file.write(log_str + '\n')

                self.logger.log_value('train_acc', train_acc, epoch)
                self.logger.log_value('train_loss', train_loss, epoch)

                print(f"epoch {epoch}==> validating...")
                valid_acc, valid_acc_top5, valid_loss = validate(self.val_loader, self.model_s, 
                                                                self.criterion_cls, self.print_freq)

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
                        'model': self.model_s.state_dict(),
                    }
                    save_file = os.path.join(self.model_folder, 'best_loss.pth')
                    # print('saving the mini loss model!')
                    torch.save(state, save_file)

                # regular saving
                if epoch % 10 == 0:
                    print('==> Saving...')
                    state = {
                        'epoch': epoch,
                        'model': self.model_s.state_dict(),
                    }
                    save_file = os.path.join(self.model_folder, f'ckpt_epoch_{epoch}.pth')
                    torch.save(state, save_file)

        # print('best accuracy: {:.3f}'.format(self.best_acc))
        # print('best loss: {:.3f}'.format(self.best_loss))

        return self.best_acc


    def train_one_epoch(self, epoch, **kwargs):
        # 在开始训练前重置 GPU 内存状态
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        # set modules as train()
        for module in self.module_list:
            module.train()
        # set teacher as eval()
        self.module_list[-1].eval()

        model_s = self.module_list[0]
        model_t = self.module_list[-1]

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        with tqdm(total=self.num_train) as pbar:
            for idx, data in enumerate(self.train_loader):
                try:
                    input, target, index = data
                except Exception as e:
                    # 捕获异常信息并记录到日志文件
                    with open('error_log_train.txt', 'a') as error_file:
                        error_file.write(f"Error occurred at index {idx}: {str(e)}\n")
                        error_file.write(traceback.format_exc())

                    # 继续处理下一个数据
                    continue                    

                data_time.update(time.time() - end)

                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()
                    index = index.cuda()

                # ===================forward=====================
                self.feat_s, self.logit_s = model_s(input, is_feat=True) 
                with torch.no_grad():
                    self.feat_t, self.logit_t = model_t(input, is_feat=True)
                    self.feat_t = [f.detach() for f in self.feat_t]

                # cls + kl div
                loss_cls = self.criterion_cls(self.logit_s, target)
                loss_kd = self._compute_kd_loss(**kwargs)
                # print("loss_cls:",loss_cls)
                # print("loss_kd:",loss_kd)

                loss = self.alpha * loss_cls + self.beta * loss_kd
                
                acc1, acc5 = accuracy(self.logit_s, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # ===================backward=====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ===================meters=====================
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_description(
                    ("{batch_time.avg:.3f}s".format(batch_time=batch_time))
                )
                self.batch_size = input.shape[0]
                pbar.update(self.batch_size)

                # print info
                if idx % self.print_freq == 0:
                    logstr = ('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, idx, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
                    self.file.write(logstr + '\n')
                #     sys.stdout.flush()
            max_memory = torch.cuda.max_memory_allocated()
            # print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Max_GPU_Mem {max_memory / 1024**3:.2f}GB')
            print(f' * Max_GPU_Mem {max_memory / 1024**3:.2f}GB')
            self.file.write(f' * Max_GPU_Mem {max_memory / 1024**3:.2f}GB'+ '\n')
            return top1.avg, losses.avg
        
    @abstractmethod
    def _compute_kd_loss(self, **kwargs):
        pass


class On_DML_Trainer(ABC):
    def __init__(self, train_loader, val_loader, num_train, class_weights, momentum, weight_decay, 
                 lr, lr_decay_rate, lr_decay_epochs, model_name, num_classes, model_attributes, 
                 num_models, epochs, save_folder, print_freq, alpha, beta, **kwargs):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_train = num_train

        self.epochs = epochs

        self.save_folder = save_folder
        self.print_freq = print_freq

        self.alpha = alpha
        self.beta = beta

        self.momentum = momentum
        self.weight_decay = weight_decay        
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs
              
        self.num_models = num_models
        self.models = []
        self.optimizers = []
        self.schedulers = []

        self.best_valid_accs = [0.] * self.num_models
        self.n_gpu = torch.cuda.device_count()

        self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion_kd = self._add_kd_components(**kwargs)

        if torch.cuda.is_available():
            device = torch.device("cuda")  # 选择设备为GPU
            self.criterion_cls = self.criterion_cls.to(device)
            self.criterion_kd = self.criterion_kd.to(device)

        self.save_name = "DML_"+str(self.num_models)
        self.model_folder = os.path.join(self.save_folder, '06_models', 'model_'+self.save_name)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)  

        self.tb_folder = os.path.join(self.save_folder, '07_model_output', 'tb_'+self.save_name)     
        if not os.path.exists(self.tb_folder):
            os.makedirs(self.tb_folder)
        self.logger = tb_logger.Logger(logdir=self.tb_folder, flush_secs=2)

        self.models = create_models(model_name, self.num_models, num_classes, model_attributes)
        for idx, model in enumerate(self.models):
            if torch.cuda.is_available():
                if self.n_gpu > 1:
                    self.models[idx] = nn.DataParallel(model)
                self.models[idx] = self.models[idx].cuda()       

            # initialize optimizer and scheduler
            optimizer = optim.SGD(model.parameters(), lr=self.lr, 
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)            
            self.optimizers.append(optimizer)

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.models[0].parameters()])))


    @abstractmethod
    def _add_kd_components(self, **kwargs):
        pass


    def train(self, **kwargs):
        training_process = os.path.join(self.tb_folder,'training_process.txt')
        with open(training_process, 'w') as self.file:
            for epoch in range(1, self.epochs + 1):
                for optimizer in self.optimizers:
                    adjust_learning_rate(epoch, optimizer, self.lr_decay_epochs, self.lr, self.lr_decay_rate)

                # evaluate on training set
                print(f"epoch {epoch}==> training...")
                tick = time.time()
                train_losses, train_accs = self.train_one_epoch(epoch)
                tock = time.time()
                log_str = ('epoch {}, total training time {:.2f}'.format(epoch, tock - tick))
                print(log_str)
                self.file.write(log_str + '\n')
                # log to tensorboard
                for i in range(self.num_models):
                    self.logger.log_value('train_loss_%d' % (i+1), train_losses[i].avg, epoch)
                    self.logger.log_value('train_acc_%d' % (i+1), train_accs[i].avg, epoch)

                # evaluate on validation set
                print(f"epoch {epoch}==> validating...")
                valid_accs, valid_losses = self.validate(epoch)
                # print('complete validating')
                # log to tensorboard for every epoch
                for i in range(self.num_models):
                    self.logger.log_value('valid_loss_%d' % (i+1), valid_losses[i].avg, epoch)
                    self.logger.log_value('valid_acc_%d' % (i+1), valid_accs[i].avg, epoch)

                for i in range(self.num_models):
                    is_best = valid_accs[i].avg> self.best_valid_accs[i]
                    msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
                    msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
                    if is_best:
                        msg2 += " [*]"
                    msg = msg1 + msg2
                    print(msg.format(i+1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))
                    self.file.write(msg.format(i+1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg) + '\n')               
                    if is_best:
                        self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                        best_state = {'epoch': epoch,
                            'model': self.models[i].state_dict(),
                            # 'optim_state': self.optimizers[i].state_dict(),
                            # 'best_valid_acc': self.best_valid_accs[i],
                            }        
                        best_model = os.path.join(self.model_folder, 'model_{model}_best.pth'.format(model=i+1))
                        # print('saving the best models!')
                        torch.save(best_state, best_model)

        for i in range(self.num_models):
            last_state = {'model': self.models[i].state_dict(),
                          'optim_state': self.optimizers[i].state_dict(),
                    }
            last_model = os.path.join(self.model_folder, 'model_{model}_last.pth'.format(model=i+1))
            # print('saving the last models!')
            torch.save(last_state, last_model)

    def train_one_epoch(self, epoch, **kwargs):
        # 在开始训练前重置 GPU 内存状态
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        batch_time = AverageMeter()
        losses = []
        accs = []

        for i in range(self.num_models):
            self.models[i].train()
            losses.append(AverageMeter())
            accs.append(AverageMeter())
      
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for idx, data in enumerate(self.train_loader):
                input, target, index = data

                input = input.float()
                if torch.cuda.is_available():
                    device = torch.device("cuda")  # 选择设备为GPU
                    input = input.to(device)
                    target = target.to(device)
                    index = index.to(device)
                
                #forward pass
                self.outputs=[]
                for model in self.models:                   
                    logit = model(input, is_feat=False)
                    self.outputs.append(logit)

                for i in range(self.num_models):
                    loss_cls = self.criterion_cls(self.outputs[i], target)
                    loss_kd = 0
                    for j in range(self.num_models):
                        if i!=j:
                            # loss_kd += self.criterion_kd(outputs[i], outputs[j].detach())
                            loss_kd = self._compute_kd_loss(loss_kd, i, j, **kwargs)
                    loss = self.alpha * loss_cls + self.beta * (loss_kd / (self.num_models - 1))
                    
                    # measure accuracy and record loss
                    acc = accuracy(self.outputs[i], target, topk=(1,))[0]
                    # acc1, acc5 = accuracy(outputs[i], target, topk=(1, 5))
                    losses[i].update(loss.item(), input.size(0))
                    accs[i].update(acc.item(), input.size(0))
                

                    # ===================backward=====================
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(
                    ("{batch_time.avg:.3f}s".format(batch_time=batch_time))
                )
                self.batch_size = input.shape[0]
                pbar.update(self.batch_size)

                # # print info
                for i in range(self.num_models):
                    if idx % self.print_freq == 0:
                        log_str = ('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, idx, self.num_train, batch_time=batch_time,
                            loss=losses[i], top1=accs[i]))
                        self.file.write(log_str + '\n')
                        # sys.stdout.flush()

            max_memory = torch.cuda.max_memory_allocated()
            print(f' * Max_GPU_Mem {max_memory / 1024**3:.2f}GB')
            self.file.write(f' * Max_GPU_Mem {max_memory / 1024**3:.2f}GB'+ '\n')
    
            return losses, accs

    @abstractmethod
    def _compute_kd_loss(self, **kwargs):
        pass


    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        batch_time = AverageMeter()
        losses = []
        accs = []
        for i in range(self.num_models):
            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        with torch.no_grad():
            end = time.time()
            for idx, (input, target, _) in enumerate(self.val_loader):
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                #forward pass
                outputs=[]
                for model in self.models:
                    outputs.append(model(input))
                for i in range(self.num_models):
                    loss = self.criterion_cls(outputs[i], target)

                    # measure accuracy and record loss
                    acc = accuracy(outputs[i], target, topk=(1,))[0]
                    losses[i].update(loss.item(), input.size(0))
                    accs[i].update(acc.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # print info
                for i in range(self.num_models):
                    if idx % self.print_freq == 0:
                        print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, idx, self.num_train, batch_time=batch_time,
                            loss=losses[i], top1=accs[i]))
                        sys.stdout.flush()

            for i in range(self.num_models):
                print(' * Acc@1 {top1.avg:.3f} Loss {loss.val:.4f}'
                    .format(top1=accs[i], loss=losses[i]))

        return accs, losses 

