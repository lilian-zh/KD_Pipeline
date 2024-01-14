from .base_trainer import Off_ST_Trainer

from distillers.KD import DistillKL
from distillers.AT import Attention
from distillers.FitNet import HintLoss
from distillers.RKD import RKDLoss
from distillers.SP import Similarity
from distillers.PKT import PKT

from distillers.util import ConvReg


class KDTrainer(Off_ST_Trainer):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, T, **kwargs):
        self.T = T 
        super().__init__(model_t=model_t, model_s=model_s, train_loader=train_loader, 
                         val_loader=val_loader, num_train=num_train, img_shape=img_shape, class_weights=class_weights,
                         momentum=momentum, weight_decay=weight_decay, lr=lr, lr_decay_rate=lr_decay_rate, 
                         lr_decay_epochs=lr_decay_epochs, step=step, epochs=epochs, save_folder=save_folder, 
                         print_freq=print_freq, alpha=alpha, beta=beta, **kwargs) 

    def _add_module_components(self, **kwargs):
        pass

    def _add_kd_components(self, **kwargs):
        criterion_kd = DistillKL(self.T)
        return criterion_kd

    def _compute_kd_loss(self, **kwargs):
        loss_kd = self.criterion_kd(self.logit_s, self.logit_t)
        return loss_kd
        

class ATTrainer(Off_ST_Trainer):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, **kwargs):
        super().__init__(model_t=model_t, model_s=model_s, train_loader=train_loader, 
                         val_loader=val_loader, num_train=num_train, img_shape=img_shape, class_weights=class_weights,
                         momentum=momentum, weight_decay=weight_decay, lr=lr, lr_decay_rate=lr_decay_rate, 
                         lr_decay_epochs=lr_decay_epochs, step=step, epochs=epochs, save_folder=save_folder, 
                         print_freq=print_freq, alpha=alpha, beta=beta, **kwargs)

    def _add_module_components(self, **kwargs):
        pass

    def _add_kd_components(self, **kwargs):
        criterion_kd = Attention()
        return criterion_kd

    def _compute_kd_loss(self, **kwargs):
        g_s = self.feat_s[1:-1]
        g_t = self.feat_t[1:-1]
        loss_group = self.criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
        return loss_kd


class HintTrainer(Off_ST_Trainer):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, hint_layer, **kwargs):
        self.hint_layer = hint_layer
        super().__init__(model_t=model_t, model_s=model_s, train_loader=train_loader, 
                         val_loader=val_loader, num_train=num_train, img_shape=img_shape, class_weights=class_weights,
                         momentum=momentum, weight_decay=weight_decay, lr=lr, lr_decay_rate=lr_decay_rate, 
                         lr_decay_epochs=lr_decay_epochs, step=step, epochs=epochs, save_folder=save_folder, 
                         print_freq=print_freq, alpha=alpha, beta=beta, **kwargs)

    def _add_module_components(self, **kwargs):
        regress_s = ConvReg(self.feat_s_init[self.hint_layer].shape, self.feat_t_init[self.hint_layer].shape)
        self.module_list.append(regress_s)
        self.trainable_list.append(regress_s)

    def _add_kd_components(self, **kwargs):
        criterion_kd = HintLoss()
        return criterion_kd

    def _compute_kd_loss(self, **kwargs):
        f_s = self.module_list[1](self.feat_s[self.hint_layer])
        f_t = self.feat_t[self.hint_layer]
        loss_kd = self.criterion_kd(f_s, f_t)
        return loss_kd


class RKDTrainer(Off_ST_Trainer):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, w_d, w_a, **kwargs):
        self.w_d=w_d
        self.w_a=w_a
        super().__init__(model_t=model_t, model_s=model_s, train_loader=train_loader, 
                         val_loader=val_loader, num_train=num_train, img_shape=img_shape, class_weights=class_weights,
                         momentum=momentum, weight_decay=weight_decay, lr=lr, lr_decay_rate=lr_decay_rate, 
                         lr_decay_epochs=lr_decay_epochs, step=step, epochs=epochs, save_folder=save_folder, 
                         print_freq=print_freq, alpha=alpha, beta=beta, **kwargs)

    def _add_module_components(self, **kwargs):
        pass

    def _add_kd_components(self, **kwargs):
        criterion_kd = RKDLoss(w_d=self.w_d, w_a=self.w_a)
        return criterion_kd

    def _compute_kd_loss(self, **kwargs):
        f_s = self.feat_s[-1]
        f_t = self.feat_t[-1]
        loss_kd = self.criterion_kd(f_s, f_t)
        return loss_kd


class SPTrainer(Off_ST_Trainer):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, **kwargs):
        super().__init__(model_t=model_t, model_s=model_s, train_loader=train_loader, 
                         val_loader=val_loader, num_train=num_train, img_shape=img_shape, class_weights=class_weights,
                         momentum=momentum, weight_decay=weight_decay, lr=lr, lr_decay_rate=lr_decay_rate, 
                         lr_decay_epochs=lr_decay_epochs, step=step, epochs=epochs, save_folder=save_folder, 
                         print_freq=print_freq, alpha=alpha, beta=beta, **kwargs)

    def _add_module_components(self, **kwargs):
        pass
    
    def _add_kd_components(self, **kwargs):
        criterion_kd = Similarity()
        return criterion_kd

    def _compute_kd_loss(self, **kwargs):
        g_s = [self.feat_s[-2]]
        g_t = [self.feat_t[-2]]
        loss_group = self.criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
        return loss_kd
    
class PKTTrainer(Off_ST_Trainer):
    def __init__(self, model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                 momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                 step, epochs, save_folder, print_freq, alpha, beta, **kwargs):
        super().__init__(model_t=model_t, model_s=model_s, train_loader=train_loader, 
                         val_loader=val_loader, num_train=num_train, img_shape=img_shape, class_weights=class_weights,
                         momentum=momentum, weight_decay=weight_decay, lr=lr, lr_decay_rate=lr_decay_rate, 
                         lr_decay_epochs=lr_decay_epochs, step=step, epochs=epochs, save_folder=save_folder, 
                         print_freq=print_freq, alpha=alpha, beta=beta, **kwargs)

    def _add_module_components(self, **kwargs):
        pass

    def _add_kd_components(self, **kwargs):
        criterion_kd = PKT()
        return criterion_kd

    def _compute_kd_loss(self, **kwargs):
        f_s = self.feat_s[-1]
        f_t = self.feat_t[-1]
        loss_kd = self.criterion_kd(f_s, f_t)
        return loss_kd