from .base_trainer import On_DML_Trainer

from distillers.KD import DistillKL


class DML_Trainer(On_DML_Trainer):
    def __init__(self, train_loader, val_loader, num_train, class_weights, momentum, weight_decay, 
                 lr, lr_decay_rate, lr_decay_epochs, model_name, num_classes, model_attributes, 
                 num_models, epochs, save_folder, print_freq, alpha, beta, T, **kwargs):
        self.T = T 
        super().__init__(train_loader=train_loader, val_loader=val_loader, num_train=num_train, 
                         class_weights=class_weights, momentum=momentum, weight_decay=weight_decay, lr=lr, 
                         lr_decay_rate=lr_decay_rate, lr_decay_epochs=lr_decay_epochs, model_name=model_name, 
                         num_classes=num_classes, model_attributes=model_attributes, num_models=num_models, 
                         epochs=epochs, save_folder=save_folder, print_freq=print_freq, alpha=alpha, 
                         beta=beta, **kwargs) 

    def _add_kd_components(self, **kwargs):
        criterion_kd = DistillKL(self.T)
        return criterion_kd

    def _compute_kd_loss(self, loss_kd, i, j, **kwargs):
        loss_kd += self.criterion_kd(self.outputs[i], self.outputs[j].detach())
        return loss_kd