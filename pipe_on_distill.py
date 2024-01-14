import yaml
import time

from node_data import create_data_loaders, calculate_class_weights
from trainers.trainer_cfg import KD_options



def main():
    start_time = time.time()
    with open('./configs/data.yaml', encoding='utf-8') as file:
        data_config = yaml.safe_load(file)
    
    num_workers = data_config['num_workers']
    batch_size = data_config['batch_size']   
    dataset = data_config['dataset']
    num_classes = data_config['num_classes']   
    train_path = data_config['train_path']   
    val_path = data_config['val_path']  
    is_class_weights = data_config['is_class_weights'] 

    print('start data loading')
    train_loader, val_loader, num_train, img_shape = create_data_loaders(
        dataset=dataset, train_path=train_path, val_path=val_path,  
        num_workers=num_workers, batch_size=batch_size, is_train=True
    )
    print('finish data loading')

    if is_class_weights:
        class_counts = data_config['class_counts']
        class_weights = calculate_class_weights(class_counts) 
    else:
        class_weights = None


    with open('./configs/model.yaml', encoding='utf-8') as file:
        model_config = yaml.safe_load(file)

    s_config = model_config['student_config']
    model_name = s_config['model_name']
    num_models = s_config['num_models']
    model_attributes = s_config['model_attributes']


    with open('./configs/distill.yaml', encoding='utf-8') as file:
        distill_config = yaml.safe_load(file) 
    epochs = distill_config['epochs']
    save_folder = distill_config['save_folder']
    momentum = distill_config['momentum']
    weight_decay = distill_config['weight_decay']
    lr = distill_config['lr']
    lr_decay_rate = distill_config['lr_decay_rate']
    lr_decay_epochs = distill_config['lr_decay_epochs']
    print_freq = distill_config['print_freq']
    alpha = distill_config['alpha']
    beta = distill_config['beta']
    kd_approach = distill_config['kd_approach']
    kd_option = KD_options[kd_approach]['Trainer']
    kd_params = KD_options[kd_approach]['Extra_Params']

    
    print('start distilling')
    distiller = kd_option(train_loader, val_loader, num_train, class_weights, momentum, weight_decay, 
                 lr, lr_decay_rate, lr_decay_epochs, model_name, num_classes, model_attributes, 
                 num_models, epochs, save_folder, print_freq, alpha, beta, **kd_params)

    distiller.train()
    end_time = time.time()
    hours = (end_time - start_time) / 3600
    print(f"Complete distillation in {hours:.2f}h.")


if __name__ == '__main__':
    main()
