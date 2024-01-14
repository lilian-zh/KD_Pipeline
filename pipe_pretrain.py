import yaml
import time

from node_data import create_data_loaders, calculate_class_weights
from node_model import create_models
from trainers.pre_trainer import Pre_Trainer


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
    # num_models = s_config['num_models']
    model_attributes = s_config['model_attributes']

    print('start model creating')
    model = create_models(model_name, 1, num_classes, model_attributes)[0]
    print('finish model creating')


    with open('./configs/pretrain.yaml', encoding='utf-8') as file:
        pretrain_config = yaml.safe_load(file)

    epochs = pretrain_config['epochs']
    save_folder = pretrain_config['save_folder']
    momentum = pretrain_config['momentum']
    weight_decay = pretrain_config['weight_decay']
    lr = pretrain_config['lr']
    lr_decay_rate = pretrain_config['lr_decay_rate']
    lr_decay_epochs = pretrain_config['lr_decay_epochs']
    print_freq = pretrain_config['print_freq']

    print('start model training')
    pretrainer = Pre_Trainer(model, train_loader, val_loader, epochs, save_folder, class_weights, 
                    momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs, print_freq, num_train)
    pretrainer.train()

    end_time = time.time()
    hours = (end_time - start_time) / 3600
    print(f"Complete pre-training in {hours:.2f}h")

if __name__ == '__main__':
    main()
