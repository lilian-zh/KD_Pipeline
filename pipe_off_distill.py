import yaml
import time

from node_data import create_data_loaders, calculate_class_weights
from node_model import create_models, load_teachers
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
    s_name = s_config['model_name']
    s_num = s_config['num_models']
    s_attributes = s_config['model_attributes']

    t_config = model_config['teacher_config']
    t_dir = t_config['teacher_dir']
    t_num = t_config['num_models']
    t_name = t_config['model_name']
    t_attributes = t_config['model_attributes']


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
    max_step = distill_config['max_step']
    step_patience = distill_config['step_patience']
    alpha = distill_config['alpha']
    beta = distill_config['beta']
    kd_approach = distill_config['kd_approach']
    print(kd_approach)
    kd_option = KD_options[kd_approach]['Trainer']
    kd_params = KD_options[kd_approach]['Extra_Params']

    
    global_best_acc = 0.0
    no_improvement_count = 0
    best_step = 0
    # print('start distilling')
    for step in range(1, max_step + 1):
        model_s = create_models(s_name, s_num, num_classes, s_attributes)[0]
        teachers = load_teachers(t_dir, t_name, num_classes, t_attributes)
        if len(teachers) != t_num:
            print(f"Number of models found ({len(teachers)}) \
                  does not match expected number ({t_num}), choose the last ({t_num}) models.")
        model_t = teachers[-t_num]

        distiller = kd_option(model_t, model_s, train_loader, val_loader, num_train, img_shape, class_weights,
                              momentum, weight_decay, lr, lr_decay_rate, lr_decay_epochs,
                              step, epochs, save_folder, print_freq, alpha, beta, **kd_params)

        best_acc = distiller.train()

        if best_acc > global_best_acc:
            global_best_acc = best_acc
            no_improvement_count = 0
            best_step=step
        else:
            no_improvement_count += 1

        if no_improvement_count >= step_patience:
            break
    end_time = time.time()
    hours = (end_time - start_time) / 3600
    print(f"Complete distillation in {hours:.2f}h, with the best accuracy {global_best_acc} in trial {best_step}")

if __name__ == '__main__':
    main()
