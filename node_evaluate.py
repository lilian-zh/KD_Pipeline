import yaml

import torch

from trainers.util import test
from node_data import create_data_loaders
from node_model import create_models


def main():
    with open('./configs/data.yaml', encoding='utf-8') as file:
        data_config = yaml.safe_load(file)
    
    num_workers = data_config['num_workers']
    batch_size = data_config['batch_size']   
    dataset = data_config['dataset']
    num_classes = data_config['num_classes']   
    test_path = data_config['test_path']   

    print('start data loading')
    test_loader = create_data_loaders(dataset=dataset, test_path=test_path, num_workers = num_workers, 
                                      batch_size = batch_size, is_train=False
    )
    print('finish data loading')


    with open('./configs/evaluate.yaml', encoding='utf-8') as file:
        test_config = yaml.safe_load(file)

    model_path = test_config['model_path']
    model_name = test_config['model_name']
    model_attributes = test_config['model_attributes']
    output_dir = test_config['output_dir']

    model = create_models(model_name, 1, num_classes, model_attributes)[0]
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])
    
    test(test_loader, model, output_dir)


if __name__ == '__main__':
    main()
