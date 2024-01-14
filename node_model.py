"""
This is a boilerplate pipeline 'model_preparation'
generated using Kedro 0.18.11
"""
import os
import torch
from models import model_dict


def create_models(model_name, num_models, num_classes, model_attributes):
    if not isinstance(model_name, str):
        raise TypeError("model_name should be a string.")
    if not isinstance(model_attributes, dict):
        raise TypeError("model_attributes should be a dictionary.")
    if not isinstance(num_models, int) or num_models <= 0:
        raise TypeError("num_models should be a positive integer.")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise TypeError("num_classes should be a positive integer.")

    models = []

    if model_name in model_dict:
        model_constructor = model_dict[model_name]
        for i in range(num_models):
            try:
                model = model_constructor(num_classes=num_classes, **model_attributes)
                models.append(model)
            except Exception as e:
                print(f"Error occurred while creating model {model_name}: {e}")
    else:
        raise NotImplementedError(f"Model {model_name} is not defined in model_dict.")

    return models



def get_latest_model_folder(model_dir):
    """
    Find the latest model folder in the specified directory.

    Args:
        model_dir (str): The directory where the models are stored.

    Returns:
        Optional[str]: The path of the latest model folder if found, otherwise None.
    """

    subdirectories = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    # Sort subdirectories by modification time in descending order
    subdirectories.sort(key=lambda d: os.path.getmtime(os.path.join(model_dir, d)), reverse=True)

    for subdir in subdirectories:
        subdir_path = os.path.join(model_dir, subdir)
        if any(os.path.isfile(os.path.join(subdir_path, f)) for f in os.listdir(subdir_path)):
            return subdir

    return None


def get_teacher_paths(model_dir, keyword="best"):
    """
    Get the paths of models with the specified keyword.

    Args:
        model_dir (str): The directory where the models are stored.
        keyword (str): The keyword to look for in the model name.

    Returns:
        List[str]: List of paths for models with the specified keyword.

    Raises:
        FileNotFoundError: If no model with the specified keyword is found in the directory.
    """

    latest_folder = get_latest_model_folder(model_dir)
    if latest_folder:
        files_in_folder = os.listdir(os.path.join(model_dir, latest_folder))
        files_in_folder.sort(key=lambda fn: os.path.getmtime(os.path.join(model_dir, latest_folder, fn)), reverse=True)

        matching_models = []
        for model in files_in_folder:
            if keyword in model:
                matching_models.append(os.path.join(model_dir, latest_folder, model))

        if matching_models:
            return matching_models

    raise FileNotFoundError(f"No {keyword} model found in {model_dir}")


def load_teachers(model_dir, model_name, num_classes, model_attributes):
    """
    Load the pre-trained model.

    Args:
        model_params (Dict[str, Any]): A dictionary containing model configuration.
        n_cls (int): The number of classes.
        model_dir (str): The directory where the model is stored.

    Returns:
        Module: The loaded teacher model.
    """

    print('==> loading teacher model')

    try:
        teacher_paths = get_teacher_paths(model_dir)
        print("model_paths:", teacher_paths)
        
        # if len(teacher_paths) != num_models:
        #     raise ValueError(f"Number of models found ({len(teacher_paths)}) \
        #                      does not match expected number ({num_models})")

        loaded_models = []
        for teacher_path in teacher_paths:
            model = create_models(model_name, 1, num_classes, model_attributes)[0]
            state_dict = torch.load(teacher_path)
            model.load_state_dict(state_dict['model'])
            loaded_models.append(model)

        print('==> done')
        return loaded_models

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        # Handle the file not found error as needed
    except ValueError as e:
        print(f"Value error: {e}")
        # Handle the value error when the number of models found does not match
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle other exceptions as needed




if __name__ == '__main__':
    model_dir = r'C:\Users\lili_\Documents\Thesis\hpc_kd4\data\06_models' 
    model_name = 'resnet18'  
    num_models = 2  
    num_classes = 13  
    model_attributes = {}  

    teacher = load_teachers(model_dir, model_name, num_classes, model_attributes)

    print(len(teacher))
