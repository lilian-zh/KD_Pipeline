import os
import pickle
from sklearn.model_selection import StratifiedKFold

def create_train_val_sets(data_dir, k_folds, output_dir):
    # Collect image paths and corresponding labels
    image_paths = []
    labels = []

    # Prepare the labels and class names for saving
    labels_class_names = [(label, class_name) for label, class_name in enumerate(os.listdir(data_dir))]

    # for label, class_name in enumerate(os.listdir(data_dir)):
    for label, class_name in labels_class_names:
        print(label, class_name)
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(label)

    # Use Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(skf.split(image_paths, labels)):
        # Create directory for each fold in the output directory
        fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Define train and validation sets
        train_images = [image_paths[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        #print('train_lables:', train_labels)

        val_images = [image_paths[i] for i in val_index]
        val_labels = [labels[i] for i in val_index]
        #print('val_labels:', val_labels)

        # Convert train and validation sets to the format similar to data_set
        train_set = [(img_path, label) for img_path, label in zip(train_images, train_labels)]
        val_set = [(img_path, label) for img_path, label in zip(val_images, val_labels)]

        with open(os.path.join(fold_dir, 'train_set.pkl'), 'wb') as train_file:
            pickle.dump(train_set, train_file)

        with open(os.path.join(fold_dir, 'val_set.pkl'), 'wb') as val_file:
            pickle.dump(val_set, val_file)
    
    # Save labels and class names to a text file in the output directory
    label_class_file = os.path.join(output_dir, 'label_map.txt')
    with open(label_class_file, 'w') as file:
        for label, class_name in labels_class_names:
            file.write(f"{label}\t{class_name}\n")

def create_dataset_pickle(data_set_path, output_file, label_map_file):
    # If the file already exists, display a message and return
    if os.path.exists(output_file):
        print(f"File {output_file} already exists.")
        return
    
    label_map = {}
    
    # Read label mappings from the label_map.txt file
    if os.path.exists(label_map_file):
        with open(label_map_file, 'r') as label_file:
            for line in label_file:
                label, class_name = line.strip().split('\t')
                label_map[class_name] = int(label)

    data_set = []
    class_folders = os.listdir(data_set_path)

    # Create new label mappings or update existing ones
    for class_folder in class_folders:
        class_path = os.path.join(data_set_path, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if class_folder not in label_map:
                # Add a new category and its label mapping
                label_map[class_folder] = len(label_map)

            label = label_map[class_folder]  # Map the class string to a numerical label

            for image in images:
                image_path = os.path.join(class_path, image)
                data_set.append((image_path, label))

    print(data_set[:5])

    # Save data to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(data_set, f)

    print(f"File {output_file} created successfully.")

    print(f"Label map used: {label_map}")



if __name__ == '__main__':
    # Example usage:
    data_dir = r'E:\Thesis\datasets\internal_data\train_val'
    k_folds = 5
    output_dir = r'C:\Users\lili_\Documents\Thesis\hpc_kd4\data\02_intermediate\internal_data'
    create_train_val_sets(data_dir, k_folds, output_dir)

    data_set_path = r'E:\Thesis\datasets\internal_data\test'
    output_file = os.path.join(output_dir, 'test_set.pkl')
    label_map_file = os.path.join(output_dir, 'label_map.txt')
    create_dataset_pickle(data_set_path, output_file, label_map_file)
