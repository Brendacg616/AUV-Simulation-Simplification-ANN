import scipy.io
import os
from pathlib import Path
from typing import List, Union
import torch
import re
import numpy as np

def read_mat_workspace(file_path):
    """
    Reads a MATLAB .mat workspace file and returns its contents as a dictionary.

    Parameters:
        file_path (str): Path to the .mat file.

    Returns:
        dict: Dictionary containing variables from the .mat file.
    """
    return scipy.io.loadmat(file_path, )




def find_mat_files(paths: Union[str, List[str]], recursive: bool = False) -> List[str]:
    """
    Find all .mat files in the given paths.

    Parameters
    ----------
    paths : str or list of str
        Directory path(s) to search.
    recursive : bool, optional (default=False)
        If True, search subdirectories as well.

    Returns
    -------
    List[str]
        List of full paths to .mat files found.
    """
    if isinstance(paths, str):
        paths = [paths]

    mat_files = []

    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.is_dir():
            print(f"Warning: {path} is not a directory.")
            continue

        if recursive:
            mat_files.extend([str(f) for f in path.rglob("*.mat")])
        else:
            mat_files.extend([str(f) for f in path.glob("*.mat")])

    return mat_files


# Compare and save the best model
def save_best_model(model, new_accuracy, threshold, directory=".", model_name="best_model"):
    """
    Saves the model if its accuracy is higher than the previous best one.
    
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        new_accuracy (float): The new model's accuracy to compare.
        threshold (float): Minimum accuracy improvement to consider saving.
        directory (str): Directory where the model files are saved.
        model_name (str): Base name of the model file.
        
    Returns:
        bool: True if the new model was saved, False otherwise.
    """
    best_accuracy = 0.0
    best_model_file = None
    if new_accuracy < threshold:
        return False
    # Look for the best model file in the directory
    for filename in os.listdir(directory):
        match = re.match(rf"{model_name}_(\d+.\d+).pt", filename)
        if match:
            saved_accuracy = float(match.group(1))
            if saved_accuracy > best_accuracy:
                best_accuracy = saved_accuracy
                best_model_file = filename

    # Compare accuracies
    if new_accuracy > best_accuracy:
        # Save new model
        new_model_file = f"{model_name}_{new_accuracy:.10f}.pt"
        torch.save(model.state_dict(), os.path.join(directory, new_model_file))
        print(f"New best model saved: {new_model_file} (Accuracy: {new_accuracy:.10f})")
        return True
    else:
        #print(f"No improvement. Current best model: {best_model_file} (Accuracy: {best_accuracy:.5f}%)")
        return False
    
#save_best_model(model, accuracy)


# Load the best model
def best_model_path(directory=".", model_name="best_model"):
    """
    Saves the model if its accuracy is higher than the previous best one.
    
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        new_accuracy (float): The new model's accuracy to compare.
        directory (str): Directory where the model files are saved.
        model_name (str): Base name of the model file.
        
    Returns:
        bool: True if the new model was saved, False otherwise.
    """
    best_accuracy = 0.0
    best_model_file = None

    # Look for the best model file in the directory
    for filename in os.listdir(directory):
        match = re.match(rf"{model_name}_(\d+.\d+).pt", filename)
        if match:
            saved_accuracy = float(match.group(1))
            if saved_accuracy > best_accuracy:
                best_accuracy = saved_accuracy
                best_model_file = filename

    # Compare accuracies
    if best_accuracy != None:
        print(f"Best model loaded: {best_model_file} (Accuracy: {best_accuracy:.10f}%)")
        return best_model_file
    else:
        print(f"No improvement. Current best model: {best_model_file} (Accuracy: {best_accuracy:.10f})")
        return None
    

# Create dataset from .mat files
def create_dataset_from_mat(file_path, list_of_vars):
    ws_files = find_mat_files(file_path, recursive=True)
    input = np.zeros((1, 26))
    output = np.zeros((1, 12))
    
    for f in ws_files:
        print(f)
        mat_data = read_mat_workspace(f)
        N = mat_data.get('N')
        if N is None:
            continue
        data = np.zeros((int(N), 1))
        skip = False
        for var in list_of_vars:
            temp_data = mat_data.get(var)
            if temp_data is not None:
                if np.isnan(temp_data).any():
                    skip = True
                    print(f"Skipping {f} due to NaN values in {var}")
                    break
                print(f"{var} found with shape {temp_data.shape}")
                data = np.hstack((data, temp_data))
        if skip: 
            continue
        input = np.vstack((input, data[:-1, 1:27]))
        output = np.vstack((output, data[1:, 1:13]))

    print(input.shape, output.shape)
    return input[1:], output[1:]


# Clean dataset
def clean_dataset(input, output):
    # Sample each 5th row
    input = input[::5, :]
    output = output[::5, :]
    print(input.shape, output.shape)
    return input, output


