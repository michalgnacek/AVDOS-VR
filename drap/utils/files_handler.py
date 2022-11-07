#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Luis Quintero | luisqtr.com
#           Michal Gnacek | gnacek.com
# Created Date: 2021/01/08
# =============================================================================
"""
Utility functions to deal with files management
"""
# =============================================================================
# Imports
# =============================================================================

import os, json, pickle

# =============================================================================
# Main
# =============================================================================

# %%
def check_or_create_folder(filename):
    """
    Creates a folder to the indicated path to be able to write files in it.
    returns True if a folder was created. False if it existed beforehand.
    """
    import os
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        return True
    return False

# %%
def create_json(dictionary, json_path = "filename.json", pretty=False):
    """
    Create a structured dictionary with the filenames of the
    files where the main data is located within the compressed dataset.

    :param json_path: Destination path of JSON file with the dictionary
    :type json_path: str
    :return: Loaded JSON file 
    :rtype: JSON
    """
    check_or_create_folder(json_path)
    with open(json_path, "w") as json_file:
        if (pretty):
            json.dump(dictionary, json_file, indent=4)
        else:
            json.dump(dictionary, json_file)
        print("JSON file was created in", json_path)
        return json_file

# %%
def load_json(json_path = "filename.json"):
    """
    Loads in memory a structured dictionary saved with `create_json`

    :param json_path: Path of JSON file with the dictionary
    :type json_path: str
    :return: Loaded JSON file 
    :rtype: dict
    """
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
        return json_data


# %%
def create_pickle(data_to_save, file_path = "data.pickle"):
    """
    Creates a binary object with the data of movements

    :param data_to_save: Python object to serialize
    :type data_to_save: object
    :param file_path: Path to save serialized object
    :type file_path: str
    :return: None
    :rtype: None
    """
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        
    with open(file_path, "wb") as writeFile:
        pickle.dump(data_to_save, writeFile)

# %%
def load_pickle(file_path = "data.pickle"):
    """
    Loads a pickle file created with `create_pickle()`

    :param file_path: Path of pickle file
    :type file_path: str
    :return: Serialized object
    :rtype: object
    """
    with open(file_path, "rb") as readFile:
        data = pickle.load(readFile)
        return data


def save_numpy_2D(data_to_save, file_path = "filenpy.csv"):
    """
    Creates a TEXT file of a 2D numpy array, only using numpy

    :param data_to_save: Numpy 2D array
    :type data_to_save: numpy ndarray
    :param file_path: Path to save array
    :type file_path: str
    :return: None
    :rtype: None
    """
    import numpy as np
    np.savetxt(file_path, data_to_save, delimiter=",", fmt='%.6f') # fmt='%.6f'
    return


def load_numpy_2D(file_path = "filenpy.csv"):
    """
    Loads a numpy array saved from a TEXT file saved with `save_numpy_2D()`

    :param file_path: Path of file
    :type file_path: str
    :return: Array in 2D
    :rtype: numpy ndarray
    """
    import numpy as np
    return np.loadtxt(file_path, delimiter=",")

def save_binaryfile_npy(data_to_save, file_path = "numpy.npy"):
    """
    Creates a BINARY file of a numpy array, alternative to pickle.

    :param data_to_save: Numpy array
    :type data_to_save: numpy ndarray
    :param file_path: Path to save array
    :type file_path: str
    :return: None
    :rtype: None
    """
    import numpy as np
    np.save(file_path, data_to_save, allow_pickle=False)
    return


def load_binaryfile_npy(file_path = "numpy.npy"):
    """
    Loads a numpy array saved with `save_binaryfile_npy()`

    :param file_path: Path of file
    :type file_path: str
    :return: Array
    :rtype: numpy ndarray
    """
    import numpy as np
    return np.load(file_path, allow_pickle=False)




# %%

"""
Generate paths for intermediate files and images
"""

def generate_complete_path(filename:str, main_folder="./temp/", subfolders='', file_extension = ".png", save_files=True):
    """
    Function to create the full path of a plot based on `name`. It creates all the subfolders required to save the final file.
    If `save_files=False` returns `None`, useful to control from a global variable whether files should be updated or not.

    :param filename: Name of the file (without extension)
    :type filename: str
    :param main_folder: Root folder for the files
    :type main_folder: str
    :param subfolders: Subfolders attached after the root.
    :type subfolders: str
    :param file_extension: Extension of the image
    :type file_extension: str
    :param subfolders: Create path or return None. 
    :type save_files: boolean
    :return: Complete path to create a file
    :rtype: str

    Example: generate_complete_path("histogram", main_folder="./plots/", subfolders='dataset1/') will return
                "./plots/dataset1/histogram.png"
    """
    if (save_files):
        path = main_folder + subfolders + filename + file_extension
        check_or_create_folder(path)
        return path
    else:
        return None
