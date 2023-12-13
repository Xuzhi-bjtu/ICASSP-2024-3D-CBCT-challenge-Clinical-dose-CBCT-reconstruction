# %% Challenge imports
import torch
import numpy as np
# %% my imports
import os
from DL_based_CBCT_reconstruction import DL_based_CBCT_reconstruction
from Data_path_loader import Data_path_loader

# Please specify the GPU number to be used (only one GPU is required)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initialize deep learning algorithm
Algorithm = DL_based_CBCT_reconstruction()

# %% Load data
folder = "test/folder/in/organizers/path"  # Please replace with the absolute path to the test dataset folder

# %% my loading loop
data_path = Data_path_loader(folder, data_type='clinical_dose')

for i, (sinogram_path, target_reconstruction_path) in enumerate(data_path):
    print(i)
    # load clinical dose sinogram & Algorithm reconstruction
    sinogram = np.load(sinogram_path, allow_pickle=True)
    reconstruction = Algorithm.process(sinogram)
    # load GT clean FDK image
    target_reconstruction = np.load(target_reconstruction_path, allow_pickle=True)
