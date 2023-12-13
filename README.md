# DL-based Clinical Dose CBCT Reconstruction

The codes for the clinical dose CBCT reconstruction task in the ICASSP-2024 3D-CBCT Challenge. 
Our team's name is BJTU_PKUCH.

## 1. Download pre-trained image domain (ID) model
* [Get pre-trained ID model in this link] (https://drive.google.com/file/d/1g9k6r1ZVBqAxszi8sGHQqszK40Bj5Vdg/view?usp=drive_link): Place the file "ID_model_ckpt.pth" into folder "pretrained_ckpt/"

## 2. OS & CUDA Environment

- Please ensure to use a Linux system for testing, with Ubuntu 18.04 being the recommended version. Ensure that CUDA 11.8 is pre-installed in your system.

## 3. Python & Conda Environment

- To deploy the required Python and Conda environment for testing, follow these steps:

### I Create a Python 3.8 Environment
Create a Conda environment named CBCT using Python 3.8.
```bash
conda create -n CBCT python=3.8
```

### II Activate the CBCT Environment
Activate the newly created CBCT environment.
```bash
conda activate CBCT
```

### III Install PyTorch Packages
Install the PyTorch related packages using the PyTorch and Nvidia channels.
```bash
conda install -c pytorch -c nvidia pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8
```

### IV Install ASTRA Toolbox
Install the ASTRA toolbox from the specified channel.
```bash
conda install -c astra-toolbox/label/dev astra-toolbox=2.1.3
```

### V Install tqdm Package
Install the tqdm package.
```bash
conda install tqdm=4.65.0
```

### VI Install tomosipo
Clone and install `tomosipo` from GitHub. After cloning, navigate to the `tomosipo` directory path to proceed with the installation.
```bash
git clone https://github.com/ahendriksen/tomosipo.git
cd [path_to_tomosipo]
pip install .
```

### VII Install ts_algorithms
Clone and install `ts_algorithms` from GitHub. After cloning, navigate to the `ts_algorithms` directory path to proceed with the installation.
```bash
git clone https://github.com/ahendriksen/ts_algorithms.git
cd [path_to_ts_algorithms]
pip install .
```

## 4. Test

- To perform the DL-based clinical dose CBCT reconstruction for the test dataset, please use the `test.py` script.  
- In the script, specify the GPU number to be used (only one GPU is required) in line 10: `os.environ['CUDA_VISIBLE_DEVICES']`.
- In the script, replace line 16, `folder = "test/folder/in/organizers/path"`, with the absolute path to the test dataset folder.
- In the script, the variable `reconstruction` represents the reconstruction results obtained from the clinical dose sinogram using the DL-based algorithm.
- In the script, the variable `target_reconstruction` is used as the ground truth.