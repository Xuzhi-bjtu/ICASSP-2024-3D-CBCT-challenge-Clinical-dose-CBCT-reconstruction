import os
import glob
import numpy as np
import argparse
from DL_based_CBCT_reconstruction import DL_based_CBCT_reconstruction


def test_algorithm(args):
    # Specify GPU number; The program requires only one GPU to run
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Initialize deep learning algorithm
    Algorithm = DL_based_CBCT_reconstruction()

    # Algorithm inference & result saving
    os.makedirs('./reconstruction_output', exist_ok=True)
    proj_file_path_list = glob.glob(os.path.join(args.test_path, '*_sino_clinical_dose.npy'))
    for i, proj_file_path in enumerate(proj_file_path_list):
        print(i, proj_file_path)
        clinical_dose_proj = np.load(proj_file_path, allow_pickle=True)
        reconstructed_image = Algorithm.process(clinical_dose_proj)
        np.save('./reconstruction_output/%s_clinical_dose_recon.npy' % os.path.basename(proj_file_path)[:4],
                reconstructed_image)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_path', type=str)
    parse.add_argument('--gpu', type=str, default='0')
    args = parse.parse_args()

    test_algorithm(args)
