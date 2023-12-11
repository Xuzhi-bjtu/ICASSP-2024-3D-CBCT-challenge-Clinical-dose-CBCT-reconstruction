import torch
from torch import nn
from torchvision.transforms import transforms
import numpy as np
import os
import tomosipo as ts
from ts_algorithms import fdk
from networks.ID_model import Unet3D

# define device
device = torch.device('cuda')

# define image_transform
transforms = transforms.Compose([
    transforms.ToTensor(),
])

# define geometry of CBCT scan & Create a tomosipo operator
image_size = [300, 300, 300]
image_shape = [256, 256, 256]
voxel_size = [1.171875, 1.171875, 1.171875]
detector_shape = [256, 256]
detector_size = [600, 600]
pixel_size = [2.34375, 2.34375]
dso = 575
dsd = 1050
angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

vg = ts.volume(shape=image_shape, size=image_size)
pg = ts.cone(angles=angles, shape=detector_shape, size=detector_size, src_orig_dist=dso, src_det_dist=dsd)
A = ts.operator(vg, pg)


class DL_based_CBCT_reconstruction:
    def __init__(self):
        # Image domain model
        try:
            # Load the pre-trained ID model from a specified path
            ID_model = Unet3D(1, 1).to(device)
            ID_model = nn.parallel.DataParallel(ID_model)
            ID_model.load_state_dict(torch.load('./pretrained_ckpt/ID_model_ckpt.pth'))
            self.ID_model = ID_model
        except IOError:
            print("ID Model file could not be loaded. Please check the path and file integrity.")

    def FDK_reconstruction(self, proj):
        """
        :param proj: ndarray (256, 360, 256)
        :return: reconstructed 3D image: ndarray (256, 256, 256)
        """
        proj = torch.from_numpy(proj).to(device)
        # Reconstruct image
        recon = fdk(A, proj)
        recon = recon.detach().cpu().numpy()  # ndarray (256, 256, 256)

        return recon

    def ID_model_processing(self, recon_image):
        """
        :param recon_image: ndarray (256, 256, 256)
        :return: refined image: ndarray (256, 256, 256)
        """
        recon_image = recon_image.swapaxes(0, 1).swapaxes(1, 2)  # ndarray (256, 256, 256) axis changed
        refined_image = np.empty((256, 256, 256), dtype='float32')  # ndarray (256, 256, 256)
        # ndarray splitting, tensor conversion, ID_model processing, ndarray conversion, & ndarray rearrangement
        self.ID_model.eval()
        with torch.no_grad():
            patch_num = 0
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        recon_patch = recon_image[128 * h:128 * (h + 1), 128 * w:128 * (w + 1),
                                      128 * d:128 * (d + 1)]  # ndarray (128,128,128)
                        recon_patch_tensor = transforms(recon_patch)  # torch (128,128,128) axis changed
                        recon_patch_tensor = recon_patch_tensor.reshape(1, 1, 128, 128,
                                                                        128)  # torch (1,1,128,128,128) add bs & channel
                        recon_patch_tensor = recon_patch_tensor.to(device)  # loading to GPU
                        output_patch_tensor = self.ID_model(recon_patch_tensor)  # torch (1,1,128,128,128)
                        refined_patch_tensor = recon_patch_tensor + output_patch_tensor  # torch (1,1,128,128,128)
                        refined_patch = torch.squeeze(
                            refined_patch_tensor).cpu().numpy()  # ndarray (128,128,128) loading to CPU

                        # patches rearrangement
                        d_index = patch_num // 4
                        h_w_index = patch_num % 4
                        h_index = h_w_index // 2
                        w_index = h_w_index % 2
                        refined_image[128 * d_index:128 * (d_index + 1), 128 * h_index:128 * (h_index + 1),
                        128 * w_index:128 * (w_index + 1)] = refined_patch
                        patch_num += 1

        return refined_image

    def process(self, clinical_dose_proj):
        """
        Process the clinical-dose projection data through the pipeline.

        :param clinical_dose_proj: ndarray (256, 360, 256)
        :return: reconstructed 3D image: ndarray (256, 256, 256)
        """
        # Step 1: Perform FDK reconstruction on the clinical-dose projection
        reconstructed_image = self.FDK_reconstruction(clinical_dose_proj)

        # Step 2: Refine the reconstructed image using ID model
        refined_image = self.ID_model_processing(reconstructed_image)

        return refined_image


if __name__ == '__main__':
    import glob

    Algorithm = DL_based_CBCT_reconstruction()
    proj_file_path_list = glob.glob('../data/validate/*_sino_clinical_dose.npy')
    for i, proj_file_path in enumerate(proj_file_path_list):
        print(i, proj_file_path)
        clinical_dose_proj = np.load(proj_file_path, allow_pickle=True)
        reconstructed_image = Algorithm.process(clinical_dose_proj)
        np.save('./reconstruction_output/%s_clinical_dose_recon.npy' % os.path.basename(proj_file_path)[:4],
                reconstructed_image)
