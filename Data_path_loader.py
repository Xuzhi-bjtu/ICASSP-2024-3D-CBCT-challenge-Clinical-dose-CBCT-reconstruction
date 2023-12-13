import os
import glob


def Data_path_loader(folder_path, data_type='clinical_dose'):
    """
    :param folder_path: test set folder path
    :return: projection data path & clean FDK image data path
    """

    data_path = []

    proj_file_path_list = glob.glob(os.path.join(folder_path, '*_sino_%s.npy') % data_type)
    for proj_file_path in proj_file_path_list:
        clean_FDK_img_path = proj_file_path.replace("sino_%s" % data_type, "clean_fdk_256")
        data_path.append([proj_file_path, clean_FDK_img_path])

    return data_path


if __name__ == "__main__":
    data_path = Data_path_loader('../../ICASSP-2024_3D-CBCT_challenge_low_dose/data/validate/')
