import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from utils import convert_labels, make_dir
import cv2
import time

# color2label_tissue = {(255, 0, 0): 1,  # Healthy_Granulation
#                       (0, 255, 0): 2,  # Infected
#                       (0, 0, 255): 3,  # Hypergranulation
#                       (125, 50, 0): 4,  # Unhealthy_granulation
#                       (255, 0, 255): 5,  # Epitialization
#                       (0, 0, 0): 6,  # Necrotic
#                       (255, 255, 0): 7,  # Slough
#                       (255, 255, 255): 0  # Background (includes Skin)
#                       }
#
# color2label_wound = {(0, 0, 0): 10,  # Wound
#                      (255, 216, 0): 8,  # Skin
#                      (255, 255, 255): 0  # Background
#                      }


def create_processed_dir():
    '''
    Creates the follow directories with saved images

    data
        - processed
            - labels
                - surfaces
                - regions
            - images
    all saved in .tif format
    '''

    # Getting paths
    # Raw data paths
    raw_data_path = os.path.join(os.getcwd(), 'data', 'raw')
    raw_img_path = os.path.join(raw_data_path, 'images')
    raw_label_path = os.path.join(raw_data_path, 'labels')

    # Prcoessed data paths
    processed_data_path = os.path.join(os.getcwd(), 'data', 'processed')
    processed_labels_path = os.path.join(processed_data_path, 'labels')
    processed_surfaces_path = os.path.join(processed_labels_path, 'surfaces')
    processed_regions_path = os.path.join(processed_labels_path, 'regions')
    processed_images_path = os.path.join(processed_data_path, 'images')

    # Making directory
    make_dir(processed_data_path)
    make_dir(processed_labels_path)
    make_dir(processed_surfaces_path)
    make_dir(processed_regions_path)
    make_dir(processed_images_path)

    # Reading in and Transforming
    # Getting names
    img_names = os.listdir(raw_img_path)
    label_names = os.listdir(raw_label_path)
    regions_names = [x for x in label_names if 'regions' in x]
    surfaces_names = [x for x in label_names if 'surfaces' in x]

    # Sort
    img_names.sort()
    regions_names.sort()
    surfaces_names.sort()

    # Saving
    common_names = [x.split('.')[0] for x in regions_names]

    for i in range(len(common_names)):
        img_temp = imread(os.path.join(raw_img_path, img_names[i]))
        with open(os.path.join(raw_label_path, regions_names[i])) as f:
            reg_temp = convert_labels(f)
        with open(os.path.join(raw_label_path, surfaces_names[i])) as f:
            surf_temp = convert_labels(f)
        cv2.imwrite(os.path.join(processed_regions_path,
                                 common_names[i]) + '.tif', np.reshape(reg_temp, img_temp.shape[:-1]))
        cv2.imwrite(os.path.join(processed_surfaces_path,
                                 common_names[i]) + '.tif', np.reshape(surf_temp, img_temp.shape[:-1]))
        cv2.imwrite(os.path.join(processed_images_path,
                                 common_names[i]) + '.tif', img_temp[:, :, [2, 1, 0]])  # Changing from RGB2BGR
        print("writing..: {}".format(regions_names[i].split('.')[0]))


if __name__ == '__main__':
    # raw data
    tic = time.time()
    create_processed_dir()
    print("Process Finished. Time taken: {}".format(time.time() - tic))
