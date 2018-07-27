r"""
There are 2 training occasions where this file is being called:
    For Network 1:
        a) original.png
        b) mask3.png
    For Network 2:
        a) original_cropped.png
        b) mask9_cropped.png

Augmentations (each with probability 0.5):
    1) rotate
    2) zoom
    3) flipping (vertically and horizontally)
    4) shearing

Notes:
- Augmentor requires that paired images have the same name!
- Load every batch of 336 pairs of images into memory; don't save to disk
"""

from PIL import Image
import numpy as np
import Augmentor
import argparse
import matplotlib
matplotlib.use('TkAgg') # so that you can close the plot window
import matplotlib.pyplot as plt
import glob
from scipy.misc import imread

def eagerplot(image):
    """
    Quick plotting for debugging
    """
    plt.imshow(image, cmap="gray")
    plt.show()

def augment(network=1, debugging=0, batch_size=1): 
    """Augments paired images (orig & mask) and saves in memory
    Args:
        network: specify which network (1 or 2)
    Returns:
        a batch of images with size batch_size
    """

    DIR_ORIG = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/original/"
    DIR_MASK3 = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-3/"
    DIR_ORIG_CROPPED = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/original-cropped/"
    DIR_MASK9_CROPPED = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-9-cropped/"
    DIR_OUTPUT = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/augments/" # for debugging

    if network == 1:
        dir_orig = DIR_ORIG
        dir_mask = DIR_MASK3
    elif network == 2:
        dir_orig = DIR_ORIG_CROPPED
        dir_mask = DIR_MASK9_CROPPED

    files_orig = glob.glob(dir_orig + '*.png')
    files_mask = glob.glob(dir_mask + '*.png')

    if debugging:
         # Specify location of originals
        pipe = Augmentor.Pipeline(source_directory=dir_orig, output_directory=DIR_OUTPUT)
         # Specify location of masks
        pipe.ground_truth(dir_mask)
    else:    
        from keras import backend as K
        K.set_image_data_format('channels_last')

        pipe = Augmentor.Pipeline()
        # pipe = Augmentor.Pipeline(source_directory=dir_orig, output_directory=DIR_OUTPUT)
        # pipe.ground_truth(dir_mask)
        # orig = [imread(fname).astype('uint8') for fname in files_orig]
        orig = np.array([np.array(Image.open(fname)) for fname in files_orig])
        mask = [imread(fname).astype('uint8') for fname in files_mask]

    # 1) Rotate and automatically zoom (to avoid artifacts)
    pipe.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)

    # 2) Zoom
    pipe.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

    # 3) Flipping vertically and horizontally
    pipe.flip_left_right(probability=0.5)
    pipe.flip_top_bottom(probability=0.5)

    # 4) Shearing
    pipe.shear(probability=0.5, max_shear_left=25, max_shear_right=25)

    if debugging:
        # Augment each pair once
        pipe.process()
    else:
        # Save in memory
        # image_batch = pipe.keras_generator(batch_size=batch_size)
        image_batch = pipe.keras_generator_from_array(images=orig, labels=mask, batch_size=batch_size)
    
        while True:
            x, y = next(image_batch)
            _, ax = plt.subplots(1,2)
            ax[0].imshow(x[0])
            ax[1].imshow(y[0])
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--network',
        type=int,
        default=1,
        help="""\
        Network 1 or 2."""
    )
    parser.add_argument(
        '--debugging_mode',
        type=int,
        default=0,
        help="""\
        0 or 1. Debugging mode saves augments to a file."""
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="""\
        Size of batch to use at every epoch."""
    )
    FLAGS, _ = parser.parse_known_args()

    augment(
        network=FLAGS.network, 
        debugging=FLAGS.debugging_mode,
        batch_size=FLAGS.batch_size
    )
