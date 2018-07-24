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

import numpy as np
import Augmentor
import argparse
import matplotlib
matplotlib.use('TkAgg') # so that you can close the plot window
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import glob


DIR_ORIG = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/original/"
DIR_MASK3 = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-3/"
DIR_ORIG_CROPPED = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/original-cropped/"
DIR_MASK9_CROPPED = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-9-cropped/"
DIR_OUTPUT = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/augments/" # for debugging
BATCH_SIZE = 336

FILE_ORIG = glob.glob(DIR_ORIG + '*.png')
FILE_MASK3 = glob.glob(DIR_MASK3 + '*.png')


def eagerplot(image):
    """
    Quick plotting for debugging
    """
    plt.imshow(image, cmap="gray")
    plt.show()

def augment(network=1, debugging=0, batch_size=BATCH_SIZE): 
    """Augments paired images (orig & mask) and saves in memory
    Args:
        network: specify which network (1 or 2)
    Returns:

    """

    if network == 1:
        orig = DIR_ORIG
        mask = DIR_MASK3
    elif network == 2:
        orig = DIR_ORIG_CROPPED
        mask = DIR_MASK9_CROPPED

    # Specify location of originals
    if debugging:
        pipe = Augmentor.Pipeline(source_directory=orig, output_directory=DIR_OUTPUT)
    else:    
        # TODO DONT GENERATE
        pipe = Augmentor.Pipeline()
        x = np.array([np.array(Image.open(fname)) for fname in FILE_ORIG])
        y = np.array([np.array(Image.open(fname)) for fname in FILE_MASK3])

    # Specify location of masks
    pipe.ground_truth(mask)

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
        # Should return a batch of 336 images
        # pipe.process()
        image_batch = pipe.keras_generator_from_array(images=x,labels=y,batch_size=batch_size)
    
        while True:
            x, y = next(image_batch)
            print(len(x))


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
        default=BATCH_SIZE,
        help="""\
        Size of batch to use at every epoch."""
    )
    FLAGS, _ = parser.parse_known_args()

    augment(
        network=FLAGS.network, 
        debugging=FLAGS.debugging_mode,
        batch_size=FLAGS.batch_size
    )

