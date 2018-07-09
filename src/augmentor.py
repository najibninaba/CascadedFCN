import Augmentor
import os
DIR_RAW = os.path.join(os.getcwd(), 'data', 'Original_Images_v3')
DIR_MASKS = os.path.join(os.getcwd(), 'data', 'Mask_Labeled_Wound_3Class')
DIR_OUTPUT = os.path.join(os.getcwd(), 'data', 'Augments')


# DIR_RAW = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/original"
# DIR_MASKS = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/masks-class-3/"
# DIR_OUTPUT = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/augments"

# Location of original images
p = Augmentor.Pipeline(DIR_RAW, output_directory=DIR_OUTPUT)

# Location of masks
# Augmentor requires that corresponding images have the same names
p.ground_truth(DIR_MASKS)
# Rotate and automatically zoom
p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)

# Zoom
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

# Flipping vertically and horizontally
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)

# Shearing
p.shear(probability=0.5, max_shear_left=25, max_shear_right=25)

# Outputs to a folder together with
p.sample(1000)
