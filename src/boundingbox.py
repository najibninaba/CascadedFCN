
"""Using OpenCV, get bounding box for region of interest. 
Two occasions where this code will be used:
    a) training: 
        Need to create mask3_grey_bin
    b) inference:

1) Load masktruth (3-class mask)
2) Transform masktruth to 2-class for contour detection
3) Perform dilation on masktruth
4) Use masktruth to get contours. For every contour, do the following for orig, masknet1 and masktruth
    a) Get bounding (upright) rectangle
    b) Remove small contours
    c) Crop
    d) Save

Pipeline:
orig      -> orig_cropped   -> orig_contour
mask3out1  -> mask3out1_gray -> mask3out1_grayinv  -> mask3out1_contour
mask9truth -> mask9ruth_gray -> mask9truth_grayinv -> mask9truth_contour

origcontour_1.png
"""
import os
import sys
import cv2
import numpy as np
from scipy.misc import imread, imsave
import matplotlib
matplotlib.use('TkAgg') # so that you can close the plot window
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.monitor_interval = 0 # to silent warnings

# Specify constants and directories
# pylint: disable=line-too-long
DIR_ORIG = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/original/"
DIR_MASK3 = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-3/"
DIR_MASK9 = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-9/"
DIR_ORIG_CROPPED = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/original-cropped/"
DIR_MASK9_CROPPED = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/masks-class-9-cropped/"
DIR_INF = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/inference/"

DIR_OUTPUT = "/Users/raimibinkarim/Desktop/Cascaded-FCN-Data/augments/" # for debugging

MIN_AREA = 50
BOX_COLOR = (70, 173, 212)
PADDING = 4 # pixels
ZOOM = 1.2
KERNEL_DILATION = np.ones((11, 11))

def eagerplot(image):
    """
    Quick plotting for debugging
    """
    plt.imshow(image, cmap="gray")
    plt.show()

def get_bounding_box_for_one_pair(training=True, idx=-1):
    """
    Get bounding box for one wound (original and mask), identified with idx
    And save it to DIR_L2_ORIG & DIR_L2_MASK directories
    """
    # If no index specified, assume this function is used for demo purposes.
    if idx == -1:
        idx = np.random.randint(1, 336+1) # 336 wounds

    if training:
        filename_orig  = DIR_ORIG  + "Wound_" + str(idx) + ".png"
        filename_mask  = DIR_MASK9 + "Wound_" + str(idx) + ".png"
        filename_guide = DIR_MASK3 + "Wound_" + str(idx) + ".png"

    # Load original, mask and guide
    orig  = imread(filename_orig,  mode="RGB")
    mask  = imread(filename_mask,  mode="RGB")
    guide = imread(filename_guide, mode="RGB")

    # Dilate mask
    guide_dilated = cv2.dilate(guide, kernel=KERNEL_DILATION)

    # Convert mask to grayscale
    guide_gray = cv2.cvtColor(guide_dilated, cv2.COLOR_RGB2GRAY)

    # Convert mask to binary mask
    _, guide_bin = cv2.threshold(src=guide_gray, thresh=0, maxval=1, type=cv2.THRESH_BINARY)
    
    # Invert
    guide_bin = 255 - guide_bin

    # Get contours based on binary image
    _, contours, _ = cv2.findContours(guide_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # print("There are " + str(len(contours)) + " contour(s).")

    num_null_contours = 0

    for contour in contours:

        # Ignore contours less than MIN_AREA
        if cv2.contourArea(contour) < MIN_AREA:
            num_null_contours += 1 
            continue 

        # Get dimensions for bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draws an up-right rectangle on original image. For debugging
        cv2.rectangle(orig, pt1=(x-PADDING,y-PADDING), pt2=(x+w+PADDING+1,y+h+PADDING+1), color=BOX_COLOR, thickness=3)
        cv2.rectangle(mask, pt1=(x-PADDING,y-PADDING), pt2=(x+w+PADDING+1,y+h+PADDING+1), color=BOX_COLOR, thickness=3)     

        # Crop image
        orig_cropped = orig[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]

        # Get bounding rotated rectangle
        # rect = cv2.minAreaRect(contour) # rect = (x,y,w,h,theta) where (x,y) is the centre point and theta is the angle of rotation
        # box  = cv2.boxPoints(rect) 
        # box  = np.int0(box)
        # area = cv2.contourArea(contour)
        # print("This contour has area " + str(area))
        # eagerplot(orig)

        # Draws a rotated rectangle on original image
        # cv2.drawContours(orig, contours=[box], contourIdx=0, color=BOX_COLOR, thickness=3)
        # cv2.drawContours(mask_gray, contours=[box], contourIdx=0, color=BOX_COLOR, thickness=3)

        # Get coordinates, width, height, angle for cropping and rotating
        # W = rect[1][0]
        # H = rect[1][1]
        # Xs = [i[0] for i in box]
        # Ys = [i[1] for i in box]
        # x1 = min(Xs)
        # x2 = max(Xs)
        # y1 = min(Ys)
        # y2 = max(Ys)
        # angle = rect[2]
        # if angle < -45:
        #     angle += 90
        #     rotated = True
        # else:
        #     rotated = False

        # Rotate
        # ctr  = (int((x1+x2)/2), int((y1+y2)/2))
        # size = (int(ZOOM*(x2-x1)), int(ZOOM*(y2-y1)))
        # cv2.circle(orig, ctr, 10, BOX_COLOR, -1) # again this was mostly for debugging purposes
        # M = cv2.getRotationMatrix2D(center=(size[0]/2, size[1]/2), angle=angle, scale=1.0)

        # Retrieve pixel rectangle
        # orig_cropped = cv2.getRectSubPix(orig, patchSize=size, center=ctr)
        # eagerplot(orig_cropped)

        # Rotate
        # orig_cropped = cv2.warpAffine(orig_cropped, M=M, dsize=size)
        # eagerplot(orig_cropped)

        # Crop out artifacts
        # croppedW = W if not rotated else H 
        # croppedH = H if not rotated else W
        # orig_cropped_rotated = cv2.getRectSubPix(orig_cropped, (int(croppedW*ZOOM), int(croppedH*ZOOM)), (size[0]/2, size[1]/2))
        # eagerplot(orig_cropped_rotated)

        # Add padding
        # box = addPadding(box)

    # Save
    if len(contours)-num_null_contours > 0:
        imsave(DIR_ORIG_CROPPED  + "Wound_" + str(idx) + ".jpg", orig_cropped)
        imsave(DIR_MASK9_CROPPED + "Wound_" + str(idx) + ".jpg", mask_cropped)

    # Plot mask, orig, orig_cropped_rotated
    # _, ax = plt.subplots(1,3)
    # ax[0].imshow(mask_gray)
    # ax[1].imshow(orig)
    # ax[2].imshow(mask_dilated)
    # ax[0].set_axis_off()
    # ax[1].set_axis_off()
    # ax[2].set_axis_off()

    # plt.show()
    # cv2.waitKey(0)

def process_layer1_output():
    """
    Takes in mask and orig images and output cropped originals to a directory
    """
    # Create directories if they do not exist
    if not os.path.isdir(DIR_ORIG_CROPPED):
        os.mkdir(DIR_ORIG_CROPPED)
    if not os.path.isdir(DIR_MASK9_CROPPED):
        os.mkdir(DIR_MASK9_CROPPED)
    
    # Get cropped images for all
    num_examples = 336
    for i in tqdm(range(1, num_examples+1)):
        get_bounding_box_for_one_pair(idx=i)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        get_bounding_box_for_one_pair()
    else:
        process_layer1_output()
