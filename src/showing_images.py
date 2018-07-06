import os
import time
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np


def get_stats(images, names, num, show=True):
    '''
    Function that gets some stats of the image given a stack of images and a number to specify the image you want to look into

    Inputs:
    images: Stack of images
    names: names of the stack of images
    num: image number of the stack
    show: Boolean. if set to True, it will print out the outputs

    Outputs:
    - the total number of images in the stack
    - choosen image's shape
    - highest pixel value of the image
    - lowest pixel value of the image
    '''

    if show:
        print("++++++++++++++++++++++++++++++++++++++++")
        print("Total number of images: {}".format(len(images)))
        print("Image shape: {}".format(images[num].shape))
        print("max pixel value: {}".format(np.max(images[num])))
        print("min pixel value: {}".format(np.min(images[num])))
        print("Showing image: {}".format(names[num]))
    return len(images), images[num].shape, np.max(images[num]), np.min(images[num])


def get_all_classes_and_counts(images, show=True):
    '''
    Function that gets all the different classes and counts for the given stack of images

    Inputs:
    images: Stack of images
    show: Boolean. if set to True, it will print out the outputs

    Output:
    returns a dictionary containing the classes and their respective counts
    '''

    # TODO Add feature to take in unlabeled images
    d = dict()
    for i in images:
        for j in i:
            for k in j:
                if k not in d:
                    d[k] = 1
                else:
                    d[k] += 1
    if show:
        print("Number of different classes and their counts for the stack of {} images is given as {}".format(
            len(images), d))
    return d


def get_classes_and_counts(image, images, show=True):
    '''
    Function that gets all the different classes and counts for the given image

    Inputs:
    image: An image
    images: A stack of images
    show: Boolean. if set to True, it will print out the outputs

    Output:
    returns a dictionary containing the classes and their respective counts
    '''

    # TODO Add feature to take in unlabeled image
    d = dict()
    for i in images:
        for j in i:
            for k in j:
                if k not in d:
                    d[k] = 0

    for i in image:
        for j in i:
            d[j] += 1
    if show:
        print("Image size : {}. \n Number of different classes and their counts for the image is given as {}".format(
            image.shape, d))
    return d


def get_all_pixel_levels(images):
    '''
    Function that gets all the different pixel levels for a given stack of images

    Inputs:
    images: a stack of images

    Outputs:
    returns a sorted list of different pixel values
    '''
    temp = []
    for i in images:
        for j in i:
            for k in j:
                if k not in temp:
                    temp.append(k)
    temp.sort()
    return temp


def get_pixel_levels(image):
    '''
    Function that gets all the different pixel levels for a given image

    Inputs:
    image: An image

    Outputs:
    returns a sorted list of different pixel values
    '''
    temp = []
    for i in image:
        for j in i:
            if j not in temp:
                temp.append(j)
    temp.sort()
    return temp


if __name__ == '__main__':
    print("Loading images")
    tic = time.time()
    data_path = os.path.join(os.getcwd(), 'data')

    # masks_v3
    masks_v3_path = os.path.join(data_path, 'Masks_v3')
    masks_v3_names = os.listdir(masks_v3_path)
    masks_v3_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    masks_v3 = [imread(os.path.join(masks_v3_path, x)) for x in masks_v3_names]

    # masks_seg_v3
    masks_seg_v3_path = os.path.join(data_path, 'Masks_Seg_v3')
    masks_seg_v3_names = os.listdir(masks_seg_v3_path)
    masks_seg_v3_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    masks_seg_v3 = [imread(os.path.join(masks_seg_v3_path, x)) for x in masks_seg_v3_names]

    # mask_labeled_full_modified
    mask_labeled_full_modified_path = os.path.join(data_path, 'Mask_Labeled_Full_Modified')
    mask_labeled_full_modified_names = os.listdir(mask_labeled_full_modified_path)
    mask_labeled_full_modified_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mask_labeled_full_modified = [imread(os.path.join(
        mask_labeled_full_modified_path, x)) for x in mask_labeled_full_modified_names]

    # mask_labeled_wound_3class
    mask_labeled_wound_3class_path = os.path.join(data_path, 'Mask_Labeled_Wound_3Class')
    mask_labeled_wound_3class_names = os.listdir(mask_labeled_wound_3class_path)
    mask_labeled_wound_3class_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mask_labeled_wound_3class = [imread(os.path.join(
        mask_labeled_wound_3class_path, x)) for x in mask_labeled_wound_3class_names]

    # mask_labeled_tissue
    mask_labeled_tissue_path = os.path.join(data_path, 'Mask_Labeled_Tissue')
    mask_labeled_tissue_names = os.listdir(mask_labeled_tissue_path)
    mask_labeled_tissue_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mask_labeled_tissue = [imread(os.path.join(mask_labeled_tissue_path, x))
                           for x in mask_labeled_tissue_names]

    # original image
    original_images_v3_path = os.path.join(data_path, 'Original_Images_v3')
    original_images_v3_names = os.listdir(original_images_v3_path)
    original_images_v3_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    original_images_v3 = [imread(os.path.join(original_images_v3_path, x))
                          for x in original_images_v3_names]

    print("Finished loading images. Time taken: {}".format(time.time() - tic))

    # getting stats
    num = int(input("Image num:"))
    print("Original Image stats")
    get_stats(original_images_v3, original_images_v3_names, num)

    print("Masks v3 stats")
    get_stats(masks_v3, masks_v3_names, num)

    print("Masks Seg v3 stats")
    get_stats(masks_seg_v3, masks_seg_v3_names, num)

    print("Masks Labeled Full Modified stats")
    get_stats(mask_labeled_full_modified, mask_labeled_full_modified_names, num)
    get_classes_and_counts(mask_labeled_full_modified[num], mask_labeled_full_modified)
    print("Mask Labeled Wound 3Class stats")
    get_stats(mask_labeled_wound_3class, mask_labeled_wound_3class_names, num)
    get_classes_and_counts(mask_labeled_wound_3class[num], mask_labeled_full_modified)
    print("Mask Labeled Tissue stats")
    get_stats(mask_labeled_tissue, mask_labeled_tissue_names, num)
    get_classes_and_counts(mask_labeled_tissue[num], mask_labeled_full_modified)

    # writing to file
    # TODO

    # plotting

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].imshow(original_images_v3[num])
    ax[0, 0].set_axis_off()
    ax[0, 0].set_title("Original Image")

    ax[0, 1].imshow(masks_v3[num])
    ax[0, 1].set_axis_off()
    ax[0, 1].set_title("Masks_v3")

    ax[0, 2].imshow(masks_seg_v3[num])
    ax[0, 2].set_axis_off()
    ax[0, 2].set_title("Masks_Seg_v3")

    ax[1, 0].imshow(mask_labeled_full_modified[num])
    ax[1, 0].set_axis_off()
    ax[1, 0].set_title("Mask_Labeled_Full_Modified")

    ax[1, 1].imshow(mask_labeled_wound_3class[num])
    ax[1, 1].set_axis_off()
    ax[1, 1].set_title("Mask_Labeled_Wound_3Class")

    ax[1, 2].imshow(mask_labeled_tissue[num])
    ax[1, 2].set_axis_off()
    ax[1, 2].set_title("Mask_Labeled_Tissue")

    plt.show()
