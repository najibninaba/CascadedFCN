"""
Using OpenCV, get bounding box / ROI. 
1) Load mask
2) Get contours
finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.
3) Get bounding rectangle for a particular contour
4) Draw the rectangle
"""
import cv2
import numpy as np
from scipy.misc import imread, imsave
import matplotlib
matplotlib.use('TkAgg') # so that I can close the plot window
import matplotlib.pyplot as plt


def eagerplot(image):
    """
    Eager plotting for debugging
    """
    plt.imshow(image, cmap="gray")
    plt.show()

def addPadding(vertices):
    """
    Add padding. 4 pairs of coordinates
    """
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    x3, y3 = vertices[3]

    print(x3)

    return vertices

ORIGINAL_EXAMPLE = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/original/Wound_306.jpg"
MASK_EXAMPLE = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/masks-class-7/Wound_306.png"
MIN_AREA = 50
BOX_COLOR = (70,173,212) # color for bounding box is light neon blue
PADDING = 4 # pixels

# Load original
img = imread(ORIGINAL_EXAMPLE, mode="RGB")

# Load mask
mask = imread(MASK_EXAMPLE, mode="RGB")


# Convert mask to grayscale
mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

# Invert mask
mask_grayinv = 255 - mask_gray

# Convert mask to binary mask
retval, mask_bin = cv2.threshold(src=mask_grayinv, thresh=0, maxval=1, type=cv2.THRESH_BINARY)
eagerplot(mask_bin)

# Get contours based on binary image
mask_contour, contours, hierarchy = cv2.findContours(mask_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE, )
print("There are " + str(len(contours)) + " contour(s).")

for contour in contours:

    # Ignore contours less than MIN_AREA
    if cv2.contourArea(contour) < MIN_AREA:
        continue

    # Get bounding rectangle
    # x, y, w, h = cv2.boundingRect(contour)

    # Draws an up-right rectangle on original image
    # cv2.rectangle(img, pt1=(x-PADDING,y-PADDING), pt2=(x+w+PADDING+1,y+h+PADDING+1), color=BOX_COLOR, thickness=3)
    # cv2.rectangle(mask_gray, pt1=(x-PADDING,y-PADDING), pt2=(x+w+PADDING+1,y+h+PADDING+1), color=BOX_COLOR, thickness=3)

    # Get bounding rotated rectangle
    rect = cv2.minAreaRect(contour)
    box  = cv2.boxPoints(rect) # box = (x,y,w,h,theta) where (x,y) is the centre point and theta is the angle of rotation
    box  = np.int0(box)

    # Add padding
    # box = addPadding(box)

    # Draws a rotated rectangle on original image
    cv2.drawContours(img, contours=[box], contourIdx=0, color=BOX_COLOR, thickness=3)
    cv2.drawContours(mask_gray, contours=[box], contourIdx=0, color=BOX_COLOR, thickness=3)

eagerplot(img)
eagerplot(mask_gray)

# Save file

# if __name__ == "__main__":
#     continue
