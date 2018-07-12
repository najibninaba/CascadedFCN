"""
Using OpenCV, get bounding box / ROI. 
1) Load mask
2) Get contours
finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.
3) Get bounding rectangle for a particular contour
4) Draw the rectangl

img -> img_cropped
mask -> mask_gray -> mask_grayinv -> mask_contour
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

def add_padding(vertices):
    """
    Add padding. 4 pairs of coordinates
    """
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    x3, y3 = vertices[3]

    print(x3)

    return vertices

WOUND_IDX = 207
ORIGINAL_EXAMPLE = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/original/Wound_" + str(WOUND_IDX) + ".jpg"
MASK_EXAMPLE = "/Users/raimibinkarim/Desktop/CascadedFCN-Data/masks-class-7/Wound_" + str(WOUND_IDX) + ".png"
MIN_AREA = 100
BOX_COLOR = (70, 173, 212) # color for bounding box is light neon blue
PADDING = 4 # pixels
mult = 1.2

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
    rect = cv2.minAreaRect(contour) # rect = (x,y,w,h,theta) where (x,y) is the centre point and theta is the angle of rotation
    box  = cv2.boxPoints(rect) 
    box  = np.int0(box)

    # Draws a rotated rectangle on original image
    cv2.drawContours(img, contours=[box], contourIdx=0, color=BOX_COLOR, thickness=3)
    cv2.drawContours(mask_gray, contours=[box], contourIdx=0, color=BOX_COLOR, thickness=3)
    
    # Get coordinates, w, h, angle
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    angle = rect[2]
    if angle < -45:
        angle += 90
        rotated = True
    else:
        rotated = False

    ctr  = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)), int(mult*(y2-y1)))
    # cv2.circle(img, ctr, 10, BOX_COLOR, -1) # again this was mostly for debugging purposes

    # Rotate
    M = cv2.getRotationMatrix2D(center=(size[0]/2, size[1]/2), angle=angle, scale=1.0)

    # Retrieve pixel rectangle
    img_cropped = cv2.getRectSubPix(img, patchSize=size, center=ctr)
    eagerplot(img_cropped)

    # Rotate
    img_cropped = cv2.warpAffine(img_cropped, M=M, dsize=size)
    eagerplot(img_cropped)

    # Crop out artifacts
    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W
    img_cropped_rotated = cv2.getRectSubPix(img_cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))
    eagerplot(img_cropped_rotated)

    # Add padding
    # box = addPadding(box)



# Crop image at boundary


# eagerplot(img)

fig, ax = plt.subplots(1,3)
ax[0].imshow(mask_bin)
ax[0].set_axis_off()

ax[1].imshow(img)
ax[1].set_axis_off()

ax[2].imshow(img_cropped_rotated)
ax[2].set_axis_off()

plt.show()
cv2.waitKey(0)

# Save file


# if __name__ == "__main__":
    # continue
