import cv2
import numpy as np
from skimage import measure


def create_mask(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Apply thresholding to create a binary image
    _, thresh_img = cv2.threshold(blurred, 80,255, cv2.THRESH_BINARY)

    # Erode the binary image to remove noise
    thresh_img = cv2.erode(thresh_img, None, iterations=2)
    
    # Dilate the eroded image to restore eroded parts of the large components
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)
    
    # Perform connected component analysis on the thresholded image
    labels = measure.label(thresh_img, background=0, connectivity=2)  # using 2-connectivity for diagonal connections
    
    # Initialize a mask to store only the "large" components
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    
    # Loop over the unique components identified by the label function
    for label in np.unique(labels):
        # Ignore the background label
        if label == 0:
            continue
        
        # Construct a mask for the current label and count the number of pixels
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # Add to the mask if the number of pixels in the component is sufficiently large
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    
    return mask

# Example use
image = cv2.imread('image2.png')
mask = create_mask(image)
result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
cv2.imshow("Mask Image", mask)
cv2.imshow("Original Image", image)
cv2.imshow("Result Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()