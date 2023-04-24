import numpy as np
import cv2

def demosaic_bayer_b(image):
    """
    Demosaic a Bayer BGRG image using nearest-neighbor interpolation.

    Parameters:
        image (numpy.ndarray): Input Bayer BGRG image.

    Returns:
        numpy.ndarray: Demosaiced RGB image.
    """
    height, width = image.shape[:2]
    bayer = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create an empty RGB image with the same dimensions as the input image
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Demosaic the image by interpolating missing colors
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            blue = bayer[y, x]
            green1 = bayer[y, x+1]
            green2 = bayer[y+1, x]
            red = bayer[y+1, x+1]

            # Nearest-neighbor interpolation for missing green pixels
            if x == 0:
                left = green1
            else:
                left = bayer[y, x-1]
            if x == width-2:
                right = green1
            else:
                right = bayer[y, x+2]
            if y == 0:
                top = green2
            else:
                top = bayer[y-1, x+1]
            if y == height-2:
                bottom = green2
            else:
                bottom = bayer[y+2, x+1]
            green = (left + right + top + bottom) // 4

            # Nearest-neighbor interpolation for missing blue and red pixels
            if y == 0:
                blue_top = blue
                red_top = red
            else:
                blue_top = bayer[y-1, x]
                red_top = bayer[y-1, x+1]
            if y == height-2:
                blue_bottom = blue
                red_bottom = red
            else:
                blue_bottom = bayer[y+2, x]
                red_bottom = bayer[y+2, x+1]
            if x == 0:
                blue_left = blue
                red_left = red
            else:
                blue_left = bayer[y, x-1]
                red_left = bayer[y+1, x-1]
            if x == width-2:
                blue_right = blue
                red_right = red
            else:
                blue_right = bayer[y, x+2]
                red_right = bayer[y+1, x+2]
            blue = (blue_top + blue_bottom + blue_left + blue_right) // 4
            red = (red_top + red_bottom + red_left + red_right) // 4

            # Set the RGB values for the corresponding pixel
            rgb[y, x] = blue
            rgb[y, x+1] = green
            rgb[y+1, x] = green
            rgb[y+1, x+1] = red

    return rgb

# Load the Bayer B image
bayer_b = cv2.imread("kwiatki.jpg")

# Demosaic the image using nearest-neighbor interpolation
rgb = demosaic_bayer_b(bayer_b)

# Display the demosaiced image
cv2.imshow("Demosaiced image", rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
