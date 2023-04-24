"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# MINIMUM #

img = cv2.imread('kwiatki.jpg', 0)

lambdas = [1, 4, 8, 64, 256, 1024]

plt.figure(figsize=(16,10))
plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Obraz oryginalny')

i = 2
for l in lambdas:
    poisson_noise = np.random.poisson(l, img.shape).astype(np.float32)
    noisy_img = img + poisson_noise
    plt.subplot(2, 4, i)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f'Szum Poissona\nλ = {l}')
    i += 1

plt.show()
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# DOMYŚLNE #

# Load the original image
img = cv2.imread('kwiatki_inny.jpg', cv2.IMREAD_GRAYSCALE)

# Define lambda values
lambdas = [1, 4, 8, 64, 256, 1024]

# Define interpolation methods
interpolation_methods = ['nearest', 'linear', 'area', 'cubic']

# Define image size
image_size = (600, 600)

# Generate and display interpolated images for each lambda and interpolation method
for lam in lambdas:
    # Generate noisy image using Poisson noise
    noisy_img = np.random.poisson(img / 255.0 * lam) / float(lam) * 255.0

    # Generate interpolated images for each interpolation method
    interpolated_images = []
    for method in interpolation_methods:
        interpolated_img = cv2.resize(noisy_img, image_size, interpolation=getattr(cv2, 'INTER_' + method.upper()))
        interpolated_images.append(interpolated_img)

    # Display results
    fig, axes = plt.subplots(nrows=1, ncols=len(interpolation_methods) + 1, figsize=(20, 5))

    for i, cell in enumerate(axes):
        if i == 0:
            cell.imshow(noisy_img, cmap='gray')
            cell.set_title('λ={}'.format(lam))
        else:
            cell.imshow(interpolated_images[i - 1], cmap='gray')
            cell.set_title('{}'.format(interpolation_methods[i - 1]))

        cell.axis('off')

    plt.tight_layout()
    plt.show()
