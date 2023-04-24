import cv2
import numpy as np
from skimage import io, filters
from matplotlib import pyplot as plt

# Wczytanie obrazu
img = io.imread('kwiatki.jpg')

# Konwersja obrazu na skale szarosci
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dodanie szumu typu Poisson
noisy_img = np.random.poisson(gray_img)

# Transformatę Anscomba
anscombe_img = 2 * np.sqrt(noisy_img + 3/8)

# Usunięcie szumu za pomocą mediany
denoised_anscombe_img = filters.median(anscombe_img, selem=None, out=None)

# Odtworzenie obrazu
denoised_img = (denoised_anscombe_img/2)**2 - 3/8

# Wyświetlenie wyników
plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(gray_img, cmap='gray'), plt.title('Obraz oryginalny')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(anscombe_img, cmap='gray'), plt.title('Po transformacie Anscomba')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(denoised_anscombe_img, cmap='gray'), plt.title('Po odszumianiu')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(denoised_img, cmap='gray'), plt.title('Odtworzony obraz')
plt.xticks([]), plt.yticks([])
plt.show()
