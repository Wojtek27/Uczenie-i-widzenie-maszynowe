import numpy as np
import cv2

def my_resize(img, krotnosc):
    # pobierz wymiary obrazu
    height, width, channels = img.shape

    # oblicz nowe wymiary obrazu
    new_height = int(height / krotnosc)
    new_width = int(width / krotnosc)

    # stwórz pusty obraz wynikowy
    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # iteruj po nowym obrazie i interpoluj wartości pikseli
    for i in range(new_height):
        for j in range(new_width):
            x = int(j * krotnosc)
            y = int(i * krotnosc)
            new_img[i, j] = img[y, x]

    return new_img

def my_rotate(img, kat):
    # pobierz wymiary obrazu
    height, width, channels = img.shape

    # oblicz macierz transformacji
    transformation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), kat, 1)

    # oblicz nowe wymiary obrazu
    cosinus = np.abs(transformation_matrix[0, 0])
    sinus = np.abs(transformation_matrix[0, 1])
    new_width = int((height * sinus) + (width * cosinus))
    new_height = int((height * cosinus) + (width * sinus))

    # dostosuj macierz transformacji dla nowych wymiarów obrazu
    transformation_matrix[0, 2] += (new_width / 2) - (width / 2)
    transformation_matrix[1, 2] += (new_height / 2) - (height / 2)

    # wykonaj obrót obrazu
    new_img = cv2.warpAffine(img, transformation_matrix, (new_width, new_height))

    return new_img
image = cv2.imread('Pasikonik-w-kwadracie.jpg')
cv2.imshow('Orginal', image)
cv2.imshow('Zmienione',my_resize(image, 2))
cv2.imshow('Obrocone',my_rotate(image,90))
cv2.waitKey(0)
cv2.destroyAllWindows()