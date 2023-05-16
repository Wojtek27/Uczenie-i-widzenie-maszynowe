import cv2
from scipy.fftpack import dct, idct
from math import log10, sqrt
import numpy as np
import pywt


def compress_image(image_path, num_nonzero_coeffs):
    image = cv2.imread(image_path)
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image_ycrcb)
    y_coeffs = compress_channel(y, num_nonzero_coeffs)
    compressed_y = decompress_channel(y_coeffs, y.shape)
    cr_resized = cv2.resize(cr, compressed_y.shape[::-1], interpolation=cv2.INTER_LINEAR)
    cb_resized = cv2.resize(cb, compressed_y.shape[::-1], interpolation=cv2.INTER_LINEAR)
    compressed_y = compressed_y.astype(np.uint8)

    print("Original shapes:")
    print("Y shape:", y.shape, "dtype:", y.dtype)
    print("Cr shape:", cr.shape, "dtype:", cr.dtype)
    print("Cb shape:", cb.shape, "dtype:", cb.dtype)

    print("Compressed shapes:")
    print("Compressed Y shape:", compressed_y.shape, "dtype:", compressed_y.dtype)
    print("Cr resized shape:", cr_resized.shape, "dtype:", cr_resized.dtype)
    print("Cb resized shape:", cb_resized.shape, "dtype:", cb_resized.dtype)

    # Połączenie skompresowanych kanałów
    compressed_image_ycrcb = cv2.merge([compressed_y, cr_resized, cb_resized])

    # Konwersja z przestrzeni kolorów YCrCb do BGR
    compressed_image = cv2.cvtColor(compressed_image_ycrcb, cv2.COLOR_YCrCb2BGR)

    return compressed_image

def compress_channel(channel, num_nonzero_coeffs):
    channel_matrix = np.float32(channel)

    # Wykonanie falkowej transformacji 2D
    coeffs = cv2.dct(channel_matrix)

    # Sortowanie współczynników malejąco po wartości bezwzględnej
    sorted_coeffs = np.sort(np.abs(coeffs), axis=None)[::-1]

    # Wybór adaptacyjnej liczby niezerowych współczynników
    threshold = sorted_coeffs[num_nonzero_coeffs]
    quantized_coeffs = np.where(np.abs(coeffs) >= threshold, coeffs, 0)

    return quantized_coeffs


def decompress_channel(coeffs, shape):
    # Wykonanie odwrotnej falkowej transformacji 2D
    channel_matrix = cv2.idct(coeffs)

    # Zmiana rozmiaru macierzy na oryginalny
    channel = np.float32(cv2.resize(channel_matrix, shape))

    return channel


def calculate_psnr(original_image, compressed_image):
    # Zamiana kolejności wymiarów skompresowanego obrazu
    compressed_image = compressed_image.transpose(1, 0, 2)

    mse = np.mean((original_image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


original_image_path = "panda.jpg"
compressed_image_path = "panda-compressed_50.jpg"
num_nonzero_coeffs = 50

compressed_image = compress_image(original_image_path, num_nonzero_coeffs)
cv2.imwrite(compressed_image_path, compressed_image)        ######## TUTAJ TAK SAMO KIEPSKO xD, przy num_nonzero_coeffs = ~50 jest mniejszy rozmiar

original_image = cv2.imread(original_image_path)
psnr = calculate_psnr(original_image, compressed_image)

print("PSNR: {:.2f} dB".format(psnr))