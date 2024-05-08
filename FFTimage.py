import cv2
import numpy as np
import time

def apply_fft(image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular FFT
    start_time = time.time()
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    end_time = time.time()
    
    # Calcular métricas de rendimiento
    execution_time = end_time - start_time
    return magnitude_spectrum, execution_time

# Cargar una imagen de ejemplo
input_image = cv2.imread('input_image.jpg')

# Aplicar FFT
magnitude_spectrum, execution_time = apply_fft(input_image)

# Mostrar resultados
cv2.imshow('Input Image', input_image)
cv2.imshow('Magnitude Spectrum (FFT)', magnitude_spectrum.astype(np.uint8))
cv2.waitKey(0)

# Mostrar métricas de rendimiento
print(f"Tiempo de ejecución (FFT): {execution_time:.4f} segundos")
