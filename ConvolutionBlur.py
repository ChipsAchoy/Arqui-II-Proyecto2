import cv2
import numpy as np
import time

def apply_blur_convolution(image, kernel_size=5):
    # Definir el kernel de desenfoque
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # Aplicar la convolución
    start_time = time.time()
    blurred_image = cv2.filter2D(image, -1, kernel)
    end_time = time.time()
    
    # Calcular métricas de rendimiento
    execution_time = end_time - start_time
    return blurred_image, execution_time

# Cargar una imagen de ejemplo
input_image = cv2.imread('input_image.jpg')

# Aplicar desenfoque por convolución
blurred_image, execution_time = apply_blur_convolution(input_image)

# Mostrar resultados
cv2.imshow('Input Image', input_image)
cv2.imshow('Blurred Image (Convolution)', blurred_image)
cv2.waitKey(0)

# Mostrar métricas de rendimiento
print(f"Tiempo de ejecución (Convolución): {execution_time:.4f} segundos")
