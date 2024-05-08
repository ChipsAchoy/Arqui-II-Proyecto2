from mpi4py import MPI
import cv2
import numpy as np
import time

def apply_fft(image_part):
    # Convertir la imagen a escala de grises
    gray_image_part = cv2.cvtColor(image_part, cv2.COLOR_BGR2GRAY)
    
    # Calcular FFT
    f = np.fft.fft2(gray_image_part)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum_part = 20 * np.log(np.abs(fshift))
    
    return magnitude_spectrum_part

# Inicializar MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Cargar la imagen en el proceso raíz (rank 0)
if rank == 0:
    input_image = cv2.imread('input_image.jpg')
    height, width = input_image.shape[:2]
else:
    input_image = None
    height, width = 0, 0

# Compartir dimensiones de la imagen entre todos los procesos
height = comm.bcast(height, root=0)
width = comm.bcast(width, root=0)

# Calcular el número de filas por proceso
rows_per_process = height // size
remainder_rows = height % size

# Distribuir las filas de la imagen entre los procesos
if rank == 0:
    start_row = 0
    for i in range(size):
        end_row = start_row + rows_per_process
        if i < remainder_rows:
            end_row += 1
        comm.send(input_image[start_row:end_row, :], dest=i, tag=11)
        start_row = end_row

# Cada proceso recibe su parte de la imagen
image_part = comm.recv(source=0, tag=11)

# Aplicar FFT en cada proceso
magnitude_spectrum_part = apply_fft(image_part)

# Recolectar todos los resultados en el proceso raíz (rank 0)
if rank == 0:
    magnitude_spectrum = np.empty(input_image.shape[:2], dtype=np.float32)
else:
    magnitude_spectrum = None

# Reunir las partes del espectro de magnitud en el proceso raíz
comm.Gather(magnitude_spectrum_part, magnitude_spectrum, root=0)

# Mostrar imagen resultante en el proceso raíz
if rank == 0:
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
    cv2.imshow('Input Image', input_image)
    cv2.imshow('Magnitude Spectrum (MPI)', magnitude_spectrum)
    cv2.waitKey(0)
