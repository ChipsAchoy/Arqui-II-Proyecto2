from mpi4py import MPI
import cv2
import numpy as np
import time

def apply_blur_convolution(image_part, kernel_size=5):
    # Definir el kernel de desenfoque
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # Aplicar la convolución
    blurred_image_part = cv2.filter2D(image_part, -1, kernel)
    
    return blurred_image_part

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

# Aplicar desenfoque por convolución en cada proceso
blurred_image_part = apply_blur_convolution(image_part)

# Recolectar todos los resultados en el proceso raíz (rank 0)
if rank == 0:
    result_image = np.empty(input_image.shape, dtype=np.uint8)
else:
    result_image = None

comm.Gather(blurred_image_part, result_image, root=0)

# Mostrar imagen resultante en el proceso raíz
if rank == 0:
    cv2.imshow('Input Image', input_image)
    cv2.imshow('Blurred Image (MPI)', result_image)
    cv2.waitKey(0)
