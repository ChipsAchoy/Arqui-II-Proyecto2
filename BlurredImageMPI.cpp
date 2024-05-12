#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;

Mat applyBlurConvolution(const Mat& imagePart, int kernelSize = 5) {
    // Definir el kernel de desenfoque
    Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (kernelSize * kernelSize);
    
    // Aplicar la convolución
    Mat blurredImagePart;
    filter2D(imagePart, blurredImagePart, -1, kernel);
    
    return blurredImagePart;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Cargar una imagen de ejemplo en el proceso raíz (rank 0)
    Mat inputImage;
    if (rank == 0) {
        inputImage = imread("input_image.jpg");
        if (inputImage.empty()) {
            cerr << "No se pudo cargar la imagen." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Compartir dimensiones de la imagen entre todos los procesos
    int rows, cols;
    if (rank == 0) {
        rows = inputImage.rows;
        cols = inputImage.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcular el número de filas por proceso
    int rowsPerProcess = rows / size;
    int remainderRows = rows % size;

    // Distribuir las filas de la imagen entre los procesos
    Mat imagePart;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int startRow = i * rowsPerProcess;
            int endRow = startRow + rowsPerProcess;
            if (i < remainderRows) {
                endRow++;
            }
            imagePart = inputImage.rowRange(startRow, endRow);
            if (i != 0) {
                MPI_Send(imagePart.data, imagePart.total() * imagePart.elemSize(), MPI_BYTE, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        imagePart = Mat::zeros(rowsPerProcess, cols, inputImage.type());
        MPI_Recv(imagePart.data, imagePart.total() * imagePart.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Aplicar desenfoque por convolución en cada proceso
    Mat blurredImagePart = applyBlurConvolution(imagePart);

    // Recolectar todos los resultados en el proceso raíz (rank 0)
    if (rank == 0) {
        Mat resultImage = inputImage.clone();
        resultImage.rowRange(0, rowsPerProcess) = blurredImagePart;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(imagePart.data, imagePart.total() * imagePart.elemSize(), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int startRow = i * rowsPerProcess;
            int endRow = startRow + rowsPerProcess;
            if (i < remainderRows) {
                endRow++;
            }
            imagePart.copyTo(resultImage.rowRange(startRow, endRow));
        }

        // Mostrar resultado
        imshow("Input Image", inputImage);
        imshow("Blurred Image (Convolution)", resultImage);
        waitKey(0);
    } else {
        MPI_Send(blurredImagePart.data, blurredImagePart.total() * blurredImagePart.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
