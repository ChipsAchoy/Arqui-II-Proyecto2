#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;

Mat applyFFT(const Mat& imagePart) {
    // Convertir la imagen a escala de grises
    Mat grayImage;
    cvtColor(imagePart, grayImage, COLOR_BGR2GRAY);
    
    // Calcular FFT
    Mat complexImage;
    Mat padded;
    int m = getOptimalDFTSize(grayImage.rows);
    int n = getOptimalDFTSize(grayImage.cols);
    copyMakeBorder(grayImage, padded, 0, m - grayImage.rows, 0, n - grayImage.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planes, 2, complexImage);
    dft(complexImage, complexImage);
    
    // Calcular el espectro de magnitud en escala logarítmica
    split(complexImage, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magnitudeSpectrum = planes[0];
    magnitudeSpectrum += Scalar::all(1);
    log(magnitudeSpectrum, magnitudeSpectrum);
    
    return magnitudeSpectrum;
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

    // Aplicar FFT en cada proceso
    Mat magnitudeSpectrumPart = applyFFT(imagePart);

    // Recolectar todos los resultados en el proceso raíz (rank 0)
    if (rank == 0) {
        Mat magnitudeSpectrum = inputImage.clone();
        magnitudeSpectrum.rowRange(0, rowsPerProcess) = magnitudeSpectrumPart;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(imagePart.data, imagePart.total() * imagePart.elemSize(), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int startRow = i * rowsPerProcess;
            int endRow = startRow + rowsPerProcess;
            if (i < remainderRows) {
                endRow++;
            }
            imagePart.copyTo(magnitudeSpectrum.rowRange(startRow, endRow));
        }

        // Mostrar resultado
        imshow("Input Image", inputImage);
        Mat magnitudeSpectrumDisplay;
        normalize(magnitudeSpectrum, magnitudeSpectrumDisplay, 0, 255, NORM_MINMAX);
        magnitudeSpectrumDisplay.convertTo(magnitudeSpectrumDisplay, CV_8U);
        imshow("Magnitude Spectrum (FFT)", magnitudeSpectrumDisplay);
        waitKey(0);
    } else {
        MPI_Send(magnitudeSpectrumPart.data, magnitudeSpectrumPart.total() * magnitudeSpectrumPart.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
