#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // Para medir el tiempo de ejecución

using namespace std;
using namespace cv;
using namespace chrono;

Mat applyFFT(const Mat& image) {
    // Convertir la imagen a escala de grises
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    
    // Calcular FFT
    Mat complexImage;
    Mat padded; // Para optimizar el tamaño de la imagen de entrada
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

int main() {
    // Cargar una imagen de ejemplo
    Mat inputImage = imread("input_image.jpg");

    if (inputImage.empty()) {
        cerr << "No se pudo cargar la imagen." << endl;
        return -1;
    }

    // Aplicar FFT
    auto start = high_resolution_clock::now();  // Iniciar temporizador
    Mat magnitudeSpectrum = applyFFT(inputImage);
    auto end = high_resolution_clock::now();    // Detener temporizador

    // Calcular tiempo de ejecución en segundos
    duration<double> executionTime = duration_cast<duration<double>>(end - start);

    // Mostrar resultados
    imshow("Input Image", inputImage);
    Mat magnitudeSpectrumDisplay;
    normalize(magnitudeSpectrum, magnitudeSpectrumDisplay, 0, 255, NORM_MINMAX);
    magnitudeSpectrumDisplay.convertTo(magnitudeSpectrumDisplay, CV_8U);
    imshow("Magnitude Spectrum (FFT)", magnitudeSpectrumDisplay);
    waitKey(0);

    // Mostrar métricas de rendimiento
    cout << "Tiempo de ejecución (FFT): " << executionTime.count() << " segundos" << endl;

    return 0;
}
