#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // Para medir el tiempo de ejecución

using namespace std;
using namespace cv;
using namespace chrono;

Mat applyBlurConvolution(const Mat& image, int kernelSize = 5) {
    // Definir el kernel de desenfoque
    Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (kernelSize * kernelSize);
    
    // Aplicar la convolución
    Mat blurredImage;
    filter2D(image, blurredImage, -1, kernel);
    
    return blurredImage;
}

int main() {
    // Cargar una imagen de ejemplo
    Mat inputImage = imread("input_image.jpg");

    if (inputImage.empty()) {
        cerr << "No se pudo cargar la imagen." << endl;
        return -1;
    }

    // Aplicar desenfoque por convolución
    auto start = high_resolution_clock::now();  // Iniciar temporizador
    Mat blurredImage = applyBlurConvolution(inputImage);
    auto end = high_resolution_clock::now();    // Detener temporizador

    // Calcular tiempo de ejecución en segundos
    duration<double> executionTime = duration_cast<duration<double>>(end - start);

    // Mostrar resultados
    imshow("Input Image", inputImage);
    imshow("Blurred Image (Convolution)", blurredImage);
    waitKey(0);

    // Mostrar métricas de rendimiento
    cout << "Tiempo de ejecución (Convolución): " << executionTime.count() << " segundos" << endl;

    return 0;
}
