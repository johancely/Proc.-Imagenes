#include "Imagen.h"
#include <iostream>

using namespace std;

// Carga una imagen desde una ruta usando OpenCV.
cv::Mat cargarImagen(const string& ruta) {
    cv::Mat imagen = cv::imread(ruta);
    if (imagen.empty()) {
        cerr << "[Error] No se pudo cargar la imagen en: " << ruta << endl;
    }
    return imagen;
}

// Guarda la imagen procesada en una ruta indicada.
bool guardarImagen(const string& ruta, const cv::Mat& imagen) {
    bool exito = cv::imwrite(ruta, imagen);
    if (!exito) {
        cerr << "[Error] No se pudo guardar la imagen en: " << ruta << endl;
    }
    return exito;
}

// Verifica que un punto esté dentro de los límites de la imagen.
bool puntoValido(const cv::Mat& imagen, int x, int y) {
    return x >= 0 && x < imagen.cols && y >= 0 && y < imagen.rows;
}

// Retorna el color de un píxel en la posición indicada.
cv::Vec3b obtenerPixel(const cv::Mat& imagen, int x, int y) {
    if (!puntoValido(imagen, x, y)) {
        return cv::Vec3b(0, 0, 0);
    }
    return imagen.at<cv::Vec3b>(y, x);
}

// Cambia el color de un píxel si está dentro de la imagen.
void pintarPixel(cv::Mat& imagen, int x, int y, cv::Vec3b color) {
    if (puntoValido(imagen, x, y)) {
        imagen.at<cv::Vec3b>(y, x) = color;
    }
}
