#include "Transformaciones.h"
#include "Geometria.h"
#include "Imagen.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Traslada una imagen moviendo cada píxel según dx y dy (mapeo directo).
cv::Mat trasladarImagen(const cv::Mat& imagen, int dx, int dy) {
    // Imagen de destino inicializada en negro.
    cv::Mat resultado = cv::Mat::zeros(imagen.size(), imagen.type());

    for (int y = 0; y < imagen.rows; y++) {
        for (int x = 0; x < imagen.cols; x++) {
            int nuevoX = x + dx;
            int nuevoY = y + dy;
            if (puntoValido(resultado, nuevoX, nuevoY)) {
                resultado.at<cv::Vec3b>(nuevoY, nuevoX) = imagen.at<cv::Vec3b>(y, x);
            }
        }
    }

    return resultado;
}

// Escala una imagen aplicando factores de escala en X y Y (mapeo inverso para evitar huecos).
cv::Mat escalarImagen(const cv::Mat& imagen, double sx, double sy) {
    if (sx <= 0 || sy <= 0) return imagen.clone();

    int nuevoAncho = (int)round(imagen.cols * sx);
    int nuevoAlto  = (int)round(imagen.rows * sy);

    cv::Mat resultado = cv::Mat::zeros(nuevoAlto, nuevoAncho, imagen.type());

    // Recorrer la imagen destino y buscar de dónde proviene cada píxel (mapeo inverso).
    for (int y = 0; y < nuevoAlto; y++) {
        for (int x = 0; x < nuevoAncho; x++) {
            int origenX = (int)round(x / sx);
            int origenY = (int)round(y / sy);
            if (puntoValido(imagen, origenX, origenY)) {
                resultado.at<cv::Vec3b>(y, x) = imagen.at<cv::Vec3b>(origenY, origenX);
            }
        }
    }

    return resultado;
}

// Rota una imagen alrededor de su centro usando la matriz de rotación inversa.
cv::Mat rotarImagen(const cv::Mat& imagen, double anguloGrados) {
    double rad = anguloGrados * M_PI / 180.0;
    double cosA = cos(rad);
    double sinA = sin(rad);

    // Centro de la imagen.
    double cx = imagen.cols / 2.0;
    double cy = imagen.rows / 2.0;

    cv::Mat resultado = cv::Mat::zeros(imagen.size(), imagen.type());

    // Mapeo inverso: para cada píxel destino, calcular su origen en la imagen fuente.
    for (int y = 0; y < resultado.rows; y++) {
        for (int x = 0; x < resultado.cols; x++) {
            double dx = x - cx;
            double dy = y - cy;

            int origenX = (int)round(cosA * dx + sinA * dy + cx);
            int origenY = (int)round(-sinA * dx + cosA * dy + cy);

            if (puntoValido(imagen, origenX, origenY)) {
                resultado.at<cv::Vec3b>(y, x) = imagen.at<cv::Vec3b>(origenY, origenX);
            }
        }
    }

    return resultado;
}

// Recorta una región de interés dejando visibles solo los píxeles dentro del polígono.
cv::Mat recortarROI(const cv::Mat& imagen, const vector<Punto>& poligono) {
    // Imagen negra del mismo tamaño como fondo.
    cv::Mat resultado = cv::Mat::zeros(imagen.size(), imagen.type());

    for (int y = 0; y < imagen.rows; y++) {
        for (int x = 0; x < imagen.cols; x++) {
            if (puntoDentroPoligono(Punto(x, y), poligono)) {
                resultado.at<cv::Vec3b>(y, x) = imagen.at<cv::Vec3b>(y, x);
            }
        }
    }

    return resultado;
}

// Aplica una traslación únicamente a los píxeles dentro del polígono.
cv::Mat trasladarROI(const cv::Mat& imagen, const vector<Punto>& poligono, int dx, int dy) {
    cv::Mat resultado = imagen.clone();

    // Borrar los píxeles originales dentro del ROI.
    for (int y = 0; y < imagen.rows; y++) {
        for (int x = 0; x < imagen.cols; x++) {
            if (puntoDentroPoligono(Punto(x, y), poligono)) {
                resultado.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    // Mover los píxeles del ROI a su nueva posición.
    for (int y = 0; y < imagen.rows; y++) {
        for (int x = 0; x < imagen.cols; x++) {
            if (puntoDentroPoligono(Punto(x, y), poligono)) {
                int nuevoX = x + dx;
                int nuevoY = y + dy;
                if (puntoValido(resultado, nuevoX, nuevoY)) {
                    resultado.at<cv::Vec3b>(nuevoY, nuevoX) = imagen.at<cv::Vec3b>(y, x);
                }
            }
        }
    }

    return resultado;
}
