#pragma once
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

// Carga una imagen desde una ruta usando OpenCV.
cv::Mat cargarImagen(const string& ruta);

// Guarda la imagen procesada en una ruta indicada.
bool guardarImagen(const string& ruta, const cv::Mat& imagen);

// Verifica que un punto esté dentro de los límites de la imagen.
bool puntoValido(const cv::Mat& imagen, int x, int y);

// Retorna el color de un píxel en la posición indicada.
cv::Vec3b obtenerPixel(const cv::Mat& imagen, int x, int y);

// Cambia el color de un píxel si está dentro de la imagen.
void pintarPixel(cv::Mat& imagen, int x, int y, cv::Vec3b color);
