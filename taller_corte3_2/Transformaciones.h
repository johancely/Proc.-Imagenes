#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Punto.h"

using namespace std;

// Traslada una imagen moviendo cada píxel según dx y dy.
cv::Mat trasladarImagen(const cv::Mat& imagen, int dx, int dy);

// Escala una imagen aplicando factores de escala en X y Y (mapeo inverso).
cv::Mat escalarImagen(const cv::Mat& imagen, double sx, double sy);

// Rota una imagen alrededor de su centro usando la matriz de rotación inversa.
cv::Mat rotarImagen(const cv::Mat& imagen, double anguloGrados);

// Recorta una región de interés dejando visibles solo los píxeles dentro del polígono.
cv::Mat recortarROI(const cv::Mat& imagen, const vector<Punto>& poligono);

// Aplica una traslación únicamente a los píxeles dentro del polígono.
cv::Mat trasladarROI(const cv::Mat& imagen, const vector<Punto>& poligono, int dx, int dy);
