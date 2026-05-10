#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Punto.h"

using namespace std;

// Dibuja una línea entre dos puntos usando el algoritmo de Bresenham.
void dibujarLinea(cv::Mat& imagen, Punto p1, Punto p2, cv::Vec3b color);

// Dibuja el contorno de un polígono conectando sus puntos con líneas.
void dibujarPoligono(cv::Mat& imagen, vector<Punto> puntos, cv::Vec3b color);

// Calcula el área de un polígono usando la fórmula del zapatero.
double calcularAreaPoligono(const vector<Punto>& puntos);

// Determina si un punto está dentro de un polígono usando ray casting.
bool puntoDentroPoligono(Punto p, const vector<Punto>& poligono);

// Rellena un polígono verificando qué píxeles están dentro de la figura.
void rellenarPoligono(cv::Mat& imagen, const vector<Punto>& puntos, cv::Vec3b color);

// Dibuja un círculo usando simetría de ocho puntos (algoritmo de punto medio).
void dibujarCirculo(cv::Mat& imagen, Punto centro, int radio, cv::Vec3b color);

// Dibuja una elipse calculando sus puntos a partir de la ecuación paramétrica.
void dibujarElipse(cv::Mat& imagen, Punto centro, int radioX, int radioY, cv::Vec3b color);
