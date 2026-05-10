#include "Geometria.h"
#include "Imagen.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <cstdlib>

using namespace std;

// Dibuja una línea entre dos puntos usando el algoritmo de Bresenham.
void dibujarLinea(cv::Mat& imagen, Punto p1, Punto p2, cv::Vec3b color) {
    int x0 = p1.x, y0 = p1.y;
    int x1 = p2.x, y1 = p2.y;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);

    // Determinar la dirección de avance en cada eje.
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;

    int error = dx - dy;

    while (true) {
        pintarPixel(imagen, x0, y0, color);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * error;

        if (e2 > -dy) {
            error -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            error += dx;
            y0 += sy;
        }
    }
}

// Dibuja el contorno de un polígono conectando sus puntos con líneas.
void dibujarPoligono(cv::Mat& imagen, vector<Punto> puntos, cv::Vec3b color) {
    int n = puntos.size();
    if (n < 2) return;

    for (int i = 0; i < n; i++) {
        // Conecta cada vértice con el siguiente; el último cierra con el primero.
        Punto siguiente = puntos[(i + 1) % n];
        dibujarLinea(imagen, puntos[i], siguiente, color);
    }
}

// Calcula el área de un polígono usando la fórmula del zapatero (Shoelace).
double calcularAreaPoligono(const vector<Punto>& puntos) {
    int n = puntos.size();
    if (n < 3) return 0.0;

    double suma = 0.0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        suma += (double)puntos[i].x * puntos[j].y;
        suma -= (double)puntos[j].x * puntos[i].y;
    }

    return abs(suma) / 2.0;
}

// Determina si un punto está dentro de un polígono usando ray casting.
bool puntoDentroPoligono(Punto p, const vector<Punto>& poligono) {
    int n = static_cast<int>(poligono.size());
    bool dentro = false;

    for (int i = 0, j = n - 1; i < n; j = i++) {
        int xi = poligono[i].x, yi = poligono[i].y;
        int xj = poligono[j].x, yj = poligono[j].y;

        // Verificar si el rayo horizontal cruza el lado del polígono.
        bool cruce = ((yi > p.y) != (yj > p.y)) &&
                     (p.x < (double)(xj - xi) * (p.y - yi) / (yj - yi) + xi);

        if (cruce) dentro = !dentro;
    }

    return dentro;
}

// Rellena un polígono verificando qué píxeles están dentro de la figura.
void rellenarPoligono(cv::Mat& imagen, const vector<Punto>& puntos, cv::Vec3b color) {
    if (puntos.empty()) return;

    // Calcular el rectángulo delimitador (bounding box) del polígono.
    int xMin = puntos[0].x, xMax = puntos[0].x;
    int yMin = puntos[0].y, yMax = puntos[0].y;

    for (const Punto& p : puntos) {
        xMin = min(xMin, p.x);
        xMax = max(xMax, p.x);
        yMin = min(yMin, p.y);
        yMax = max(yMax, p.y);
    }

    // Limitar a los límites de la imagen.
    xMin = max(xMin, 0);
    yMin = max(yMin, 0);
    xMax = min(xMax, imagen.cols - 1);
    yMax = min(yMax, imagen.rows - 1);

    // Pintar cada píxel del bounding box que esté dentro del polígono.
    for (int y = yMin; y <= yMax; y++) {
        for (int x = xMin; x <= xMax; x++) {
            if (puntoDentroPoligono(Punto(x, y), puntos)) {
                pintarPixel(imagen, x, y, color);
            }
        }
    }
}

// Pinta los 8 puntos simétricos de un círculo (simetría octagonal).
static void pintarOctantes(cv::Mat& imagen, Punto centro, int dx, int dy, cv::Vec3b color) {
    pintarPixel(imagen, centro.x + dx, centro.y + dy, color);
    pintarPixel(imagen, centro.x - dx, centro.y + dy, color);
    pintarPixel(imagen, centro.x + dx, centro.y - dy, color);
    pintarPixel(imagen, centro.x - dx, centro.y - dy, color);
    pintarPixel(imagen, centro.x + dy, centro.y + dx, color);
    pintarPixel(imagen, centro.x - dy, centro.y + dx, color);
    pintarPixel(imagen, centro.x + dy, centro.y - dx, color);
    pintarPixel(imagen, centro.x - dy, centro.y - dx, color);
}

// Dibuja un círculo usando simetría de ocho puntos (algoritmo de punto medio).
void dibujarCirculo(cv::Mat& imagen, Punto centro, int radio, cv::Vec3b color) {
    int x = 0;
    int y = radio;
    int decision = 1 - radio; // Parámetro de decisión inicial del algoritmo.

    pintarOctantes(imagen, centro, x, y, color);

    while (x < y) {
        x++;
        if (decision < 0) {
            decision += 2 * x + 1;
        } else {
            y--;
            decision += 2 * (x - y) + 1;
        }
        pintarOctantes(imagen, centro, x, y, color);
    }
}

// Dibuja una elipse calculando sus puntos a partir de la ecuación paramétrica.
void dibujarElipse(cv::Mat& imagen, Punto centro, int radioX, int radioY, cv::Vec3b color) {
    // Recorrer los 360 grados y calcular cada punto con la ecuación paramétrica.
    for (int angulo = 0; angulo < 360; angulo++) {
        double rad = angulo * M_PI / 180.0;
        int x = (int)round(centro.x + radioX * cos(rad));
        int y = (int)round(centro.y + radioY * sin(rad));
        pintarPixel(imagen, x, y, color);
    }
}
