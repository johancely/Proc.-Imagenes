#pragma once

// Representa un punto en el plano de la imagen con coordenadas enteras.
struct Punto {
    int x;
    int y;

    Punto(int x = 0, int y = 0) : x(x), y(y) {}
};
