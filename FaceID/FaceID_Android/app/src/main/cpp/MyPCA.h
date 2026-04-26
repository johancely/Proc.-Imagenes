#ifndef MY_PCA_H
#define MY_PCA_H

#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

class MyPCA {
public:
    // Constructor: al crear el objeto entrena PCA con las rutas recibidas.
    explicit MyPCA(vector<string>& facesPath) {
        build(facesPath);
    }

    // Flujo PCA:
    // 1) vectoriza cada imagen 100x100
    // 2) calcula cara promedio
    // 3) centra datos (x - media)
    // 4) calcula eigenvectores (eigenfaces)
    void build(vector<string>& facesPath) {
        Mat sample = imread(facesPath[0], IMREAD_GRAYSCALE);
        imgRows = sample.rows;
        imgSize = sample.rows * sample.cols;

        allFaces.create(imgSize, (int)facesPath.size(), CV_32FC1);
        for (int i = 0; i < (int)facesPath.size(); i++) {
            Mat img = imread(facesPath[i], IMREAD_GRAYSCALE);
            resize(img, img, Size(100, 100));
            img.convertTo(img, CV_32FC1);
            img.reshape(1, imgSize).copyTo(allFaces.col(i));
        }

        reduce(allFaces, averageFace, 1, REDUCE_AVG);
        centeredFaces = allFaces.clone();
        for (int i = 0; i < centeredFaces.cols; i++) {
            subtract(centeredFaces.col(i), averageFace, centeredFaces.col(i));
        }

        Mat covariance = centeredFaces.t() * centeredFaces;
        Mat eigenValues;
        eigen(covariance, eigenValues, eigenVectors);
        eigenVectors = eigenVectors * centeredFaces.t();
        for (int i = 0; i < eigenVectors.rows; i++) {
            normalize(eigenVectors.row(i), eigenVectors.row(i));
        }
    }

    // Matriz de entrenamiento: cada columna es una cara vectorizada.
    Mat getFacesMatrix() { return allFaces; }
    // Vector promedio del conjunto de entrenamiento.
    Mat getAverage() { return averageFace; }
    // Base PCA (eigenfaces) para proyectar nuevas caras.
    Mat getEigenvectors() { return eigenVectors; }

private:
    // Dimensiones base de la imagen de rostro normalizada.
    int imgRows = 0;
    int imgSize = 0;
    // Datos usados durante entrenamiento.
    Mat allFaces;
    Mat averageFace;
    Mat centeredFaces;
    Mat eigenVectors;
};

#endif
