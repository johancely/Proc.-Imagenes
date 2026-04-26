#ifndef WRITE_TRAIN_DATA_H
#define WRITE_TRAIN_DATA_H

#include <fstream>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "MyPCA.h"

using namespace cv;
using namespace std;

class WriteTrainData {
public:
    // Constructor: proyecta las caras a PCA y persiste todo el modelo en disco.
    WriteTrainData(MyPCA& pca, vector<string>& trainFacesID, const string& dataDir) {
        numberOfFaces = pca.getFacesMatrix().cols;
        trainFacesInEigen.create(numberOfFaces, numberOfFaces, CV_32FC1);
        projectFaces(pca);
        writeTrainFacesData(trainFacesID, dataDir);
        writeMean(pca.getAverage(), dataDir);
        writeEigen(pca.getEigenvectors(), dataDir);
    }

    // Guarda cada foto como vector PCA para comparar despues.
    void projectFaces(MyPCA& pca) {
        Mat facesMatrix = pca.getFacesMatrix();
        Mat avg = pca.getAverage();
        Mat eigenVec = pca.getEigenvectors();

        for (int i = 0; i < numberOfFaces; i++) {
            Mat centered;
            subtract(facesMatrix.col(i), avg, centered);
            Mat projected = eigenVec * centered;
            projected.copyTo(trainFacesInEigen.col(i));
        }
    }

    // Guarda proyecciones PCA por cada muestra, junto con su ID.
    void writeTrainFacesData(vector<string>& trainFacesID, const string& dataDir) {
        ofstream file((dataDir + "/data/facesdata.txt").c_str(), ofstream::out | ofstream::trunc);
        if (!file) return;

        for (int i = 0; i < (int)trainFacesID.size(); i++) {
            file << trainFacesID[i] << ":";
            for (int j = 0; j < trainFacesInEigen.rows; j++) {
                file << trainFacesInEigen.at<float>(j, i) << " ";
            }
            file << "\n";
        }
    }

    // Guarda la cara promedio (media) para centrar nuevas muestras en reconocimiento.
    void writeMean(Mat avg, const string& dataDir) {
        ofstream file((dataDir + "/data/mean.txt").c_str(), ofstream::out | ofstream::trunc);
        if (!file) return;
        for (int i = 0; i < avg.rows; i++) file << avg.at<float>(i, 0) << " ";
    }

    // Guarda eigenvectores de PCA que definen el subespacio facial entrenado.
    void writeEigen(Mat eigenMat, const string& dataDir) {
        ofstream file((dataDir + "/data/eigen.txt").c_str(), ofstream::out | ofstream::trunc);
        if (!file) return;
        for (int i = 0; i < eigenMat.rows; i++) {
            for (int j = 0; j < eigenMat.cols; j++) file << eigenMat.at<float>(i, j) << " ";
            file << "\n";
        }
    }

private:
    // Proyecciones de entrenamiento en espacio PCA.
    Mat trainFacesInEigen;
    // Numero de muestras entrenadas.
    int numberOfFaces = 0;
};

#endif
