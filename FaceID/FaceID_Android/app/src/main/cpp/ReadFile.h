#ifndef READ_FILE_H
#define READ_FILE_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/core.hpp"

using namespace cv;
using namespace std;

// Lee train_list.txt (formato: ID;ruta) y separa IDs y rutas.
inline void readList(string& listFilePath, vector<string>& facesPath, vector<string>& facesID) {
    ifstream file(listFilePath.c_str(), ifstream::in);
    if (!file) return;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream lines(line);
        string id;
        string path;
        getline(lines, id, ';');
        getline(lines, path);

        path.erase(remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(remove(path.begin(), path.end(), '\n'), path.end());
        if (!id.empty() && !path.empty()) {
            facesID.push_back(id);
            facesPath.push_back(path);
        }
    }
}

// Lee facesdata.txt: cada fila es una cara ya proyectada en PCA.
inline Mat readFaces(int noOfFaces, vector<string>& loadedFaceID, const string& dataDir) {
    Mat faces = Mat::zeros(noOfFaces, noOfFaces, CV_32FC1);
    ifstream file((dataDir + "/data/facesdata.txt").c_str(), ifstream::in);
    if (!file) return faces;

    loadedFaceID.clear();
    string line;
    for (int i = 0; i < noOfFaces && getline(file, line); i++) {
        stringstream lines(line);
        string id;
        getline(lines, id, ':');
        loadedFaceID.push_back(id);
        for (int j = 0; j < noOfFaces; j++) {
            string value;
            getline(lines, value, ' ');
            faces.at<float>(j, i) = (float)atof(value.c_str());
        }
    }
    return faces;
}

// Lee mean.txt: vector promedio del entrenamiento (10000x1 para 100x100).
inline Mat readMean(const string& dataDir) {
    Mat mean = Mat::zeros(10000, 1, CV_32FC1);
    ifstream file((dataDir + "/data/mean.txt").c_str(), ifstream::in);
    if (!file) return mean;

    for (int i = 0; i < mean.rows; i++) {
        string value;
        file >> value;
        mean.at<float>(i, 0) = (float)atof(value.c_str());
    }
    return mean;
}

// Lee eigen.txt: matriz de eigenvectores que define el subespacio PCA.
inline Mat readEigen(int noOfFaces, const string& dataDir) {
    Mat eigenMat = Mat::zeros(noOfFaces, 10000, CV_32FC1);
    ifstream file((dataDir + "/data/eigen.txt").c_str(), ifstream::in);
    if (!file) return eigenMat;

    for (int i = 0; i < eigenMat.rows; i++) {
        for (int j = 0; j < eigenMat.cols; j++) {
            string value;
            file >> value;
            eigenMat.at<float>(i, j) = (float)atof(value.c_str());
        }
    }
    return eigenMat;
}

#endif
