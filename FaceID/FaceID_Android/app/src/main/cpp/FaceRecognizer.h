#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

class FaceRecognizer {
public:
    FaceRecognizer(Mat testFace, Mat avgVec, Mat eigenVec, Mat facesInEigen, vector<string>& loadedFacesID) {
        recognize(testFace, avgVec, eigenVec, facesInEigen, loadedFacesID);
    }

    // Proyecta el rostro en PCA y busca el template mas cercano.
    void recognize(Mat testFace, Mat avgVec, Mat eigenVec, Mat facesInEigen, vector<string>& loadedFacesID) {
        if (testFace.empty() || avgVec.empty() || eigenVec.empty() || facesInEigen.empty()) return;

        Mat testFloat;
        testFace.convertTo(testFloat, CV_32FC1);
        Mat testVector;
        testFloat.reshape(1, testFloat.rows * testFloat.cols).copyTo(testVector);

        Mat centered;
        subtract(testVector, avgVec, centered);
        Mat projected = eigenVec * centered;

        for (int i = 0; i < (int)loadedFacesID.size(); i++) {
            double dist = norm(facesInEigen.col(i), projected, NORM_L2);
            if (dist < bestDistance) {
                bestDistance = dist;
                bestFaceID = loadedFacesID[i];
            }
        }
    }

    string getBestFaceID() { return bestFaceID; }
    double getBestDistance() { return bestDistance; }

private:
    string bestFaceID = "None";
    double bestDistance = 1.0e18;
};

#endif
