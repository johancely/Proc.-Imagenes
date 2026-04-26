#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <algorithm>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace std;

class FaceDetector {
public:
    // Constructor vacio: los clasificadores se cargan en init().
    FaceDetector() = default;

    // Carga los clasificadores Haar que se copiaron desde assets.
    bool init(const string& faceCascadePath, const string& eyeCascadePath) {
        faceReady = faceCascade.load(faceCascadePath);
        eyeReady = eyeCascade.load(eyeCascadePath);
        return faceReady;
    }

    // Busca el rostro mas grande, lo pasa a gris y lo deja en 100x100.
    // requireEyes reduce falsos positivos cuando estamos reconociendo.
    bool extractFace(const Mat& frameBgr, Mat& faceGray100, bool requireEyes = false) {
        faceGray100.release();
        if (!faceReady || frameBgr.empty()) return false;

        Mat gray;
        cvtColor(frameBgr, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(60, 60));
        if (faces.empty()) return false;

        Rect bestFace = *max_element(faces.begin(), faces.end(), [](const Rect& a, const Rect& b) {
            return a.area() < b.area();
        });

        // Filtro simple de tamano para evitar detectar camisetas u objetos pequenos.
        double frameArea = static_cast<double>(gray.cols * gray.rows);
        double bestArea = static_cast<double>(bestFace.area());
        if (bestArea < frameArea * 0.06) return false;

        if (requireEyes && eyeReady) {
            // En captura si exigimos ojos para evitar entrenar basura.
            // Lo hacemos flexible para lentes/iluminacion variable, pero
            // seguimos exigiendo al menos un ojo para descartar objetos.
            Rect upperHalf(bestFace.x, bestFace.y, bestFace.width, max(1, bestFace.height * 3 / 4));
            Mat upperFace = gray(upperHalf);
            vector<Rect> eyes;
            int minEye = max(8, bestFace.width / 12);
            eyeCascade.detectMultiScale(upperFace, eyes, 1.05, 1, 0 | CASCADE_SCALE_IMAGE, Size(minEye, minEye));
            if (eyes.empty()) return false;
        }

        Mat face = gray(bestFace).clone();
        resize(face, faceGray100, Size(100, 100));
        return !faceGray100.empty();
    }

    // Dibuja un rectangulo para que el usuario vea que la app detecto cara.
    bool drawFaceBox(const Mat& frameBgr, Mat& outputBgr) {
        outputBgr = frameBgr.clone();
        if (!faceReady || frameBgr.empty()) return false;

        Mat gray;
        cvtColor(frameBgr, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(60, 60));
        if (faces.empty()) return false;

        Rect bestFace = *max_element(faces.begin(), faces.end(), [](const Rect& a, const Rect& b) {
            return a.area() < b.area();
        });
        rectangle(outputBgr, bestFace, Scalar(0, 255, 0), 3);
        return true;
    }

private:
    // Clasificador principal de rostro frontal.
    CascadeClassifier faceCascade;
    // Clasificador auxiliar de ojos para validar calidad de captura.
    CascadeClassifier eyeCascade;
    // Banderas para saber si cada clasificador cargo correctamente.
    bool faceReady = false;
    bool eyeReady = false;
};

#endif
