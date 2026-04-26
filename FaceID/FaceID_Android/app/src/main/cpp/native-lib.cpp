#include <jni.h>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <android/log.h>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "FaceDetector.h"
#include "MyPCA.h"
#include "ReadFile.h"
#include "WriteTrainData.h"

#define LOG_TAG "FaceID_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;

struct VerificationResult {
    string bestId = "None";
    double nearestDistance = numeric_limits<double>::max();
    double meanDistance = numeric_limits<double>::max();
    double reconstructionError = numeric_limits<double>::max();
    bool valid = false;
};

// Estado global del motor nativo (se carga una vez y se reutiliza entre frames).
static FaceDetector gDetector;
static bool gDetectorReady = false;
static Mat gAvgVec;
static Mat gEigenVec;
static Mat gFacesInEigen;
static vector<string> gLoadedFacesID;
static bool gModelLoaded = false;
static double gNearestThreshold = 2200.0;
static double gReconstructionThreshold = 2600.0;
static double gRelativeThreshold = 0.62;

// Convierte jstring (Java) a std::string (C++) para manejar rutas.
static string toString(JNIEnv* env, jstring value) {
    const char* chars = env->GetStringUTFChars(value, nullptr);
    string result(chars == nullptr ? "" : chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

// La camara llega como RGBA; para OpenCV de deteccion usamos BGR.
static Mat frameToBgr(jlong matAddr) {
    Mat& rgba = *(Mat*)matAddr;
    Mat bgr;
    cvtColor(rgba, bgr, COLOR_RGBA2BGR);
    return bgr;
}

// Limita un valor a un rango [min, max].
static double clampDouble(double value, double minValue, double maxValue) {
    return std::max(minValue, std::min(value, maxValue));
}

// Calcula percentil para estimar umbrales robustos sin depender de un solo valor.
static double percentile(vector<double> values, double p) {
    if (values.empty()) return 0.0;
    sort(values.begin(), values.end());
    double index = p * static_cast<double>(values.size() - 1);
    int low = static_cast<int>(floor(index));
    int high = static_cast<int>(ceil(index));
    if (low == high) return values[low];
    double alpha = index - static_cast<double>(low);
    return values[low] * (1.0 - alpha) + values[high] * alpha;
}

// Normaliza una cara a 100x100 y la vectoriza (10000x1) para PCA.
static bool faceToVector(const Mat& faceGray100, Mat& outVector) {
    if (faceGray100.empty()) return false;
    Mat normalizedFace;
    if (faceGray100.rows != 100 || faceGray100.cols != 100) {
        resize(faceGray100, normalizedFace, Size(100, 100));
    } else {
        normalizedFace = faceGray100;
    }
    Mat faceFloat;
    normalizedFace.convertTo(faceFloat, CV_32FC1);
    outVector = faceFloat.reshape(1, faceFloat.rows * faceFloat.cols).clone();
    return !outVector.empty();
}

// Proyecta una cara al espacio PCA y calcula metricas de verificacion.
static bool evaluateFaceVector(const Mat& faceVector, VerificationResult& result) {
    if (faceVector.empty() || gAvgVec.empty() || gEigenVec.empty() || gFacesInEigen.empty() || gLoadedFacesID.empty()) {
        return false;
    }

    Mat centered;
    subtract(faceVector, gAvgVec, centered);
    Mat projected = gEigenVec * centered;

    double sumDistances = 0.0;
    int validDistances = 0;
    for (int i = 0; i < gFacesInEigen.cols; i++) {
        double dist = norm(gFacesInEigen.col(i), projected, NORM_L2);
        sumDistances += dist;
        validDistances++;
        if (dist < result.nearestDistance) {
            result.nearestDistance = dist;
            result.bestId = gLoadedFacesID[i];
        }
    }
    if (validDistances > 0) {
        result.meanDistance = sumDistances / static_cast<double>(validDistances);
    }

    Mat reconstructedCentered = gEigenVec.t() * projected;
    result.reconstructionError = norm(centered, reconstructedCentered, NORM_L2);
    result.valid = true;
    return true;
}

// Selecciona muestras representativas para calibrar umbrales.
static vector<int> buildCalibrationIndices(const vector<string>& trainFacePaths) {
    // Calibramos con tomas reales y rotaciones leves (_v0, _v1, _v2):
    // es mas robusto que solo _v0, pero evita usar todas las variaciones
    // sinteticas que vuelven demasiado permisivo al verificador.
    vector<int> selected;
    selected.reserve(trainFacePaths.size());
    for (int i = 0; i < (int)trainFacePaths.size(); i++) {
        const string& path = trainFacePaths[i];
        if (path.find("_v0.") != string::npos ||
            path.find("_v1.") != string::npos ||
            path.find("_v2.") != string::npos) {
            selected.push_back(i);
        }
    }

    if (!selected.empty()) return selected;

    vector<int> all(trainFacePaths.size());
    iota(all.begin(), all.end(), 0);
    return all;
}

// Umbral de distancia euclidiana: si supera esto, se considera desconocido.
static double computeNearestThresholdFromModel(const Mat& facesInEigen, const vector<int>& calibrationIndices) {
    if (facesInEigen.empty()) return 2200.0;
    if (calibrationIndices.size() < 2) return 2200.0;

    vector<double> leaveOneOut;
    leaveOneOut.reserve(calibrationIndices.size());
    for (int idx = 0; idx < (int)calibrationIndices.size(); idx++) {
        int i = calibrationIndices[idx];
        if (i < 0 || i >= facesInEigen.cols) continue;
        double best = numeric_limits<double>::max();
        for (int jdx = 0; jdx < (int)calibrationIndices.size(); jdx++) {
            int j = calibrationIndices[jdx];
            if (j < 0 || j >= facesInEigen.cols) continue;
            if (i == j) continue;
            double dist = norm(facesInEigen.col(i), facesInEigen.col(j), NORM_L2);
            if (dist < best) best = dist;
        }
        if (best < numeric_limits<double>::max()) leaveOneOut.push_back(best);
    }

    if (leaveOneOut.empty()) return 2200.0;

    double sum = 0.0;
    for (double v : leaveOneOut) sum += v;
    double mean = sum / static_cast<double>(leaveOneOut.size());

    double variance = 0.0;
    for (double v : leaveOneOut) {
        double delta = v - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(leaveOneOut.size());
    double stdDev = sqrt(variance);

    double p95 = percentile(leaveOneOut, 0.95);
    double adaptive = max(p95 * 1.25, mean + (3.0 * stdDev));
    return clampDouble(adaptive, 500.0, 12000.0);
}

// Umbral relativo nearest/mean para evitar aceptar caras "medianamente parecidas".
static double computeRelativeThresholdFromModel(const Mat& facesInEigen, const vector<int>& calibrationIndices) {
    if (facesInEigen.empty() || calibrationIndices.size() < 2) return 0.62;

    vector<double> relativeScores;
    relativeScores.reserve(calibrationIndices.size());

    for (int idx = 0; idx < (int)calibrationIndices.size(); idx++) {
        int i = calibrationIndices[idx];
        if (i < 0 || i >= facesInEigen.cols) continue;

        double nearest = numeric_limits<double>::max();
        double sum = 0.0;
        int count = 0;

        for (int jdx = 0; jdx < (int)calibrationIndices.size(); jdx++) {
            int j = calibrationIndices[jdx];
            if (j < 0 || j >= facesInEigen.cols) continue;
            if (i == j) continue;

            double dist = norm(facesInEigen.col(i), facesInEigen.col(j), NORM_L2);
            sum += dist;
            count++;
            if (dist < nearest) nearest = dist;
        }

        if (count > 0 && nearest < numeric_limits<double>::max()) {
            double mean = sum / static_cast<double>(count);
            relativeScores.push_back(nearest / max(1.0, mean));
        }
    }

    if (relativeScores.empty()) return 0.62;

    double p95 = percentile(relativeScores, 0.95);
    double adaptive = (p95 * 1.12) + 0.02;
    return clampDouble(adaptive, 0.40, 0.86);
}

// Umbral de reconstruccion PCA: mide cuanto se parece la cara al subespacio entrenado.
static double computeReconstructionThresholdFromTrainingImages(const vector<string>& trainFacePaths, const vector<int>& calibrationIndices) {
    if (trainFacePaths.empty() || gAvgVec.empty() || gEigenVec.empty()) return 2600.0;
    if (calibrationIndices.empty()) return 2600.0;

    vector<double> errors;
    errors.reserve(calibrationIndices.size());

    for (int idx : calibrationIndices) {
        if (idx < 0 || idx >= (int)trainFacePaths.size()) continue;
        const string& path = trainFacePaths[idx];
        Mat img = imread(path, IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        resize(img, img, Size(100, 100));

        Mat vectorized;
        if (!faceToVector(img, vectorized)) continue;

        Mat centered;
        subtract(vectorized, gAvgVec, centered);
        Mat projected = gEigenVec * centered;
        Mat reconstructedCentered = gEigenVec.t() * projected;
        double err = norm(centered, reconstructedCentered, NORM_L2);
        errors.push_back(err);
    }

    if (errors.empty()) return 2600.0;

    double sum = 0.0;
    for (double v : errors) sum += v;
    double mean = sum / static_cast<double>(errors.size());

    double variance = 0.0;
    for (double v : errors) {
        double delta = v - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(errors.size());
    double stdDev = sqrt(variance);

    double p95 = percentile(errors, 0.95);
    double adaptive = max(p95 * 1.25, mean + (3.0 * stdDev));
    return clampDouble(adaptive, 1000.0, 18000.0);
}

// Guarda umbrales aprendidos para auditoria y depuracion.
static void saveModelStats(const string& appDir, double nearestThreshold, double reconstructionThreshold, double relativeThreshold) {
    ofstream out((appDir + "/data/model_stats.txt").c_str(), ofstream::out | ofstream::trunc);
    if (!out) return;
    out << nearestThreshold << "\n" << reconstructionThreshold << "\n";
    out << relativeThreshold << "\n";
}

// Data augmentation para ampliar variacion sin pedir demasiadas fotos al usuario.
static vector<Mat> createVariations(const Mat& face) {
    vector<Mat> result;
    result.push_back(face.clone());

    // Rotacion pequena para simular que la cabeza no siempre queda igual.
    for (double angle : {-8.0, 8.0}) {
        Mat rotated;
        Point2f center(face.cols / 2.0f, face.rows / 2.0f);
        Mat matrix = getRotationMatrix2D(center, angle, 1.0);
        warpAffine(face, rotated, matrix, face.size(), INTER_LINEAR, BORDER_REPLICATE);
        result.push_back(rotated);
    }

    // Escala y traslacion para aumentar datos sin pedir demasiadas fotos.
    vector<pair<double, Point2f>> transforms = {
        {0.94, Point2f(0, 0)},
        {1.06, Point2f(0, 0)},
        {1.0, Point2f(-5, 0)},
        {1.0, Point2f(5, 0)},
        {1.0, Point2f(0, -5)},
        {1.0, Point2f(0, 5)}
    };

    for (const auto& item : transforms) {
        Mat transformed;
        Point2f center(face.cols / 2.0f, face.rows / 2.0f);
        Mat matrix = getRotationMatrix2D(center, 0.0, item.first);
        matrix.at<double>(0, 2) += item.second.x;
        matrix.at<double>(1, 2) += item.second.y;
        warpAffine(face, transformed, matrix, face.size(), INTER_LINEAR, BORDER_REPLICATE);
        result.push_back(transformed);
    }

    return result;
}

// JNI: inicializa detector de rostro/ojos con cascadas Haar.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_johan_faceidpca_MainActivity_nativeInitDetector(JNIEnv* env, jobject, jstring cascadePath) {
    string facePath = toString(env, cascadePath);
    string eyePath = facePath;
    size_t pos = eyePath.find("haarcascade_frontalface_default.xml");
    if (pos != string::npos) {
        eyePath.replace(pos, string("haarcascade_frontalface_default.xml").length(), "haarcascade_eye.xml");
    }

    gDetectorReady = gDetector.init(facePath, eyePath);
    LOGI("Detector listo: %d", gDetectorReady ? 1 : 0);
    return gDetectorReady ? JNI_TRUE : JNI_FALSE;
}

// JNI: captura una muestra valida y guarda variantes para entrenamiento.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_johan_faceidpca_MainActivity_nativeCaptureFace(JNIEnv* env, jobject, jlong matAddr, jstring savePath, jint index) {
    if (!gDetectorReady) return JNI_FALSE;

    Mat face;
    Mat bgr = frameToBgr(matAddr);
    if (!gDetector.extractFace(bgr, face, true)) return JNI_FALSE;

    string saveDir = toString(env, savePath);
    vector<Mat> variations = createVariations(face);
    for (int i = 0; i < (int)variations.size(); i++) {
        string filename = saveDir + "/s" + to_string((int)index) + "_v" + to_string(i) + ".bmp";
        if (!imwrite(filename, variations[i])) {
            LOGE("No se pudo guardar %s", filename.c_str());
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

// JNI: entrena PCA con las caras capturadas y recalcula umbrales adaptativos.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_johan_faceidpca_MainActivity_nativeTrainPCA(JNIEnv* env, jobject, jstring trainListPath, jstring dataDir) {
    string listPath = toString(env, trainListPath);
    string appDir = toString(env, dataDir);

    vector<string> trainFacesPath;
    vector<string> trainFacesID;
    readList(listPath, trainFacesPath, trainFacesID);

    if (trainFacesPath.size() < 5) {
        LOGE("Entrenamiento cancelado: solo hay %d imagenes", (int)trainFacesPath.size());
        return JNI_FALSE;
    }

    MyPCA pca(trainFacesPath);
    WriteTrainData writer(pca, trainFacesID, appDir);

    gFacesInEigen = readFaces((int)trainFacesID.size(), gLoadedFacesID, appDir);
    gAvgVec = readMean(appDir);
    gEigenVec = readEigen((int)trainFacesID.size(), appDir);

    vector<int> calibrationIndices = buildCalibrationIndices(trainFacesPath);
    gNearestThreshold = computeNearestThresholdFromModel(gFacesInEigen, calibrationIndices);
    gReconstructionThreshold = computeReconstructionThresholdFromTrainingImages(trainFacesPath, calibrationIndices);
    gRelativeThreshold = computeRelativeThresholdFromModel(gFacesInEigen, calibrationIndices);
    saveModelStats(appDir, gNearestThreshold, gReconstructionThreshold, gRelativeThreshold);
    LOGI("Umbrales: nearest=%.2f recon=%.2f relative=%.3f bases=%d", gNearestThreshold, gReconstructionThreshold, gRelativeThreshold, (int)calibrationIndices.size());

    gModelLoaded = !gLoadedFacesID.empty() && !gAvgVec.empty() && !gEigenVec.empty() && !gFacesInEigen.empty();
    return gModelLoaded ? JNI_TRUE : JNI_FALSE;
}

// JNI: verifica una cara en vivo. Si falla cualquier filtro, responde "None".
extern "C" JNIEXPORT jstring JNICALL
Java_com_johan_faceidpca_MainActivity_nativeRecognizeFace(JNIEnv* env, jobject, jlong matAddr, jstring dataDir) {
    string appDir = toString(env, dataDir);
    string listPath = appDir + "/list/train_list.txt";

    vector<string> paths;
    vector<string> ids;
    readList(listPath, paths, ids);
    if (ids.empty()) return env->NewStringUTF("None");

    if (!gModelLoaded) {
        gFacesInEigen = readFaces((int)ids.size(), gLoadedFacesID, appDir);
        gAvgVec = readMean(appDir);
        gEigenVec = readEigen((int)ids.size(), appDir);
        // Recalcular siempre evita quedarse con umbrales viejos o demasiado estrictos.
        vector<int> calibrationIndices = buildCalibrationIndices(paths);
        gNearestThreshold = computeNearestThresholdFromModel(gFacesInEigen, calibrationIndices);
        gReconstructionThreshold = computeReconstructionThresholdFromTrainingImages(paths, calibrationIndices);
        gRelativeThreshold = computeRelativeThresholdFromModel(gFacesInEigen, calibrationIndices);
        saveModelStats(appDir, gNearestThreshold, gReconstructionThreshold, gRelativeThreshold);
        LOGI("Umbrales recargados: nearest=%.2f recon=%.2f relative=%.3f", gNearestThreshold, gReconstructionThreshold, gRelativeThreshold);
        gModelLoaded = !gLoadedFacesID.empty() && !gAvgVec.empty() && !gEigenVec.empty() && !gFacesInEigen.empty();
    }

    if (!gDetectorReady || !gModelLoaded) return env->NewStringUTF("None");

    Mat face;
    Mat bgr = frameToBgr(matAddr);
    // En reconocimiento no exigimos ojos porque puede fallar con lentes/luz.
    if (!gDetector.extractFace(bgr, face, false)) return env->NewStringUTF("None");

    Mat faceVector;
    if (!faceToVector(face, faceVector)) return env->NewStringUTF("None");

    VerificationResult verification;
    if (!evaluateFaceVector(faceVector, verification) || !verification.valid) {
        return env->NewStringUTF("None");
    }

    double relativeScore = verification.nearestDistance / max(1.0, verification.meanDistance);
    // Regla de decision 1: distancia absoluta contra el template del duenio.
    bool passDistanceStrict = verification.nearestDistance <= gNearestThreshold;
    // Regla de decision 2: distancia relativa contra todas las muestras entrenadas.
    bool passRelative = relativeScore <= (gRelativeThreshold + 0.02);
    // Regla de decision 3: error de reconstruccion dentro del espacio PCA del duenio.
    bool passReconstruction = verification.reconstructionError <= (gReconstructionThreshold * 1.08);

    if (!passDistanceStrict) {
        LOGI("Rechazado por distancia: nearest=%.2f mean=%.2f rel=%.3f thr=%.2f", verification.nearestDistance, verification.meanDistance, relativeScore, gNearestThreshold);
        return env->NewStringUTF("None");
    }
    if (!passRelative) {
        LOGI("Rechazado por score relativo: rel=%.3f thr=%.3f nearest=%.2f mean=%.2f", relativeScore, gRelativeThreshold + 0.02, verification.nearestDistance, verification.meanDistance);
        return env->NewStringUTF("None");
    }
    if (!passReconstruction) {
        LOGI("Rechazado por reconstruccion: recon=%.2f threshold=%.2f", verification.reconstructionError, gReconstructionThreshold * 1.08);
        return env->NewStringUTF("None");
    }

    LOGI("Aceptado: id=%s nearest=%.2f mean=%.2f rel=%.3f recon=%.2f", verification.bestId.c_str(), verification.nearestDistance, verification.meanDistance, relativeScore, verification.reconstructionError);
    return env->NewStringUTF(verification.bestId.c_str());
}
