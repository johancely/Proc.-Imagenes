#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Para armar la cuadrícula de resultados en una sola imagen
Mat buildGrid(const vector<Mat>& images, const vector<string>& titles, int gridCols, int cellW = 500, int cellH = 500) {
    int n = images.size();
    int gridRows = (n + gridCols - 1) / gridCols;
    
    // Canvas fondo gris oscuro
    Mat canvas(gridRows * cellH, gridCols * cellW, CV_8UC3, Scalar(40, 40, 40));
    
    for (int i = 0; i < n; i++) {
        int r = i / gridCols;
        int c = i % gridCols;
        
        Mat resized;
        if(images[i].channels() == 1) {
            cvtColor(images[i], resized, COLOR_GRAY2BGR);
        } else {
            images[i].copyTo(resized);
        }
        resize(resized, resized, Size(cellW, cellH));
        
        // Fondo semi-transparente para el texto
        Mat overlay;
        resized.copyTo(overlay);
        rectangle(overlay, Point(0, 0), Point(cellW, 35), Scalar(0, 0, 0), FILLED);
        addWeighted(overlay, 0.6, resized, 0.4, 0, resized);
        
        // Texto
        putText(resized, titles[i], Point(10, 22), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 255, 255), 1, LINE_AA);
        
        // Pegar en el canvas
        Rect roi(c * cellW, r * cellH, cellW, cellH);
        resized.copyTo(canvas(roi));
    }
    return canvas;
}

// Normalizar y pasar a 8 bits para poder mostrar
Mat convertToDisplayable(const Mat& src) {
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8U);
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            float val = src.at<float>(r, c);
            if (val < 0) val = -val; // Valor absoluto
            if (val > 255) val = 255; // Clamping
            dst.at<uchar>(r, c) = static_cast<uchar>(val);
        }
    }
    return dst;
}

// Para ver imágenes con valores negativos (mapea el 0 al 128)
Mat convertToDisplayableSigned(const Mat& src) {
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8U);
    float valMin = 0;
    float valMax = 0;
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            float val = src.at<float>(r, c);
            if (val < valMin) valMin = val;
            if (val > valMax) valMax = val;
        }
    }
    
    float range = max(abs(valMin), abs(valMax));
    if (range == 0) range = 1;

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            float val = src.at<float>(r, c);
            float normalized = (val / range) * 127.5f + 127.5f;
            dst.at<uchar>(r, c) = static_cast<uchar>(normalized);
        }
    }
    return dst;
}

// Convertir uchar a float para convoluciones en cascada
Mat convertTo32F(const Mat& src) {
    Mat dst(src.rows, src.cols, CV_32F);
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            dst.at<float>(r, c) = static_cast<float>(src.at<uchar>(r, c));
        }
    }
    return dst;
}

// Convolución 2D
Mat manualConvolve2D(const Mat& src32F, const vector<vector<double>>& kernel) {
    int kSize = kernel.size();
    int kHalf = kSize / 2;
    int rows = src32F.rows;
    int cols = src32F.cols;

    Mat dst = Mat::zeros(rows, cols, CV_32F);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            double sum = 0.0;

            for (int kr = -kHalf; kr <= kHalf; kr++) {
                for (int kc = -kHalf; kc <= kHalf; kc++) {
                    int ir = r + kr;
                    int ic = c + kc;

                    // Clamping para los bordes
                    if (ir < 0) ir = 0;
                    if (ir >= rows) ir = rows - 1;
                    if (ic < 0) ic = 0;
                    if (ic >= cols) ic = cols - 1;

                    double pixelVal = static_cast<double>(src32F.at<float>(ir, ic));
                    double kernelVal = kernel[kr + kHalf][kc + kHalf];
                    
                    sum += pixelVal * kernelVal;
                }
            }
            dst.at<float>(r, c) = static_cast<float>(sum);
        }
    }
    return dst;
}

// Crear el kernel de Gauss
vector<vector<double>> createGaussianKernel(int size, double sigma) {
    vector<vector<double>> kernel(size, vector<double>(size));
    int halfSize = size / 2;
    double sum = 0.0;
    for (int r = -halfSize; r <= halfSize; r++) {
        for (int c = -halfSize; c <= halfSize; c++) {
            double var = 2 * sigma * sigma;
            double value = (1 / (CV_PI * var)) * exp(-(r * r + c * c) / var);
            kernel[r + halfSize][c + halfSize] = value;
            sum += value;
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

// Kernel Laplaciano
vector<vector<double>> createLaplacianKernel() {
    return {
        { 0, -1,  0},
        {-1,  4, -1},
        { 0, -1,  0}
    };
}

// Kernels Sobel X
vector<vector<double>> createSobelXKernel() {
    return {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
}

// Kernels Sobel Y
vector<vector<double>> createSobelYKernel() {
    return {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
}

// Kernels Scharr X
vector<vector<double>> createScharrXKernel() {
    return {
        { -3, 0,  3},
        {-10, 0, 10},
        { -3, 0,  3}
    };
}

// Kernels Scharr Y
vector<vector<double>> createScharrYKernel() {
    return {
        { -3,-10, -3},
        {  0,  0,  0},
        {  3, 10,  3}
    };
}

// Calcular la magnitud del gradiente
Mat computeMagnitude(const Mat& gx32F, const Mat& gy32F) {
    Mat mag = Mat::zeros(gx32F.rows, gx32F.cols, CV_32F);
    for (int r = 0; r < gx32F.rows; r++) {
        for (int c = 0; c < gx32F.cols; c++) {
            float x = gx32F.at<float>(r, c);
            float y = gy32F.at<float>(r, c);
            mag.at<float>(r, c) = sqrt(x*x + y*y);
        }
    }
    return mag;
}

// Filtro de Mediana Manual (para reducir ruido)
Mat manualMedianFilter(const Mat& src, int kSize) {
    Mat dst = src.clone();
    int edge = kSize / 2;
    vector<uchar> window(kSize * kSize);

    for (int r = edge; r < src.rows - edge; r++) {
        for (int c = edge; c < src.cols - edge; c++) {
            int k = 0;
            for (int kr = -edge; kr <= edge; kr++) {
                for (int kc = -edge; kc <= edge; kc++) {
                    window[k++] = src.at<uchar>(r + kr, c + kc);
                }
            }
            // Ordenamiento manual simple
            sort(window.begin(), window.end());
            dst.at<uchar>(r, c) = window[kSize * kSize / 2];
        }
    }
    return dst;
}

// Aplicar umbral simple
Mat thresholdImage(const Mat& src32F, float thresholdValue) {
    Mat dst = Mat::zeros(src32F.rows, src32F.cols, CV_8U);
    for (int r = 0; r < src32F.rows; r++) {
        for (int c = 0; c < src32F.cols; c++) {
            if (src32F.at<float>(r, c) >= thresholdValue) {
                dst.at<uchar>(r, c) = 255;
            }
        }
    }
    return dst;
}

// Encontrar cruces por cero
Mat findZeroCrossings(const Mat& laplacianImg, float threshold = 1.0f) {

    Mat output = Mat::zeros(laplacianImg.rows, laplacianImg.cols, CV_8U);
    for (int r = 1; r < laplacianImg.rows - 1; ++r) {
        for (int c = 1; c < laplacianImg.cols - 1; ++c) {
            float dx1 = laplacianImg.at<float>(r, c-1);
            float dx2 = laplacianImg.at<float>(r, c+1);
            float dy1 = laplacianImg.at<float>(r-1, c);
            float dy2 = laplacianImg.at<float>(r+1, c);
            
            float d11 = laplacianImg.at<float>(r-1, c-1);
            float d12 = laplacianImg.at<float>(r+1, c+1);
            float d21 = laplacianImg.at<float>(r-1, c+1);
            float d22 = laplacianImg.at<float>(r+1, c-1);

            bool cross = false;

            // Revisar cambios de signo entre vecinos
            if (dx1 * dx2 < 0 && abs(dx1 - dx2) > threshold) cross = true;
            else if (dy1 * dy2 < 0 && abs(dy1 - dy2) > threshold) cross = true;
            else if (d11 * d12 < 0 && abs(d11 - d12) > threshold) cross = true;
            else if (d21 * d22 < 0 && abs(d21 - d22) > threshold) cross = true;
            
            if (cross) {
                output.at<uchar>(r, c) = 255;
            }
        }
    }
    return output;
}

// Calcular fase (dirección) del gradiente
Mat computePhase(const Mat& gx32F, const Mat& gy32F) {
    Mat phase = Mat::zeros(gx32F.rows, gx32F.cols, CV_32F);
    for (int r = 0; r < gx32F.rows; r++) {
        for (int c = 0; c < gx32F.cols; c++) {
            float x = gx32F.at<float>(r, c);
            float y = gy32F.at<float>(r, c);
            // atan2 retorna en radianes [-pi, pi], convertimos a grados
            float angle = atan2(y, x) * 180.0f / CV_PI;
            if (angle < 0) angle += 180.0f; // Queremos ángulos de 0 a 180
            phase.at<float>(r, c) = angle;
        }
    }
    return phase;
}

// Non-maximum suppression para Canny
Mat nonMaximumSuppression(const Mat& mag, const Mat& phase) {
    Mat nms = Mat::zeros(mag.rows, mag.cols, CV_32F);
    for (int r = 1; r < mag.rows - 1; r++) {
        for (int c = 1; c < mag.cols - 1; c++) {
            float angle = phase.at<float>(r, c);
            float q = 255.0f, p = 255.0f;

            // angle 0
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                q = mag.at<float>(r, c+1);
                p = mag.at<float>(r, c-1);
            }
            // angle 45
            else if (angle >= 22.5 && angle < 67.5) {
                q = mag.at<float>(r+1, c-1);
                p = mag.at<float>(r-1, c+1);
            }
            // angle 90
            else if (angle >= 67.5 && angle < 112.5) {
                q = mag.at<float>(r+1, c);
                p = mag.at<float>(r-1, c);
            }
            // angle 135
            else if (angle >= 112.5 && angle < 157.5) {
                q = mag.at<float>(r-1, c-1);
                p = mag.at<float>(r+1, c+1);
            }

            float v = mag.at<float>(r, c);
            if (v >= q && v >= p) {
                nms.at<float>(r, c) = v;
            } else {
                nms.at<float>(r, c) = 0.0f;
            }
        }
    }
    return nms;
}

// Histéresis para Canny
Mat hysteria(const Mat& nms, float lowThresh, float highThresh) {
    Mat res = Mat::zeros(nms.rows, nms.cols, CV_8U);
    // 1. Doble umbral
    for (int r = 0; r < nms.rows; r++) {
        for (int c = 0; c < nms.cols; c++) {
            float v = nms.at<float>(r, c);
            if (v >= highThresh) {
                res.at<uchar>(r, c) = 255;
            } else if (v >= lowThresh) {
                res.at<uchar>(r, c) = 50; // Weak edge
            } else {
                res.at<uchar>(r, c) = 0;
            }
        }
    }

    // 2. Tracking: si un weake edge (50) está conectado a un strong (255), se vuelve 255
    bool changed = true;
    while (changed) {
        changed = false;
        for (int r = 1; r < nms.rows - 1; r++) {
            for (int c = 1; c < nms.cols - 1; c++) {
                if (res.at<uchar>(r, c) == 50) {
                    // Revisar vecinos 8-conexos
                    if (res.at<uchar>(r-1, c-1) == 255 || res.at<uchar>(r-1, c) == 255 || res.at<uchar>(r-1, c+1) == 255 ||
                        res.at<uchar>(r, c-1) == 255   ||                                  res.at<uchar>(r, c+1) == 255 ||
                        res.at<uchar>(r+1, c-1) == 255 || res.at<uchar>(r+1, c) == 255 || res.at<uchar>(r+1, c+1) == 255) {
                        res.at<uchar>(r, c) = 255;
                        changed = true;
                    }
                }
            }
        }
    }

    // Suprimir los weak restantes
    for (int r = 0; r < res.rows; r++) {
        for (int c = 0; c < res.cols; c++) {
            if (res.at<uchar>(r, c) == 50) {
                res.at<uchar>(r, c) = 0;
            }
        }
    }

    return res;
}

int main() {
    cout << "Iniciando aplicacion de filtros en tiempo real..." << endl;

    // Abrir la camara (indice 0 por defecto)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: No se pudo abrir la camara." << endl;
        return -1;
    }

    // Configurar ventana redimensionable
    namedWindow("Taller Filtros - Tiempo Real", WINDOW_NORMAL);

    Mat frame, gray, gray32F;
    const int processingSize = 300; // Tamano reducido para mantener fluidez con filtros manuales

    cout << "Aplicando filtros... Presiona 'q' o 'Esc' para salir." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Redimensionar para que la convolucion manual sea mas rapida
        Mat frameSmall;
        resize(frame, frameSmall, Size(processingSize, processingSize));

        // Convertir a escala de grises
        cvtColor(frameSmall, gray, COLOR_BGR2GRAY);
        
        // --- Reducción de Ruido ---
        // Aplicar Filtro de Mediana antes de los demás procesos
        Mat grayDenoisy = manualMedianFilter(gray, 3); 
        gray32F = convertTo32F(grayDenoisy);

        // --- Aplicacion de Filtros ---

        // 1. Gaussiano
        auto kernelGauss = createGaussianKernel(5, 1.4);
        Mat blur32F = manualConvolve2D(gray32F, kernelGauss);
        Mat blur8U = convertToDisplayable(blur32F);

        // 2. Laplaciano
        auto kernelLaplace = createLaplacianKernel();
        Mat laplace32F = manualConvolve2D(gray32F, kernelLaplace);
        Mat laplace8U = convertToDisplayableSigned(laplace32F);

        // 3. LoG (Laplacian of Gaussian)
        Mat log32F = manualConvolve2D(blur32F, kernelLaplace);
        Mat log8U = convertToDisplayableSigned(log32F);

        // 4. Zero Crossings
        Mat zerocross = findZeroCrossings(log32F, 10.0f);

        // 5. Sobel
        auto kernelSobelX = createSobelXKernel();
        auto kernelSobelY = createSobelYKernel();
        Mat sobelX32F = manualConvolve2D(blur32F, kernelSobelX);
        Mat sobelY32F = manualConvolve2D(blur32F, kernelSobelY);
        Mat sobelMag = computeMagnitude(sobelX32F, sobelY32F);
        Mat sobel8U = convertToDisplayable(sobelMag);

        // 6. Scharr
        auto kernelScharrX = createScharrXKernel();
        auto kernelScharrY = createScharrYKernel();
        Mat scharrX32F = manualConvolve2D(blur32F, kernelScharrX);
        Mat scharrY32F = manualConvolve2D(blur32F, kernelScharrY);
        Mat scharrMag = computeMagnitude(scharrX32F, scharrY32F);
        Mat scharr8U = convertToDisplayable(scharrMag);

        // 7. Umbral Sobel
        Mat linesSobel = thresholdImage(sobelMag, 80.0f);

        // 8. Canny
        Mat phase = computePhase(sobelX32F, sobelY32F);
        Mat nms = nonMaximumSuppression(sobelMag, phase);
        Mat canny = hysteria(nms, 20.0f, 60.0f);

        // lista de resultados
        vector<Mat> results = {frameSmall, grayDenoisy, blur8U, log8U, zerocross, sobel8U, scharr8U, linesSobel, canny};
        vector<string> titles = {
            "Original", "Sin Ruido (Mediana)", "Gauss", "LoG", "Zero Crossings", 
            "Sobel Mag", "Scharr Mag", "Sobel Threshold", "Canny"
        };

        // Crear cuadricula (3x3 en este caso)
        Mat grid = buildGrid(results, titles, 3, 300, 300);

        // Mostrar
        imshow("Taller Filtros - Tiempo Real", grid);

        // Salir con 'q' o Esc
        char c = (char)waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
