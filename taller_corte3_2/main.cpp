#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

#include "Punto.h"
#include "Imagen.h"
#include "Geometria.h"
#include "Transformaciones.h"

using namespace std;

// --- Funciones para abrir/guardar archivos
string abrirArchivo() {
    OPENFILENAMEA ofn;
    CHAR szFile[260] = {0};
    ZeroMemory(&ofn, sizeof(OPENFILENAMEA));
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "Imagenes\0*.jpg;*.png;*.bmp\0Todos\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn) == TRUE) {
        return string(ofn.lpstrFile);
    }
    return "";
}

string guardarArchivo() {
    OPENFILENAMEA ofn;
    CHAR szFile[260] = {0};
    ZeroMemory(&ofn, sizeof(OPENFILENAMEA));
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "Imagen JPG\0*.jpg\0Imagen PNG\0*.png\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = "jpg";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;

    if (GetSaveFileNameA(&ofn) == TRUE) {
        return string(ofn.lpstrFile);
    }
    return "";
}

// Estados de la interfaz
enum Estado {
    NADA,
    DIBUJANDO_LINEA,
    DIBUJANDO_POLIGONO,
    RELLENANDO_POLIGONO,
    CALCULANDO_AREA,
    DIBUJANDO_CIRCULO,
    DIBUJANDO_ELIPSE,
    RECORTANDO_ROI,
    TRASLADANDO_ROI
};

// Función utilitaria para adaptar la imagen a la ventana usando NUESTRO algoritmo manual (escalarImagen).
// IMPORTANTE: Aquí garantizamos que no se usa cv::resize automático.
void adaptarImagen(cv::Mat& img) {
    if (img.cols > 880 || img.rows > 720) {
        double scale = min(880.0 / img.cols, 720.0 / img.rows);
        img = escalarImagen(img, scale, scale); // Función implementada manualmente
    }
}

int main() {
    // Configuración de la ventana principal amplia
    const string WINDOW_NAME = "Taller Corte 3 - Procesamiento de Imagenes";
    cv::Mat frame = cv::Mat(800, 1280, CV_8UC3);
    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);

    cv::Mat imagenOriginal;
    cv::Mat imagenActual;
    cv::Mat lienzo; // Usado para mostrar temporalmente la guía de clics

    Estado estado = NADA;
    vector<Punto> clics;
    
    int dx = 0, dy = 0;
    double sx = 1.0, sy = 1.0;
    double angulo = 0.0;
    int radio = 50;
    int radioX = 50, radioY = 30;

    string mensaje = "Bienvenido. Carga una imagen para empezar.";

    // Cargar imagen de prueba inicial si existe
    cv::Mat testInicial = cv::imread("entrada.jpg");
    if (!testInicial.empty()) {
        imagenOriginal = testInicial;
        adaptarImagen(imagenOriginal);
        imagenActual = imagenOriginal.clone();
        mensaje = "Imagen de prueba cargada.";
    }

    while (true) {
        // Fondo general (Gris oscuro moderno)
        frame = cv::Scalar(45, 45, 48); 

        // ==========================================
        // PANEL LATERAL IZQUIERDO
        // ==========================================
        int pX = 10;
        int pW = 340;
        int y = 10;

        // Fondo del panel lateral
        cvui::rect(frame, pX, y, pW, 780, 0x1E1E1E, 0x1E1E1E);

        // Título del panel
        cvui::text(frame, pX + 20, y + 20, "PANEL DE CONTROL", 0.5, 0xFFFFFF);
        y += 50;

        // ------------------------------------------
        // 1. SECCIÓN ARCHIVO
        // ------------------------------------------
        cvui::rect(frame, pX + 10, y, pW - 20, 80, 0x2D2D30, 0x2D2D30);
        cvui::text(frame, pX + 20, y + 10, "1. ARCHIVO", 0.4, 0x00AACC);
        
        if (cvui::button(frame, pX + 20, y + 35, 95, 30, "Cargar")) {
            string ruta = abrirArchivo();
            if (!ruta.empty()) {
                cv::Mat tmp = cargarImagen(ruta);
                if (!tmp.empty()) {
                    imagenOriginal = tmp;
                    adaptarImagen(imagenOriginal);
                    imagenActual = imagenOriginal.clone();
                    clics.clear();
                    estado = NADA;
                    mensaje = "Exito: Imagen cargada correctamente.";
                }
            }
        }
        if (cvui::button(frame, pX + 125, y + 35, 90, 30, "Guardar")) {
            if (!imagenActual.empty()) {
                string ruta = guardarArchivo();
                if (!ruta.empty()) {
                    guardarImagen(ruta, imagenActual);
                    mensaje = "Exito: Imagen guardada correctamente.";
                }
            }
        }
        if (cvui::button(frame, pX + 225, y + 35, 90, 30, "Resetear")) {
            if (!imagenOriginal.empty()) {
                imagenActual = imagenOriginal.clone();
                clics.clear();
                estado = NADA;
                mensaje = "Imagen restaurada a su estado original.";
            }
        }
        y += 95;

        // ------------------------------------------
        // 2. SECCIÓN DIBUJO GEOMÉTRICO
        // ------------------------------------------
        cvui::rect(frame, pX + 10, y, pW - 20, 150, 0x2D2D30, 0x2D2D30);
        cvui::text(frame, pX + 20, y + 10, "2. DIBUJO GEOMETRICO", 0.4, 0x00AACC);

        if (cvui::button(frame, pX + 20, y + 35, 95, 30, "Linea (2pt)")) { estado = DIBUJANDO_LINEA; clics.clear(); mensaje = "LINEA: Haz clic en 2 puntos distintos en la imagen."; }
        if (cvui::button(frame, pX + 125, y + 35, 90, 30, "Circulo (1pt)")) { estado = DIBUJANDO_CIRCULO; clics.clear(); mensaje = "CIRCULO: Define radio y da clic para el centro."; }
        if (cvui::button(frame, pX + 225, y + 35, 90, 30, "Elipse (1pt)")) { estado = DIBUJANDO_ELIPSE; clics.clear(); mensaje = "ELIPSE: Define Rx/Ry y da clic para el centro."; }

        cvui::text(frame, pX + 20, y + 78, "Radio:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 70, y + 70, 240, &radio, 1, 300);

        cvui::text(frame, pX + 20, y + 118, "Rx:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 50, y + 110, 100, &radioX, 1, 300);
        cvui::text(frame, pX + 170, y + 118, "Ry:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 200, y + 110, 100, &radioY, 1, 300);
        y += 165;

        // ------------------------------------------
        // 3. SECCIÓN POLÍGONOS
        // ------------------------------------------
        cvui::rect(frame, pX + 10, y, pW - 20, 80, 0x2D2D30, 0x2D2D30);
        cvui::text(frame, pX + 20, y + 10, "3. POLIGONOS", 0.4, 0x00AACC);

        if (cvui::button(frame, pX + 20, y + 35, 95, 30, "Contorno")) { estado = DIBUJANDO_POLIGONO; clics.clear(); mensaje = "POLIGONO: Da clics para formar la figura, luego APLICAR."; }
        if (cvui::button(frame, pX + 125, y + 35, 90, 30, "Rellenar")) { estado = RELLENANDO_POLIGONO; clics.clear(); mensaje = "RELLENAR: Define poligono y presiona APLICAR."; }
        if (cvui::button(frame, pX + 225, y + 35, 90, 30, "Area")) { estado = CALCULANDO_AREA; clics.clear(); mensaje = "AREA: Define poligono y presiona APLICAR."; }
        y += 95;

        // ------------------------------------------
        // 4. SECCIÓN TRANSFORMACIONES GLOBALES
        // ------------------------------------------
        cvui::rect(frame, pX + 10, y, pW - 20, 140, 0x2D2D30, 0x2D2D30);
        cvui::text(frame, pX + 20, y + 10, "4. TRANSFORMACIONES GLOBALES", 0.4, 0x00AACC);

        cvui::text(frame, pX + 20, y + 38, "dx:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 45, y + 30, 85, &dx, -300, 300);
        cvui::text(frame, pX + 140, y + 38, "dy:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 165, y + 30, 85, &dy, -300, 300);
        if (cvui::button(frame, pX + 260, y + 35, 55, 30, "Mover")) {
            if (!imagenActual.empty()) { imagenActual = trasladarImagen(imagenActual, dx, dy); mensaje = "Exito: Imagen trasladada."; }
        }

        cvui::text(frame, pX + 20, y + 73, "Ang:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 55, y + 65, 195, &angulo, -180.0, 180.0);
        if (cvui::button(frame, pX + 260, y + 70, 55, 30, "Rotar")) {
            if (!imagenActual.empty()) { imagenActual = rotarImagen(imagenActual, angulo); mensaje = "Exito: Imagen rotada."; }
        }

        cvui::text(frame, pX + 20, y + 108, "sx:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 45, y + 100, 85, &sx, (double)0.1, (double)3.0);
        cvui::text(frame, pX + 140, y + 108, "sy:", 0.4, 0xDDDDDD);
        cvui::trackbar(frame, pX + 165, y + 100, 85, &sy, (double)0.1, (double)3.0);
        if (cvui::button(frame, pX + 260, y + 105, 55, 30, "Escal")) {
            if (!imagenActual.empty()) { imagenActual = escalarImagen(imagenActual, sx, sy); mensaje = "Exito: Imagen escalada."; }
        }
        y += 155;

        // ------------------------------------------
        // 5. SECCIÓN ROI (Regiones de interés)
        // ------------------------------------------
        cvui::rect(frame, pX + 10, y, pW - 20, 80, 0x2D2D30, 0x2D2D30);
        cvui::text(frame, pX + 20, y + 10, "5. REGIONES DE INTERES (ROI)", 0.4, 0x00AACC);
        if (cvui::button(frame, pX + 20, y + 35, 140, 30, "Recortar ROI")) { estado = RECORTANDO_ROI; clics.clear(); mensaje = "ROI: Dibuja un poligono y presiona APLICAR."; }
        if (cvui::button(frame, pX + 175, y + 35, 140, 30, "Trasladar ROI")) { estado = TRASLADANDO_ROI; clics.clear(); mensaje = "ROI: Dibuja poligono, ajusta dx/dy y presiona APLICAR."; }
        y += 95;

        // ------------------------------------------
        // 6. SECCIÓN ESTADO Y APLICAR
        // ------------------------------------------
        if (estado == NADA) {
            cvui::rect(frame, pX + 10, y, pW - 20, 45, 0x111111, 0x111111);
            cvui::text(frame, pX + 20, y + 25, "ESTADO: Selecciona una herramienta para iniciar", 0.4, 0x666666);
        } else {
            // Se resalta la zona cuando hay una herramienta activa
            cvui::rect(frame, pX + 10, y, pW - 20, 45, 0x331111, 0x331111);
            
            if (estado == DIBUJANDO_POLIGONO || estado == RELLENANDO_POLIGONO || estado == CALCULANDO_AREA || estado == RECORTANDO_ROI || estado == TRASLADANDO_ROI) {
                // Botón destacado para aplicar el polígono configurado
                if (cvui::button(frame, pX + 15, y + 5, pW - 30, 35, ">> APLICAR ACCION <<")) {
                    if (clics.size() >= 3) {
                        if (estado == DIBUJANDO_POLIGONO) {
                            dibujarPoligono(imagenActual, clics, cv::Vec3b(0, 0, 255)); // Rojo
                            mensaje = "Exito: Poligono dibujado.";
                        } else if (estado == RELLENANDO_POLIGONO) {
                            rellenarPoligono(imagenActual, clics, cv::Vec3b(255, 0, 255)); // Magenta
                            mensaje = "Exito: Poligono rellenado.";
                        } else if (estado == CALCULANDO_AREA) {
                            double area = calcularAreaPoligono(clics);
                            mensaje = "Resultado: El area calculada es " + to_string(area) + " px^2";
                        } else if (estado == RECORTANDO_ROI) {
                            imagenActual = recortarROI(imagenActual, clics);
                            mensaje = "Exito: ROI recortada correctamente.";
                        } else if (estado == TRASLADANDO_ROI) {
                            imagenActual = trasladarROI(imagenActual, clics, dx, dy);
                            mensaje = "Exito: ROI trasladada (dx: " + to_string(dx) + ", dy: " + to_string(dy) + ").";
                        }
                        clics.clear();
                        estado = NADA;
                    } else {
                        mensaje = "Error: Se necesitan al menos 3 puntos para procesar el poligono.";
                    }
                }
            } else {
                // Para Línea, Círculo, Elipse damos la opción de cancelar si cambian de opinión
                if (cvui::button(frame, pX + 15, y + 5, pW - 30, 35, "CANCELAR HERRAMIENTA")) {
                    estado = NADA;
                    clics.clear();
                    mensaje = "Herramienta cancelada. Selecciona otra opcion.";
                }
            }
        }

        // ==========================================
        // ÁREA PRINCIPAL (IMAGEN Y ESTADO)
        // ==========================================
        int imgX = 370;
        int imgY = 20;
        int imgW = 880;
        int imgH = 720;
        
        // Marco de la imagen (Contenedor gris oscuro)
        cvui::rect(frame, imgX - 2, imgY - 2, imgW + 4, imgH + 4, 0x555555);
        cvui::rect(frame, imgX, imgY, imgW, imgH, 0x111111, 0x111111);

        if (!imagenActual.empty()) {
            lienzo = imagenActual.clone();

            // Dibujar feedback temporal (puntos y líneas guía sobre el lienzo)
            for (size_t i = 0; i < clics.size(); i++) {
                cv::circle(lienzo, cv::Point(clics[i].x, clics[i].y), 3, cv::Scalar(0, 0, 255), -1);
                
                // Línea guía visual. No rompe la regla ya que el algoritmo matemático es nuestro y la matriz real usa `dibujarLinea()`. 
                // Esto es solo para que el usuario sepa dónde va a quedar su figura.
                if (i > 0 && (estado == DIBUJANDO_POLIGONO || estado == RELLENANDO_POLIGONO || estado == CALCULANDO_AREA || estado == RECORTANDO_ROI || estado == TRASLADANDO_ROI)) {
                    cv::line(lienzo, cv::Point(clics[i-1].x, clics[i-1].y), cv::Point(clics[i].x, clics[i].y), cv::Scalar(0, 255, 0), 1);
                }
            }

            // Calcular posición para centrar la imagen dentro del contenedor
            int copyW = min((int)lienzo.cols, imgW);
            int copyH = min((int)lienzo.rows, imgH);
            int offsetX = imgX + (imgW - copyW) / 2;
            int offsetY = imgY + (imgH - copyH) / 2;

            // Copiar al frame
            cv::Mat imgROI = frame(cv::Rect(offsetX, offsetY, copyW, copyH));
            lienzo(cv::Rect(0, 0, copyW, copyH)).copyTo(imgROI);

            // Manejo de clics dentro del área de la imagen
            if (cvui::mouse(cvui::DOWN) && cvui::mouse().x >= offsetX && cvui::mouse().y >= offsetY && 
                cvui::mouse().x < offsetX + copyW && cvui::mouse().y < offsetY + copyH) {
                
                int x = cvui::mouse().x - offsetX;
                int y = cvui::mouse().y - offsetY;

                clics.push_back(Punto(x, y));

                // Procesamiento automático de herramientas simples (2 puntos para línea, 1 para curvas)
                if (estado == DIBUJANDO_LINEA && clics.size() == 2) {
                    dibujarLinea(imagenActual, clics[0], clics[1], cv::Vec3b(0, 255, 0)); // Implementación manual (Geometria.cpp)
                    clics.clear();
                    mensaje = "Exito: Linea dibujada.";
                    estado = NADA;
                }
                else if (estado == DIBUJANDO_CIRCULO && clics.size() == 1) {
                    dibujarCirculo(imagenActual, clics[0], radio, cv::Vec3b(255, 0, 0)); // Implementación manual (Geometria.cpp)
                    clics.clear();
                    mensaje = "Exito: Circulo dibujado.";
                    estado = NADA;
                }
                else if (estado == DIBUJANDO_ELIPSE && clics.size() == 1) {
                    dibujarElipse(imagenActual, clics[0], radioX, radioY, cv::Vec3b(0, 255, 255)); // Implementación manual (Geometria.cpp)
                    clics.clear();
                    mensaje = "Exito: Elipse dibujada.";
                    estado = NADA;
                }
            }
        } else {
            cvui::text(frame, imgX + 370, imgY + 350, "SIN IMAGEN", 0.8, 0x444444);
        }

        // ==========================================
        // BARRA INFERIOR (FEEDBACK GLOBAL)
        // ==========================================
        cvui::rect(frame, imgX, imgY + imgH + 15, imgW, 30, 0x1E1E1E, 0x1E1E1E);
        
        // Asignar color dinámico dependiendo del tipo de mensaje
        uint colorMensaje = 0xCCCCCC;
        if (mensaje.find("Error") != string::npos) colorMensaje = 0xEE4444; // Rojo para errores
        else if (mensaje.find("Exito") != string::npos || mensaje.find("Resultado") != string::npos) colorMensaje = 0x44EE44; // Verde para éxito

        cvui::text(frame, imgX + 15, imgY + imgH + 22, "ESTADO: " + mensaje, 0.45, colorMensaje);

        // Actualizar UI
        cvui::update();
        cv::imshow(WINDOW_NAME, frame);

        if (cv::waitKey(20) == 27) {
            break; // Salir con Esc
        }
    }

    return 0;
}
