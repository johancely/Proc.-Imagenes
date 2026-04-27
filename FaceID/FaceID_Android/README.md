# FaceID Android (PCA)

## Integrantes
- Johan Cely
- Jonathan Florez
- Harold Burbano

## Estructura Del Proyecto
```text
FaceID/FaceID_Android/
├── app/
│   ├── src/main/java/com/johan/faceidpca/MainActivity.kt
│   ├── src/main/cpp/native-lib.cpp
│   ├── src/main/cpp/FaceDetector.h
│   ├── src/main/cpp/MyPCA.h
│   ├── src/main/cpp/ReadFile.h
│   ├── src/main/cpp/WriteTrainData.h
│   ├── src/main/assets/haarcascade_frontalface_default.xml
│   ├── src/main/assets/haarcascade_eye.xml
│   └── src/main/res/layout/activity_main.xml
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

## Flujo Tecnico
1. `MainActivity` inicializa camara, interfaz y puente JNI con el motor nativo.
2. `nativeInitDetector` carga los clasificadores Haar de rostro y ojos.
3. `nativeCaptureFace` detecta rostro valido, normaliza a `100x100` y genera variaciones controladas.
4. `nativeTrainPCA` entrena el modelo Eigenfaces, guarda proyecciones y calcula umbrales adaptativos.
5. `nativeRecognizeFace` procesa cada frame en vivo, proyecta al espacio PCA y ejecuta verificacion 1:1.
6. La identidad se acepta solo si pasa cuatro filtros: distancia minima, distancia al centroide, score relativo y reconstruccion.
7. En Android se aplica consenso temporal entre frames para evitar aceptaciones espurias.

## Implementacion Tecnica
- Deteccion: Haar Cascade de OpenCV para ejecucion local en tiempo real sobre Android.
- Preprocesamiento: escala de grises, ecualizacion de histograma, recorte de rostro y normalizacion a `100x100`.
- Representacion: cada rostro se convierte a un vector de `10,000` caracteristicas.
- Entrenamiento PCA: calculo de cara promedio, eigenvectores (eigenfaces) y proyecciones en subespacio.
- Calibracion 1:1: umbrales adaptativos para distancia, score relativo, reconstruccion y centroide de identidad.
- Verificacion robusta: comparacion prioritaria con referencias base (`v0-v2`) y validacion por consenso temporal.

## Cambios De Mejora (Fix 1:1)
- Se reforzo la verificacion para reducir falsos positivos frente a rostros no registrados.
- Se agrego filtro por distancia al centroide de identidad en el espacio PCA.
- Se endurecio la decision en tiempo real con reglas combinadas y mayor estabilidad entre frames.
