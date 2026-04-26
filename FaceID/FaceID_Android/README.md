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
1. MainActivity inicia la camara y llama funciones nativas JNI.
2. nativeInitDetector carga cascadas Haar para rostro y ojos.
3. nativeCaptureFace detecta, recorta y normaliza rostro; guarda muestras y variaciones.
4. nativeTrainPCA entrena PCA (eigenfaces), proyecta muestras y guarda modelo.
5. nativeRecognizeFace procesa cada frame en vivo, proyecta al espacio PCA y compara.
6. Se acepta identidad solo si pasa tres filtros: distancia, score relativo y reconstruccion.
7. En Android se aplica estabilidad temporal por varios frames para reducir falsos positivos.

## Implementacion Tecnica
- Deteccion: Haar Cascade de OpenCV por eficiencia en tiempo real en Android.
- Preprocesamiento: escala de grises + ecualizacion + recorte de rostro.
- Tamano fijo 100x100: estandariza entrada para PCA y mantiene costo computacional bajo.
- Representacion: cada rostro se vectoriza a 10,000 valores (100x100).
- Entrenamiento: PCA calcula cara promedio y eigenvectores (eigenfaces).
- Verificacion: distancia euclidiana en PCA + score relativo + error de reconstruccion.
- Robustez: data augmentation (rotacion, escala, traslacion) y consenso temporal entre frames.
