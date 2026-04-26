cmake_minimum_required(VERSION 3.22.1)
project("faceid_pca")

# Usamos OpenCV estatico para que la libreria JNI propia sea autocontenida.
set(OpenCV_STATIC ON)
set(OpenCV_DIR "C:/Users/Johan/OpenCV-android-sdk/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED)

# Libreria JNI propia: aqui vive la deteccion, PCA y reconocimiento.
add_library(
        faceid_pca
        SHARED
        native-lib.cpp
)

target_link_libraries(
        faceid_pca
        ${OpenCV_LIBS}
        log
)
target_link_options(faceid_pca PRIVATE "-Wl,-z,max-page-size=16384")
