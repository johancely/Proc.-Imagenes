cmake_minimum_required(VERSION 3.10)
project(taller_corte3_2)

set(CMAKE_CXX_STANDARD 17)

# Buscar OpenCV instalado en el sistema.
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Archivos fuente del proyecto.
add_executable(taller_corte3_2
    main.cpp
    Imagen.cpp
    Geometria.cpp
    Transformaciones.cpp
)

# Enlazar con las bibliotecas de OpenCV y Windows
target_link_libraries(taller_corte3_2 ${OpenCV_LIBS} comdlg32)
