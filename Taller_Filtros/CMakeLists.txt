cmake_minimum_required(VERSION 3.10)
project(Taller_Filtros)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Encontrar OpenCV (solamente módulos básicos para I/O y visualización en cuadrícula)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# Incluir las cabeceras de OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Crear el ejecutable
add_executable(Taller_Filtros main.cpp)

# Enlazar las librerías de OpenCV al ejecutable
target_link_libraries(Taller_Filtros ${OpenCV_LIBS})
