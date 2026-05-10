# Taller Corte 3 - Procesamiento de Imagenes

## Integrantes
- Johan Cely
- Jonathan Florez
- Harold Burbano

## Estructura del proyecto

- `main.cpp`: interfaz principal con `cvui`, flujo de herramientas y manejo de eventos de mouse.
- `Imagen.h` / `Imagen.cpp`: carga, guardado y operaciones basicas de pixeles.
- `Geometria.h` / `Geometria.cpp`: algoritmos geometricos manuales (linea, poligonos, area, circulo y elipse).
- `Transformaciones.h` / `Transformaciones.cpp`: traslacion, rotacion, escalado y operaciones ROI.
- `Punto.h`: estructura de datos para coordenadas `(x, y)`.
- `cvui.h`: libreria de interfaz grafica en un solo archivo.
- `entrada.jpg`: imagen de prueba inicial (opcional).
- `CMakeLists.txt`: configuracion de compilacion con CMake.

## Funcionalidades

- Cargar, guardar y resetear imagenes.
- Dibujo manual de linea (Bresenham), circulo (punto medio) y elipse (parametrica).
- Dibujo de poligonos, relleno de poligonos y calculo de area (Shoelace).
- Transformaciones globales: traslacion, rotacion y escalado.
- Operaciones ROI: recorte y traslacion de region poligonal.

## Compilacion

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Ejecucion en PC

```powershell
.\build\Release\taller_corte3_2.exe
```
