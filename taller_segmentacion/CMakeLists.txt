cmake_minimum_required(VERSION 3.15)
project(TallerSegmentacion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})

if(MSVC)
    set(OpenCV_ARCH x64 CACHE STRING "" FORCE)
    if(NOT DEFINED OpenCV_RUNTIME)
        set(OpenCV_RUNTIME vc16 CACHE STRING "" FORCE)
    endif()
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(parte_a
    src/main_frame_diff.cpp
    src/frame_diff.cpp
)
target_link_libraries(parte_a ${OpenCV_LIBS})

add_executable(parte_b
    src/main_gmm.cpp
    src/gmm_segmenter.cpp
)
target_link_libraries(parte_b ${OpenCV_LIBS})

if(WIN32)
    file(GLOB OPENCV_RUNTIME_DLLS "${OpenCV_DIR}/x64/vc16/bin/opencv_world*.dll")
    foreach(target_name IN ITEMS parte_a parte_b)
        foreach(runtime_dll IN LISTS OPENCV_RUNTIME_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${runtime_dll}" "$<TARGET_FILE_DIR:${target_name}>"
            )
        endforeach()
    endforeach()
endif()
