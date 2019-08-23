cmake_minimum_required(VERSION 3.0)

project(testyolov3)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv not found")
endif (NOT OpenCV_FOUND)

link_directories(/usr/lib/x86_64-linux-gnu)

find_library(CUDART cudart HINTS /usr/local/cuda/targets/x86_64-linux/lib/)

file(GLOB yolov3_source
     *.cpp
     )
add_executable(testyolov3 ${yolov3_source})
target_compile_options(testyolov3 PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testyolov3 tinytrt)
target_link_libraries(testyolov3 nvinfer)
target_link_libraries(testyolov3 nvinfer_plugin)
target_link_libraries(testyolov3 nvparsers)
target_link_libraries(testyolov3 nvcaffe_parser)
target_link_libraries(testyolov3 ${CUDART})
target_link_libraries(testyolov3 ${OpenCV_LIBRARIES})