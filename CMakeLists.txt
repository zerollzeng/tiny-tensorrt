cmake_minimum_required(VERSION 3.0)


project(tinytrt)

set(CMAKE_CXX_FLAGS "-std=c++11")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")

find_package(CUDA REQUIRED)

include_directories(spdlog)
include_directories(./)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

file(GLOB trt_source
     Trt.cpp
     Int8EntropyCalibrator.cpp
     plugin/*.cu
     plugin/*.cpp
     )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wfloat-conversion -DONNX_ML)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)
pybind11_add_module(pytrt SHARED PyTrt.cpp)
target_link_libraries(pytrt PRIVATE tinytrt)
target_link_libraries(pytrt PRIVATE nvinfer)
target_link_libraries(pytrt PRIVATE nvinfer_plugin)
target_link_libraries(pytrt PRIVATE nvparsers)
target_link_libraries(pytrt PRIVATE nvonnxparser)
target_link_libraries(pytrt PRIVATE nvcaffe_parser)
