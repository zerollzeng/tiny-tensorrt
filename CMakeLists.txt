cmake_minimum_required(VERSION 3.0)


project(tinytrt)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(CUDA REQUIRED)

include_directories(plugin)
include_directories(spdlog)
include_directories(./)
# include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

file(GLOB trt_source
     *.cpp
     plugin/*.cu
     plugin/*.cpp
     )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wfloat-conversion -DONNX_ML)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(tinytrt nvinfer)
# target_link_libraries(tinytrt nvinfer_plugin)
# target_link_libraries(tinytrt nvparsers)
# target_link_libraries(tinytrt nvcaffe_parser)
# target_link_libraries(tinytrt ${CUDART})

