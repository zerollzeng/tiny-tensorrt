cmake_minimum_required(VERSION 3.0)


project(tinytrt)

set(CMAKE_CXX_FLAGS "-std=c++11")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" ON)
option(BUILD_TEST "compile test" OFF)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Discover what architectures does nvcc support
include(cmake/CUDA_utils.cmake)
set(CUDA_targeted_archs "60;61;70;75")

CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
message(STATUS "Generated gencode flags: ${CUDA_gencode_flags}")

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

include_directories(spdlog)
include_directories(pybind11/include)
include_directories(./)
include_directories(./plugin)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")


file(GLOB_RECURSE trt_source
     Trt.cpp
     Int8EntropyCalibrator.cpp
     plugin/*.cu
     plugin/*.cpp
     )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_PYTHON)
  message(STATUS "Build python")
  # set(Python3_ROOT_DIR /root/miniconda3/bin)
  # find_package(Python3 REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  add_subdirectory(pybind11)
  pybind11_add_module(pytrt SHARED PyTrt.cpp)
  target_link_libraries(pytrt PRIVATE tinytrt)
  target_link_libraries(pytrt PRIVATE nvinfer)
  target_link_libraries(pytrt PRIVATE nvinfer_plugin)
  target_link_libraries(pytrt PRIVATE nvparsers)
  target_link_libraries(pytrt PRIVATE nvonnxparser)
  target_link_libraries(pytrt PRIVATE nvcaffe_parser)
endif()

## custom test
if(BUILD_TEST)
message(STATUS "Build test")
  file(GLOB test_source
      test/test.cpp
      )
  add_executable(unit_test ${test_source})
  target_compile_options(unit_test PUBLIC -std=c++11 -Wall -Wfloat-conversion)
  target_link_libraries(unit_test tinytrt)
  target_link_libraries(unit_test nvinfer)
  target_link_libraries(unit_test nvinfer_plugin)
  target_link_libraries(unit_test nvparsers)
  target_link_libraries(unit_test nvonnxparser)
  target_link_libraries(unit_test nvcaffe_parser)
  target_link_libraries(unit_test ${CUDART})
endif()
