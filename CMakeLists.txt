cmake_minimum_required(VERSION 3.0)


project(tinytrt)

set(CMAKE_CXX_FLAGS "-std=c++11")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" ON)

# Discover what architectures does nvcc support
include(cmake/CUDA_utils.cmake)
CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
list(SORT CUDA_TARGET_ARCHS_SORTED)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
if (NOT CUDA_targeted_archs)
  message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
endif()

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
