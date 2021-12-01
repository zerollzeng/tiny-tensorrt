message(STATUS "
=============> USAGE <===============
cmake -DGPU_ARCH=xx -DBUILD_PYTHON=ON/OFF -DBUILD_SAMPLE=ON/OFF -DTENSORRT_INCLUDE_PATH=/path \
-DTENSORRT_LIB_PATH=/path ..
=====================================
")

cmake_minimum_required(VERSION 3.0)
project(tinytrt VERSION 8.0.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_INSTALL_LIBDIR /usr/lib CACHE PATH "Install dir for shared libraries")
set(CMAKE_INSTALL_INCLUDEDIR /usr/include CACHE PATH "Install dir for headers")
set(CMAKE_INSTALL_LIBDIR /usr/lib CACHE PATH "Install dir for shared libraries")
set(CMAKE_INSTALL_BINDIR /usr/bin CACHE PATH "Install dir for binary")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" OFF)
option(BUILD_SAMPLE "build samples" ON)

set(GPU_ARCH "" CACHE STRING "GPU compute capability")
set(TENSORRT_LIB_PATH "" CACHE STRING "absolute path to tensorrt libraries")
set(TENSORRT_INCLUDE_PATH "" CACHE STRING "absolute path to tensorrt public header")

find_package(CUDA REQUIRED)
find_package(ZLIB REQUIRED)
# TensorRT
find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVPARSERS NAME nvparsers HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_INCLUDE_PATH})

include(cmake/CUDA_utils.cmake)

if(GPU_ARCH)
  set(CUDA_targeted_archs ${GPU_ARCH})
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
else()
  # Discover what architectures does nvcc support
  CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
  set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
  list(SORT CUDA_TARGET_ARCHS_SORTED)
  CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
  if (NOT CUDA_targeted_archs)
    message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) \
      is supported by nvcc, please set appropriate arch via -DGPU_ARCH=XX")
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
endif()

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

include_directories(third_party/spdlog/include)
include_directories(third_party/pybind11/include)
include_directories(./)
include_directories(./src)
include_directories(./plugin)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})

message(STATUS "
=============> Final Config <===============
TensorRT headers: ${TENSORRT_INCLUDE_DIR}
TensorRT library: ${LIBNVINFER}
Generated gencode flags: ${CUDA_gencode_flags} 
BUILD_PYTHON : ${BUILD_PYTHON} 
BUILD_SAMPLE : ${BUILD_SAMPLE} 

for -DGPU_ARCH, refer to 
https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
============================================
")


file(GLOB_RECURSE trt_source
     src/Trt.cpp
     src/cnpy.cpp
     src/Int8Calibrator.cpp
     plugin/*.cu
     plugin/*.cpp
     )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wno-deprecated -Wfloat-conversion)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(tinytrt PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(tinytrt PROPERTIES PUBLIC_HEADER Trt.h)
install(TARGETS tinytrt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BUILD_PYTHON)
  add_subdirectory(third_party/pybind11)
  pybind11_add_module(pytrt SHARED src/PyTrt.cpp)
  target_link_libraries(pytrt PRIVATE tinytrt)
  target_link_libraries(pytrt PRIVATE ${LIBNVINFER})
  target_link_libraries(pytrt PRIVATE ${LIBNVINFER_PLUGIN})
  target_link_libraries(pytrt PRIVATE ${LIBNVPARSERS})
  target_link_libraries(pytrt PRIVATE ${LIBNVONNXPARSER})
  target_link_libraries(pytrt PRIVATE ${ZLIB_LIBRARIES})
endif()

link_directories(lib/)
## custom test
if(BUILD_SAMPLE)
  file(GLOB tinyexec_source
      samples/tinyexec/tinyexec.cpp
      )
  add_executable(tinyexec ${tinyexec_source})
  target_compile_options(tinyexec PUBLIC -std=c++11 -Wall -Wno-deprecated -Wfloat-conversion)
  target_link_libraries(tinyexec tinytrt)
  target_link_libraries(tinyexec ${LIBNVINFER})
  target_link_libraries(tinyexec ${LIBNVINFER_PLUGIN})
  target_link_libraries(tinyexec ${LIBNVPARSERS})
  target_link_libraries(tinyexec ${LIBNVONNXPARSER})
  target_link_libraries(tinyexec ${ZLIB_LIBRARIES})
  install(TARGETS tinyexec DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

## uninstall with xargs rm < install_manifest.txt
