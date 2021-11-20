message(STATUS "
  =============> USAGE <===============
  cmake -DGPU_ARCH=xx -DBUILD_PYTHON=ON/OFF -DBUILD_TEST=ON/OFF ..
  =====================================
")

cmake_minimum_required(VERSION 3.0)
project(tinytrt VERSION 8.0.1)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_INSTALL_LIBDIR /usr/lib CACHE PATH "Install dir for shared libraries")
set(CMAKE_INSTALL_INCLUDEDIR /usr/include CACHE PATH "Install dir for headers")
set(CMAKE_INSTALL_LIBDIR /usr/lib CACHE PATH "Install dir for shared libraries")
set(CMAKE_INSTALL_BINDIR /usr/bin CACHE PATH "Install dir for binary")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" OFF)
option(BUILD_TEST "compile test" ON)

find_package(CUDA REQUIRED)

include(cmake/CUDA_utils.cmake)
set(GPU_ARCH "" CACHE STRING "Description")
if(GPU_ARCH)
  set(CUDA_targeted_archs ${GPU_ARCH})
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
else()
  # Discover what architectures does nvcc support
  message(WARNING "
  =============> SM VERSION SETTING <===============
  It would be better that you specify with your device's sm version.
  you will get faster compilation speed, usage:
  cmake -DGPU_ARCH=53 ..

  For more information about compute capability, please refer to 
  https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  ")
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
endif()

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

include_directories(spdlog)
include_directories(pybind11/include)
include_directories(./)
include_directories(./plugin)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

message(STATUS "
Generated gencode flags: ${CUDA_gencode_flags} 
BUILD_PYTHON : ${BUILD_PYTHON} 
BUILD_TEST : ${BUILD_TEST} 
")


file(GLOB_RECURSE trt_source
     Trt.cpp
     cnpy.cpp
     Int8Calibrator.cpp
     plugin/*.cu
     plugin/*.cpp
     )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(tinytrt PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(tinytrt PROPERTIES PUBLIC_HEADER Trt.h)
install(TARGETS tinytrt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BUILD_PYTHON)
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
  target_link_libraries(pytrt PRIVATE ${ZLIB_LIBRARIES})
endif()

link_directories(lib/)
## custom test
if(BUILD_TEST)
  file(GLOB test_source
      test/test.cpp
      )
  add_executable(tinyexec ${test_source})
  target_compile_options(tinyexec PUBLIC -std=c++11 -Wall -Wfloat-conversion)
  target_link_libraries(tinyexec tinytrt)
  target_link_libraries(tinyexec nvinfer)
  target_link_libraries(tinyexec nvinfer_plugin)
  target_link_libraries(tinyexec nvparsers)
  target_link_libraries(tinyexec nvonnxparser)
  target_link_libraries(tinyexec ${ZLIB_LIBRARIES})
  install(TARGETS tinyexec DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

## uninstall with xargs rm < install_manifest.txt
