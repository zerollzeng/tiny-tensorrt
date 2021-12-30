#!/bin/bash
set -x

# TRT 8.2.1
image="nvcr.io/nvidia/tensorrt:21.12-py3"

test_command="rm -rf build && mkdir build && cd build && cmake .. && make && \
./tinyexec --onnx /usr/src/tensorrt/data/resnet50/ResNet50.onnx && \
./tinyexec --onnx /usr/src/tensorrt/data/resnet50/ResNet50.onnx --int8 --fp16 && \"

cd ..
docker run --rm --gpus all -v `pwd`:/tiny-tensorrt -w /tiny-tensorrt ${image} /bin/bash -c "${test_command}"