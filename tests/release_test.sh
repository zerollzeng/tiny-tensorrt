#!/bin/bash
set -x

image_list=(
    # TRT 8.0.3
    "nvcr.io/nvidia/tensorrt:21.09-py3"
    # TRT 7.2.3.4
    "nvcr.io/nvidia/tensorrt:21.06-py3"
    # TRT 7.1.3
    "nvcr.io/nvidia/tensorrt:20.09-py3"
    # TRT 7.0.0
    "nvcr.io/nvidia/tensorrt:20.03-py3"
)

test_command="rm -rf build && mkdir build && cd build && cmake .. && make && \
./tinyexec --onnx /usr/src/tensorrt/data/resnet50/ResNet50.onnx"

cd ..
for image in ${image_list[@]}
do
    docker run --rm --gpus all -v `pwd`:/tiny-tensorrt -w /tiny-tensorrt ${image} /bin/bash -c "${test_command}"
done