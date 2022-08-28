#!/bin/bash
set -x

image_list=(
    # TRT 8.4.1
    "nvcr.io/nvidia/tensorrt:22.07-py3"
    # TRT 8.2.1
    "nvcr.io/nvidia/tensorrt:21.12-py3"
    # TRT 8.0.3
    "nvcr.io/nvidia/tensorrt:21.09-py3"
    # TRT 7.2.3.4
    "nvcr.io/nvidia/tensorrt:21.06-py3"
    # TRT 7.1.3
    "nvcr.io/nvidia/tensorrt:20.09-py3"
    # TRT 7.0.0
    "nvcr.io/nvidia/tensorrt:20.03-py3"
)

test_command="rm -rf build && mkdir build && cd build && cmake .. && make"

cd ..
for image in ${image_list[@]}
do
    docker run --rm --gpus all -u $(id -u):$(id -g) -v `pwd`:/tiny-tensorrt -w /tiny-tensorrt ${image} /bin/bash -c "${test_command}"
done
