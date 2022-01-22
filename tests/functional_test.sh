#!/bin/bash
set -x

# TRT 8.2.1
image="nvcr.io/nvidia/tensorrt:21.12-py3"

test_command="rm -rf build && mkdir build && cd build && cmake .. && make && \
./samples/tinyexec/tinyexec --onnx /usr/src/tensorrt/data/resnet50/ResNet50.onnx --save_engine /tmp/resnet50_fp32.plan && \
./samples/tinyexec/tinyexec --load_engine /tmp/resnet50_fp32.plan && \
./samples/tinyexec/tinyexec --onnx /usr/src/tensorrt/data/resnet50/ResNet50.onnx --int8 --fp16 --save_engine /tmp/resnet50_fp32.plan && \
pip3 install -r ../samples/sampleDynamicShape/requirements.txt && \
python3 ../samples/sampleDynamicShape/make_add_op.py && \
./samples/sampleDynamicShape/sampleDynamicShape && \
pip3 install -r ../samples/sampleINT8/requirements.txt && \
python3 ../samples/sampleINT8/generate_calibration_data.py && \
./samples/sampleINT8/sampleINT8"

cd ..
docker run --rm --gpus all -v `pwd`:/tiny-tensorrt -w /tiny-tensorrt ${image} /bin/bash -c "${test_command}"
