<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2019-12-09 12:39:49
 * @LastEditors: zerollzeng
 -->

English | [中文简体](https://github.com/zerollzeng/tiny-tensorrt/blob/master/README-CN.md)

# tiny-tensorrt
a simple, efficient, easy-to-use nvidia TensorRT wrapper for cnn with c++ and python api,support caffe, uff and onnx format models.

# Note
TensorRT release it's 6.x version, I upgrade tiny-tensorrt with it, so the old 5.x version was in trt-5.1.5.0 branch. 

I won't add features to 5.x version. keep with new version is better

# Roadmap
- [x] custom plugin tutorial and well_commented sample!, will finish in this week :fire::fire::fire:
- [ ] uff support 
- [x] custom onnx model output node  ---2019.10.18
- [x] upgrade with TensorRT 6.0.1.5 --- 2019.9.29
- [ ] support more model and layer --working on
- [x] caffe model support
- [x] PRELU support
- [x] upsample support
- [x] engine serialization
- [x] caffe model int8 support
- [x] onnx support
- [x] python api support
- [x] test in nvidia p4
- [x] set device

# System Requirements
cuda 10.0+

TensorRT 6.x

for python api, python 2.x/3.x and numpy in needed

this project is fully test in ubuntu 16.04. I tested it with 1060ti, 1050ti, 1080ti, 1660ti, 2080, 2080ti and p4.

# Installation
Make sure you had install dependencies list above, if you are familiar with docker, you can use [offcial docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
```bash
# clone project and submodule
git clone --recurse-submodules -j8 https://github.com/zerollzeng/tiny-tensorrt.git

cd tiny-tensorrt

mkdir build && cd build && cmake .. && make
```
then you can intergrate it into your own project with libtinytrt.so and Trt.h, for python module, you get pytrt.so

# How-to-use-it
c++
```c++
#include "Trt.h"

Trt trt;
// create engine and running context, note that engine file is device specific, so don't copy engine file to new device, it may cause crash
trt.CreateEngine("path/to/sample.prototxt",
                 "path/to/sample.caffemodel",
                 "path/to/engineFile", // since build engine is time consuming,so save we can serialize engine to file, it's much more faster
                 "outputblob",
                 calibratorData,
                 maxBatchSize,
                 runMode);
// trt.CreateEngine(onnxModel,engineFile,maxBatchSize); // for onnx model

// you might need to do some pre-processing in input such as normalization, it depends on your model.
trt.DataTransfer(input,0,True); // 0 for input index, you can get it from CreateEngine phase log output, True for copy input date to gpu

//run model, it will read your input and run inference. and generate output.
trt.Forward();

//  get output.
trt.DataTransfer(output, outputIndex, False) // you can get outputIndex in CreateEngine phase
// them you can do post processing in output
```

python
```python
import sys
sys.path.append("path/to/where_pytrt.so_located/")
import pytrt

trt = pytrt.Trt()
trt.CreateEngine(prototxt, caffemodel, engineFile, outputBlobName, calibratorData, maxBatchSize, mode)
# trt.CreateEngine(onnxModel, engineFile, maxBatchSize)
# see c++ CreateEngine

trt.DoInference(input_numpy_array) # slightly different from c++
output_numpy_array = trt.GetOutput(outputIndex)
# post processing
```

also see [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo), it implement some common computer vision model with tiny tensor_rt, it has serveral good samples

# Extra Support layer
- upsample with custom scale, under test with yolov3.
- yolo-det, last layer of yolov3 which sum three scales output and generate final result for nms. under test with yolov3.
- PRELU, under test with openpose and mtcnn.


# Acknowledgement
this project is originally motivated by [lewes6369/tensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper) and [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3), I make use of his upsample and yolo-det plugin with slightly optimization.

and I use [spdlog](https://github.com/gabime/spdlog) for some fancy log output, it's very lightweight for intergrate. 

I use pybind11 for python api binding.

# About License
For the 3rd-party module and TensorRT, maybe you need to follow their license

For the part I wrote, you can do anything you want

