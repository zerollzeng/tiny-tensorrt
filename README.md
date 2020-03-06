<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2020-03-06 17:12:14
 * @LastEditors: zerollzeng
 -->

English | [中文简体](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/README-CN.md)

![image](https://user-images.githubusercontent.com/38289304/71065174-aafc3100-21ab-11ea-9bcb-665d38181e74.png)

# tiny-tensorrt
a simple, efficient, easy-to-use nvidia TensorRT wrapper for cnn with c++ and python api,support caffe, uff and onnx format models. you will be able use tiny-tensorrt deploy your model with few lines of code!
```c++
// create engine
trt.CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, calibratorData);
// transfer you input data to tensorrt engine
trt.DataTransfer(input,0,True);
// inference!!!
trt.Forward();
//  retrieve network output
trt.DataTransfer(output, outputIndex, False) // you can get outputIndex in CreateEngine phase
```

# Features
- [x] Support TensorRT 7 now --- 2019-12-25 :christmas_tree::christmas_tree::christmas_tree:
- [x] Custom plugin tutorial and well_commented sample! ---2019-12-11 :fire::fire::fire:
- [x] Custom onnx model output node  ---2019.10.18
- [x] Upgrade with TensorRT 6.0.1.5 --- 2019.9.29
- [x] Support onnx,caffe and tensorflow model
- [ ] Support more model and layer --working on
- [x] PReLU and up-sample plugin
- [x] Engine serialization and deserialization
- [x] INT8 support for caffe model
- [x] Python api support
- [x] Set device

# System Requirements
cuda 10.0+

TensorRT 6 or 7

for python api, python 2.x/3.x and numpy in needed

# Installation
Make sure you had install dependencies list above, if you are familiar with docker, you can use [official docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
```bash
# clone project and submodule
git clone --recurse-submodules -j8 https://github.com/zerollzeng/tiny-tensorrt.git

cd tiny-tensorrt

mkdir build && cd build && cmake .. && make
```
then you can intergrate it into your own project with libtinytrt.so and Trt.h, for python module, you get pytrt.so

# Docs

[User Guide](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/UserGuide.md)

[Custom Plugin Tutorial](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/CustomPlugin.md) (En-Ch)

if you want some examples with tiny-tensorrt, you can refer to [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo)

# Extra Support layer
- upsample with custom scale, under test with yolov3.
- yolo-det, last layer of yolov3 which sum three scales output and generate final result for nms. under test with yolov3.
- PRELU, under test with openpose and mtcnn.

# About License
For the 3rd-party module and TensorRT, maybe you need to follow their license

For the part I wrote, you can do anything you want

