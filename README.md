<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2020-03-06 17:12:14
 * @LastEditors: zerollzeng
 -->

![image](https://user-images.githubusercontent.com/38289304/71065174-aafc3100-21ab-11ea-9bcb-665d38181e74.png)

## tiny-tensorrt
An easy-to-use nvidia TensorRT wrapper for onnx model with c++ and python api. you will be able to deploy your model with tiny-tensorrt in few lines of code!

```c++
Trt* net = Trt();
net->CreateEngine(onnxModel, engineFile,maxBatchSize, precision);
net->CopyFromHostToDevice(input, inputBindIndex);
net->Forward();
net->CopyFromDeviceToHost(output, outputBindIndex)
```

## Install

tiny-tensorrt rely on CUDA 10.2+ and TRT 8+. Make sure you has installed those dependencies already. For a quick start, you can use [official docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)

To build tiny-tensorrt, you also need some extra packages.
```bash
sudo apt-get update -y
sudo apt-get install cmake zlib1g-dev

## this is for python binding
sudo apt-get install python3 python3-pip
pip3 install numpy

## clone project and submodule
git clone --recurse-submodules -j8 https://github.com/zerollzeng/tiny-tensorrt.git

cd tiny-tensorrt
mkdir build && cd build

cmake .. && make
```
Then you can intergrate it into your own project with libtinytrt.so and Trt.h, for python module, you get pytrt.so

## News

Add support for dynamic shapes, **currently dynamic shape can not work with int8** - 2021-8-18

Upgrade to TensorRT 8.0 API, **checkout 7.x branch for use under TensorRT 7** - 2021-7-9

Better int8 calibrator api, refer to [User Guide](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/UserGuide.md) - 2021-5-24

Remove caffe and uff support, convert to onnx with tf2onnx or keras.onnx. - 2021-4-23

Want to implement your own onnx plugin and don't know where to start? - 2021-1-29

[onnx plugin template](https://github.com/zerollzeng/tiny-tensorrt/blob/master/plugin/CuteSamplePlugin)

## Features
- [x] Add dynamic shapes support
- [x] Add DLA support
- [x] Custom plugin tutorial and well_commented sample
- [x] Custom onnx model output node
- [x] Engine serialization and deserialization
- [x] INT8 support
- [x] Python api support
- [x] Set device

## Docs

[User Guide](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/UserGuide.md)

[Custom Plugin Tutorial](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/CustomPlugin.md) (En-Ch)

## About License

For the 3rd-party module and TensorRT, you need to follow their license

For the part I wrote, you can do anything you want

