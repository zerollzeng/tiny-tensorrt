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
Trt* net = new Trt();
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

## Docs

Please refer to [Wiki](https://github.com/zerollzeng/tiny-tensorrt/wiki)

## About License

For the 3rd-party module and TensorRT, you need to follow their license

For the part I wrote, you can do anything you want

