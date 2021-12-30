<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2020-03-06 17:12:14
 * @LastEditors: zerollzeng
 -->

![image](https://user-images.githubusercontent.com/38289304/71065174-aafc3100-21ab-11ea-9bcb-665d38181e74.png)

tiny-tensorrt 1.0.0 release! a lot of code clean && optimization, support both of TRT 7 and TRT 8!

## tiny-tensorrt
An easy-to-use nvidia TensorRT wrapper for onnx model with c++ and python api. you will be able to deploy your model with tiny-tensorrt in few lines of code!

```c++
Trt* net = new Trt();
net->SetFP16();
net->BuildEngine(onnxModel, engineFile);
net->CopyFromHostToDevice(input, inputBindIndex);
net->Forward();
net->CopyFromDeviceToHost(output, outputBindIndex)
```

## Install

tiny-tensorrt rely on CUDA, CUDNN and TensorRT. Make sure you has installed those dependencies already. For a quick start, you can use [official docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)

Support CUDA version: 10.2, 11.0, 11.1, 11.2, 11.3, 11.4

Support TensorRT version: 7.0, 7.1, 7.2, 8.0, 8.2

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

