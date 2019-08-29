<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2019-08-29 11:18:36
 * @LastEditors: zerollzeng
 -->

# tiny-tensorrt
a simple, efficient, easy-to-use nvidia TensorRT wrapper for cnn with c++ and python api,sopport caffe and onnx format models.

this project is fully test with TensorRT 5.1.5.0, cuda 9.0/9.2/10.0, ubuntu 16.04. I test it with 1060ti, 1050ti, 1080ti, 1660ti, 2080, and 2080ti.

# Quick start
```bash
# register at Nvidia NGC and pull official TensorRT image(https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
docker pull nvcr.io/nvidia/tensorrt:19.07-py3
# build in docker
mkdir build && cd build && cmake .. && make
```
then you can intergrate it into your own project with libtinytrt.so and Trt.h.

# How to use tiny-tensorrt
see [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo), it implement some common computer vision model with tiny tensor_rt

# Support layer
- upsample with custom scale (it have bug in deserialization, will fix it next week), under test with yolov3.
- yolo-det, last layer of yolov3 which sum three scales output and generate final result for nms. under test with yolov3.
- PRELU, under test with openpose.

# Roadmap
- [ ] support more model and layer --working on
- [x] caffe model support
- [x] PRELU support
- [x] upsample support
- [x] engine serialization
- [x] caffe model int8 support
- [x] onnx support
- [x] python api support
- [ ] maybe a handing calibrator data creating tool
- [ ] test in nvidia p4


# Acknowledgement
this project is originally motivated by [lewes6369/tensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper) and [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3), I make use of his upsample and yolo-det plugin with slightly optimization.

and I use [spdlog](https://github.com/gabime/spdlog) for some fancy log output, it's very lightweight for intergrate. 


