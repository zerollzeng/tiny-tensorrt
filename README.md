<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2019-08-23 11:08:25
 * @LastEditors: Please set LastEditors
 -->

# tiny-tensorrt
a simple, efficient, easy-to-use nvidia TensorRT wrapper for cnn,sopport yolov3,openpose,mtcnn etc...

this project is fully test with TensorRT 5.1.5.0, cuda 9.0/9.2/10.0, ubuntu 16.04. I test it with 1060ti, 1050ti, 1080ti, 1660ti, 2080, and 2080ti.

current version is develop with TensorRT 5.1.5.0, and this project is still under development, any issue is welcome :)

# Quick start
```bash
# register at Nvidia NGC and pull official TensorRT image(https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
docker pull nvcr.io/nvidia/tensorrt:19.07-py3
mkdir build
mkdir lib
cd build && cmake .. && make
```
then you can intergrate it into your own project with libtinytrt.so in ./lib and Trt.h.

# How to use tiny-tensorrt
see [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo), it implement some common computer vision model with tiny tensor_rt

# Support layer
- upsample with custom scale (it have bug in deserialization, will fix it next week), under test with yolov3.
- yolo-det, last layer of yolov3 which sum three scales output and generate final result for nms. under test with yolov3.
- PRELU, under test with openpose.

# Roadmap
- [x] int8 support
- [ ] fix upsample bug
- [ ] support more model and layer


# Acknowledgement
this project is highly motivated by [lewes6369/tensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper) 


