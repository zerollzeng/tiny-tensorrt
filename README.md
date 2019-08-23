<!--
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 09:16:35
 * @LastEditTime: 2019-08-23 11:08:25
 * @LastEditors: Please set LastEditors
 -->

# tiny-tensorrt
a simple, efficient, easy-to-use nvidia TensorRT wrapper for cnn,sopport for yolov3,openpose,mtcnn etc...

current version is develop with TensorRT 5.1.5.0, and this project is still under development, any issue is welcome :)
# Quick start
```bash
# register at Nvidia NGC and pull official TensorRT image(https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
docker pull nvcr.io/nvidia/tensorrt:19.07-py3
mkdir build
mkdir lib
cd build && cmake .. && make
```

# runtime requirememt
cuda(version > 9.0 tested)
Nvidia TensorRT
