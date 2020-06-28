[English](https://github.com/zerollzeng/tiny-tensorrt/blob/master/README.md) | 中文简体

![image](https://user-images.githubusercontent.com/38289304/71065174-aafc3100-21ab-11ea-9bcb-665d38181e74.png)

# tiny-tensorrt
一个非常高效易用的nvidia TensorRT封装,支持c++,python调用,支持caffe,onnx,tensorflow模型.只需要几行代码,就可以完成模型的推理部署
```c++
// 创建引擎
trt.CreateEngine(prototxt,caffemodel,engingefile,outputBlob,calibratorData,maxBatchSize,runMode);
// 将模型输入传输到显卡
trt.DataTransfer(input,0,True);
// 松手刹,挂挡,松离合,踩油门!
trt.Forward();
// 获取网络输出
trt.DataTransfer(output, outputIndex, False) // you can get outputIndex in CreateEngine phase
```

# 功能
- [x] 现在已经可以支持TensorRT 7 --- 2019-12-25 :christmas_tree::christmas_tree::christmas_tree:
- [x] 自定义插件教程和非常详细的示例代码! ---2019-12-11 :fire::fire::fire:
- [x] 自定义onnx模型输出节点 ---2019.10.18
- [x] 升级到TensorRT 6.0.1.5 --- 2019.9.29
- [x] 支持onnx, caffe, tensorflow模型
- [ ] 实现更多的层(有需要请给我提issue阿喂,但是我希望是通用的层) --working on
- [x] PRELU和upsample自定义层
- [x] 引擎序列化及反序列化
- [x] caffe int8支持
- [x] onnx支持
- [x] python api支持
- [x] 自定义使用显卡

# 系统需求
TensorRT 6或7 和 cuda 10.0+

如果要使用python api, 那么还需要安装python2/3 和 numpy

# 文档

[UserGuide](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/UserGuide.md)

[自定义层编写教程](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/CustomPlugin-CN.md) (En-Ch)

如果需要一些tiny-tensorrt的使用示例,可以参考[tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo)

如果你想在windows上使用tiny-tensorrt,你可以参考一下@Devincool的[移植](https://github.com/Devincool/tiny-tensorrt)

# 支持的额外层
- 自定义尺度upsample,在yolov3上测试
- yolo-det, 就是yolov3的最后一层,将三个尺度的输出集合起来产生检测结果
- PRELU, 在openpose和mtcnn上做了测试

# 版权
这个项目包含了一些第三方模块,对于它们需要遵守他们的版权要求

对于我写的那一部分,你可以做任何你想做的事
