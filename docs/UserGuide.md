# Installation
Make sure you had install dependencies list above, if you are familiar with docker, you can use [official docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
```bash
# clone project and submodule
git clone --recurse-submodules -j8 https://github.com/zerollzeng/tiny-tensorrt.git

cd tiny-tensorrt

mkdir build && cd build && cmake .. && make
```
then you can intergrate it into your own project with libtinytrt.so and Trt.h, for python module, you get pytrt.so

# C++ API
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
// trt.CreateEngine(uffModel, engineFile, inputTensorName, inputDims, outputTensorName, maxBatchSize); // for tensorflow model

// you might need to do some pre-processing in input such as normalization, it depends on your model.
trt.DataTransfer(input,0,True); // 0 for input index, you can get it from CreateEngine phase log output, True for copy input date to gpu

//run model, it will read your input and run inference. and generate output.
trt.Forward();

//  get output.
trt.DataTransfer(output, outputIndex, False) // you can get outputIndex in CreateEngine phase
// them you can do post processing in output
```

# python API
```python
import sys
sys.path.append("path/to/where_pytrt.so_located/")
import pytrt

trt = pytrt.Trt()
trt.CreateEngine(prototxt, caffemodel, engineFile, outputBlobName, calibratorData, maxBatchSize, mode)
# trt.CreateEngine(onnxModel, engineFile, maxBatchSize)
# trt.CreateEngine(uffModel, engineFile, inputTensorName, inputDims, outputTensorName, maxBatchSize)
# see c++ CreateEngine

trt.DoInference(input_numpy_array) # slightly different from c++
output_numpy_array = trt.GetOutput(outputIndex)
# post processing
```

also see [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo), it implement some common computer vision model with tiny tensor_rt, it has serveral good samples