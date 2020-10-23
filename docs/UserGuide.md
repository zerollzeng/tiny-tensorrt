
# Usage

Take a look at test/test.cpp or test/test.py

# C++ API
```c++
// for detailed usage, please refer to Trt.h, it's pretty well commented.
#include "Trt.h"

Trt trt;
// create engine and running context, note that engine file is device specific, so don't copy engine file to new device, it may cause crash
// trt.CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, maxBatchSize, mode); // for caffe model
trt.CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode); // for onnx model
// trt.CreateEngine(uffModelpath, engineFile, input, inputDims, output, maxBatchSize, mode); // for tensorflow model

// you might need to do some pre-processing in input such as normalization, it depends on your model.
trt.CopyFromHostToDevice(input,0); // 0 for input index, you can get it from CreateEngine phase log output.

//run model, it will read your input and run inference. and generate output.
trt.Forward();

//  get output.
trt.CopyFromDeviceToHost(output, outputIndex) // you can get outputIndex in CreateEngine phase
// them you can do post processing in output
```

# python API
```python
import sys
sys.path.append("path/to/where_pytrt.so_located/")
import pytrt
# for detailed usage, try uncomment next line
# help(pytrt)

trt = pytrt.Trt()
trt.CreateEngine(prototxt, caffemodel, engineFile, outputBlobName, maxBatchSize, mode)
# trt.CreateEngine(onnxModel, engineFile,customOutput,maxBatchSize,mode)
# trt.CreateEngine(uffModel, engineFile, inputTensorName, inputDims, outputTensorName,maxBatchSize,mode)
trt.CopyFromHostToDevice(input_numpy_array, 0)
trt.Forward()
output_numpy_array = trt.CopyFromDeviceToHost(1)
# post processing
```

also see [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo) and, it implement some common computer vision model with tiny-tensorrt, it has serveral good samples