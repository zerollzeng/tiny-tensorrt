
# Usage

you can take a loog at ./test

# C++ API
```c++
#include "Trt.h"

Trt trt;
// create engine and running context, note that engine file is device specific, so don't copy engine file to new device, it may cause crash
trt.CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, maxBatchSize, mode, calibratorData);
// trt.CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, calibratorData); // for onnx model
// trt.CreateEngine(uffModelpath, engineFile, input, inputDims, output, maxBatchSize, mode, calibratorData); // for tensorflow model

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
# trt.CreateEngine(onnxModel, engineFile,customOutput,maxBatchSize,mode,calibratorData)
# trt.CreateEngine(uffModel, engineFile, inputTensorName, inputDims, outputTensorName,maxBatchSize,mode,calibratorData)

trt.DoInference(input_numpy_array) # slightly different from c++
output_numpy_array = trt.GetOutput(outputIndex)
# post processing
```

also see [tensorrt-zoo](https://github.com/zerollzeng/tensorrt-zoo) and, it implement some common computer vision model with tiny tensor_rt, it has serveral good samples