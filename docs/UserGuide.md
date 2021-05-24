
# Usage

Take a look at test/test.cpp or test/test.py

# C++ API
```c++
// for detailed usage, please refer to Trt.h, it's pretty well commented.
#include "Trt.h"

Trt trt;
// create engine and running context, note that engine file is device specific, so don't copy engine file to new device, it may cause crash
trt.CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode); // for onnx model

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
trt.CopyFromHostToDevice(input_numpy_array, 0)
trt.Forward()
output_numpy_array = trt.CopyFromDeviceToHost(1)
# post processing
```

# How to do int8 calibration with tiny-tensorrt

tiny-tensorrt provide a easy-to-use calibration solution now. you can use python to generate the calibration data and load it with c++.

following steps below:

1. modify and run generate_calibration_data.py to generate the calibration data, you can do pre-processing with python, it's pretty easy.

2. use the SetInt8Calibrator(), or use the tinyexec for test purpose.
```
./tinyexec --onnx /usr/src/tensorrt/data/resnet50/ResNet50.onnx --mode 2 --batch_size 1 --calibrate_data calibrate_data/
```