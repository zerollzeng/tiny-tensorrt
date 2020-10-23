'''
@Email: zerollzeng@gmail.com
@Author: zerollzeng
@Date: 2020-03-02 15:16:16
@LastEditors: zerollzeng
@LastEditTime: 2020-03-06 17:06:45
'''

import sys
sys.path.append("../build")
model_path = str(sys.argv[1])
import pytrt
print(help(pytrt))
import numpy as np
def test_onnx():
    trt = pytrt.Trt()
    onnxModel = model_path
    engineFile = ""
    customOutput = []
    maxBatchSize = 1
    # calibratorData = [np.ones(28*28)]
    mode = 0
    trt.CreateEngine( onnxModel, engineFile,customOutput,maxBatchSize,mode)
    input_numpy_array = np.zeros(3*224*224)
    trt.CopyFromHostToDevice(input_numpy_array, 0)
    trt.Forward()
    output_numpy_array = trt.CopyFromDeviceToHost(1)

if __name__ == "__main__":
    test_onnx()