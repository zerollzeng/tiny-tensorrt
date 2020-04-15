/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2020-03-02 15:16:08
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-04-15 10:17:12
 */

#include "Trt.h"

#include <string>
#include <vector>

void test_caffe(
        const std::string& prototxt, 
        const std::string& caffeModel,
        const std::vector<std::string>& outputBlobName) {
    std::string engineFile = "";
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = 0;
    Trt* caffe_net = new Trt();
    caffe_net->CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, maxBatchSize, mode, calibratorData);
    caffe_net->Forward();
}

void test_onnx(const std::string& onnxModelpath) {
    std::string engineFile = "";
    const std::vector<std::string> customOutput;
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = 0;
    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, calibratorData);
    onnx_net->Forward();
}

void test_uff(const std::string& uffModelpath) {
    std::string engineFile = "";
    std::vector<std::vector<float>> calibratorData;
    std::vector<std::string> input{"normalized_input_image_tensor"};
    std::vector<int> input_dims{1,320,320,3};
    std::vector<std::vector<int>> inputDims{input_dims};
    std::vector<std::string> output{"raw_outputs/class_predictions","raw_outputs/box_encodings"};
    int maxBatchSize = 1;
    int mode = 0;
    Trt* uff_net = new Trt();
    uff_net->CreateEngine(uffModelpath, engineFile, input, inputDims, output, maxBatchSize, mode, calibratorData);
    uff_net->Forward();
}

int main() {
    test_onnx("../models/retinaface.onnx");

    // std::vector<std::string> outputBlobName{"prob"};
    // test_caffe("../models/lenet.prototxt","../models/lenet_iter_10000.caffemodel",outputBlobName);

    // test_uff("../models/frozen_inference_graph.uff");
    
    return 0;
}