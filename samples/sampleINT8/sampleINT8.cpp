#include "Trt.h"

#include <memory>

int main() {
    // create instance 
    std::unique_ptr<Trt> onnx_net{new Trt()};

    onnx_net->EnableFP16();
    onnx_net->EnableINT8();
    onnx_net->SetInt8Calibrator("EntropyCalibratorV2", 1, "/tmp/calibrate_data/", ""/*calibrateCachePath*/);
    onnx_net->AddDynamicShapeProfile("gpu_0/data_0", {1,3,224,224}, {1,3,224,224}, {1,3,224,224});
    // build engine
    onnx_net->BuildEngine("/usr/src/tensorrt/data/resnet50/ResNet50.onnx", "/tmp/resnet50.plan");

    // do inference
    for(int i=0; i<10; i++) {
        onnx_net->Forward();
    }
    printf("sampleINT8 PASSED\n");
}