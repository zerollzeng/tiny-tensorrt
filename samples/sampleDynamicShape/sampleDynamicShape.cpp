#include "Trt.h"

#include <memory>
#include <cassert>
#include <vector>

int main() {
    // create instance 
    std::unique_ptr<Trt> onnx_net{new Trt()};

    // set dynamic shape config
    onnx_net->AddDynamicShapeProfile("x", {4,128,128,128}, {4,128,256,256}, {4,128,512,512});
    onnx_net->AddDynamicShapeProfile("y", {4,128,128,128}, {4,128,256,256}, {4,128,512,512});

    // build engine
    onnx_net->BuildEngine("/tmp/sample_add.onnx", "/tmp/sample_add.plan");

    // do inference
    int start = 128;
    int step = 48;
    for(int i=0; i<=8; i++) {
        int h = start + i*step;
        int w = start + i*step;
        std::vector<float> x(4*128*h*w, 1.0);
        std::vector<float> y(4*128*h*w, 2.0);
        std::vector<float> z(4*128*h*w, 0.0);

        std::vector<int> shape{4,128,h,w};
        onnx_net->SetBindingDimensions(shape, 0);
        onnx_net->SetBindingDimensions(shape, 1);

        onnx_net->CopyFromHostToDevice(x, 0);
        onnx_net->CopyFromHostToDevice(y, 1);
        onnx_net->Forward();
        onnx_net->CopyFromDeviceToHost(z, 2);

        for(int j=0;j<4*128*h*w;j++) {
            assert(z[j] == 3.0);
        }
        printf("Test case with input shape 4x128x%dx%d PASSED\n", h, w);
    }

}