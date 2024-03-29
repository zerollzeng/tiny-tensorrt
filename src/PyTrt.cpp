#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "spdlog/spdlog.h"

namespace py = pybind11;

#include "Trt.h"


PYBIND11_MODULE(pytrt, m) {
    m.doc() = "python interface of tiny-tensorrt";
    py::class_<Trt>(m, "Trt")
        .def(py::init([]() {
            return std::unique_ptr<Trt>(new Trt());
        }))
        .def("BuildEngine", (void (Trt::*)(
            const std::string&,
            const std::string&
            )) &Trt::BuildEngine, "create engine with onnx model")
        .def("Forward", (bool (Trt::*)()) &Trt::Forward, "inference")
        .def("CopyFromHostToDevice", [](Trt& self, py::array_t<float, py::array::c_style | py::array::forcecast> array, int index) {
            std::vector<float> input;
            input.resize(array.size());
            std::memcpy(input.data(), array.data(), array.size()*sizeof(float));
            self.CopyFromHostToDevice(input, index);
        })
        .def("CopyFromDeviceToHost", [](Trt& self, int outputIndex) {
            std::vector<float> output;
            self.CopyFromDeviceToHost(output, outputIndex);
            nvinfer1::Dims dims = self.GetBindingDims(outputIndex);
            ssize_t nbDims= dims.nbDims;
            std::vector<ssize_t> shape;
            for(int i=0;i<nbDims;i++){
                shape.push_back(dims.d[i]);
            }
            std::vector<ssize_t> strides;
            for(int i=0;i<nbDims;i++){
                ssize_t stride = sizeof(float);
                for(int j=i+1;j<nbDims;j++) {
                    stride = stride * shape[j];
                }
                strides.push_back(stride);
            }
            return py::array(py::buffer_info(
                output.data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                nbDims,
                shape,
                strides
            ));            
        })
        ;
}
