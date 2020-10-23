/*
 * @Author: zerollzeng
 * @Date: 2019-08-29 15:45:15
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-03-02 15:09:53
 */
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "spdlog/spdlog.h"

namespace py = pybind11;

#include "Trt.h"


PYBIND11_MODULE(pytrt, m) {
    m.doc() = "python interface of tiny-tensorrt";
    py::class_<TrtPluginParams>(m, "TrtPluginParams")
        .def(py::init<>());
    py::class_<Trt>(m, "Trt")
        .def(py::init([]() {
            return std::unique_ptr<Trt>(new Trt());
        }))
        .def(py::init([](TrtPluginParams params) { 
            return std::unique_ptr<Trt>(new Trt(params));
        }))
        .def("CreateEngine", (void (Trt::*)(
            const std::string&,
            const std::string&,
            const std::string&,
            const std::vector<std::string>&,
            int,
            int
            )) &Trt::CreateEngine, "create engine with caffe model")
        .def("CreateEngine", (void (Trt::*)(
            const std::string&,
            const std::string&,
            const std::vector<std::string>&,
            int,
            int
            )) &Trt::CreateEngine, "create engine with onnx model")
        .def("CreateEngine", (void (Trt::*)(
            const std::string&,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::vector<int>>&,
            const std::vector<std::string>&,
            int,
            int
            )) &Trt::CreateEngine, "create engine with tensorflow model")
        .def("Forward", (void (Trt::*)()) &Trt::Forward, "inference")
        .def("SetDevice", (void (Trt::*)(
            int
            )) &Trt::SetDevice, "set building and inference device")
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