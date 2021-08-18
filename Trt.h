/*
 * @Date: 2019-08-29 09:48:01
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-03-02 14:58:37
 */

#ifndef TRT_HPP
#define TRT_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "NvInfer.h"



class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg)  noexcept override
    {
        // suppress info-level messages
        if (severity != Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};

class Trt {
public:
    /**
     * @description: default constructor, will initialize plugin factory with default parameters.
     */
    Trt();

    ~Trt();

    /**
     * @description: create engine from onnx model
     * @onnxModel: path to onnx model
     * @engineFile: path to saved engien file will be load or save, if it's empty them will not
     *              save engine file
     * @maxBatchSize: max batch size for inference.
     * @return:
     */
    void CreateEngine(
        const std::string& onnxModel,
        const std::string& engineFile,
        int maxBatchSize,
        int mode);

    /**
     * @description: do inference on engine context, make sure you already copy your data to device memory,
     *               see DataTransfer and CopyFromHostToDevice etc.
     */
    void Forward();

    /**
     * @description: async inference on engine context
     * @stream cuda stream for async inference and data transfer
     */
    void ForwardAsync(const cudaStream_t& stream);

    void SetBindingDimensions(std::vector<int>& inputDims, int bindIndex);
    /**
     * @description: data transfer between host and device, for example befor Forward, you need
     *               copy input data from host to device, and after Forward, you need to transfer
     *               output result from device to host.
     * @bindIndex binding data index, you can see this in CreateEngine log output.
     */
    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex,const cudaStream_t& stream);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,const cudaStream_t& stream);

    void SetDevice(int device);

    int GetDevice() const;

    /**
     * @description: setting a int8 calibrator.To run INT8 calibration for a network with dynamic shapes, calibration optimization profile must be set. Calibration is performed using kOPT values of the profile. Calibration input data size must match this profile.
     * @calibratorData: use for int8 mode, calabrator data is a batch of sample input,
     *                  for classification task you need around 500 sample input. and this
     *                  is for int8 mode
     * @calibratorType: there are four calibrator types now.
     *                  EntropyCalibratorV2: This is the recommended calibrator and is required for DLA. Calibration happens before Layer fusion by default. This is recommended for CNN based networks.
     *                  MinMaxCalibrator:This is the preferred calibrator for NLP tasks for all backends. Calibration happens before Layer fusion by default. This is recommended for BERT like networks.
     *                  EntropyCalibrator:This is the legacy entropy calibrator.This is less complicated than a legacy calibrator and produces better results. Calibration happens after Layer fusion by default. See kCALIBRATION_BEFORE_FUSION for enabling calibration before fusion.
     *                  LegacyCalibrator:This calibrator is for compatibility with TensorRT 2.0 EA. This calibrator requires user parameterization, and is provided as a fallback option if the other calibrators yield poor results. Calibration happens after Layer fusion by default. See kCALIBRATION_BEFORE_FUSION for enabling calibration before fusion. Users can customize this calibrator to implement percentile max, like 99.99% percentile max is proved to have best accuracy for BERT. For more information, refer to the Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation paper.
     */
    void SetInt8Calibrator(const std::string& calibratorType, const int batchSize,
                           const std::string& dataPath, const std::string& calibrateCachePath);

    /**
     * @description: set dla core
     * @dlaCore dla core index, eg 0,1...
     */
    void SetDLACore(int dlaCore);

    /**
     * @description: set custom output, this will un-mark the original output
     * @customOutputs custom output node name list
     */
    void SetCustomOutput(const std::vector<std::string>& customOutputs);

    /**
     * @description: add dynamic shape profile
     */
    void AddDynamicShapeProfile(const std::string& inputName,
                                const std::vector<int>& minDimVec,
                                const std::vector<int>& optDimVec,
                                const std::vector<int>& maxDimVec);

    /**
     * @description: get max batch size of build engine.
     * @return: max batch size of build engine.
     */
    int GetMaxBatchSize() const;

    /**
     * @description: get binding data pointer in device. for example if you want to do some post processing
     *               on inference output but want to process them in gpu directly for efficiency, you can
     *               use this function to avoid extra data io
     * @return: pointer point to device memory.
     */
    void* GetBindingPtr(int bindIndex) const;

    /**
     * @description: get binding data size in byte, so maybe you need to divide it by sizeof(T) where T is data type
     *               like float.
     * @return: size in byte.
     */
    size_t GetBindingSize(int bindIndex) const;

    /**
     * @description: get binding dimemsions
     * @return: binding dimemsions, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_dims.html
     */
    nvinfer1::Dims GetBindingDims(int bindIndex) const;

    /**
     * @description: get binding data type
     * @return: binding data type, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/namespacenvinfer1.html#afec8200293dc7ed40aca48a763592217
     */
    nvinfer1::DataType GetBindingDataType(int bindIndex) const;

    /**
     * @description: get binding name
     */
    std::string GetBindingName(int bindIndex) const;

    int GetNbInputBindings() const;

    int GetNbOutputBindings() const;

protected:

    bool DeserializeEngine(const std::string& engineFile);

    void BuildEngine();

    bool BuildEngineWithOnnx(const std::string& onnxModel,
                     const std::string& engineFile,
                     const std::vector<std::string>& customOutput);

    /**
     * description: Init resource such as device memory
     */
    void InitEngine();

    /**
     * description: save engine to engine file
     */
    void SaveEngine(const std::string& fileName);

protected:
    TrtLogger mLogger;

    // tensorrt run mode 0:fp32 1:fp16 2:int8
    int mRunMode;

    // batch size
    int mBatchSize;

    nvinfer1::NetworkDefinitionCreationFlags mFlags = 0;

    nvinfer1::IBuilderConfig* mConfig = nullptr;

    nvinfer1::IBuilder* mBuilder = nullptr;

    nvinfer1::INetworkDefinition* mNetwork = nullptr;

    nvinfer1::ICudaEngine* mEngine = nullptr;

    nvinfer1::IExecutionContext* mContext = nullptr;

    nvinfer1::IRuntime* mRuntime = nullptr;

    nvinfer1::IOptimizationProfile* mProfile = nullptr;

    std::vector<std::string> mCustomOutputs;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<nvinfer1::DataType> mBindingDataType;

    std::vector<std::string> mBindingName;

    int mNbInputBindings = 0;

    int mNbOutputBindings = 0;
};

#endif
