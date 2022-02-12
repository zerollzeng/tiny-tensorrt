#ifndef TRT_HPP
#define TRT_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>

#include "NvInfer.h"
#include "NvInferVersion.h"

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t)
    {
#if NV_TENSORRT_MAJOR < 8
        t->destroy();
#else
        delete t;
#endif
    }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

using Severity = nvinfer1::ILogger::Severity;
class TrtLogger : public nvinfer1::ILogger {
public:
    void setLogSeverity(Severity severity);

private:
    void log(Severity severity, const char* msg)  noexcept override;

    Severity mSeverity = Severity::kINFO;
};

/**
    * Set the GPU to use
    */
void SetDevice(int device);

/**
    * Get the GPU to use
    */
int GetDevice();

class Trt {
public:

    Trt();

    ~Trt();

    Trt(const Trt& trt) = delete;

    Trt& operator=(const Trt& trt) = delete;

    /**
     * Enable FP16 precision, by default TensorRT enable FP32 precision.
     */
    void EnableFP16();

    /**
     * Enable INT8 precision, by default TensorRT enable FP32 precision.
     */
    void EnableINT8();

    /**
     * Setting a int8 calibrator.To run INT8 calibration for a network with dynamic shapes, calibration optimization 
     * profile must be set. Calibration is performed using kOPT values of the profile. Calibration input data size 
     * must match this profile.
     * @calibratorData: use for int8 mode, calabrator data is a batch of sample input,
     *                  for classification task you need around 500 sample input. and this
     *                  is for int8 mode
     * @calibratorType: there are 3 calibrator types now.
     *                  "EntropyCalibratorV2" : This is the recommended calibrator and is required for DLA. Calibration 
     *                  happens before Layer fusion by default. This is recommended for CNN based networks.
     *                  "MinMaxCalibrator" : This is the preferred calibrator for NLP tasks for all backends. 
     *                  Calibration happens before Layer fusion by default. This is recommended for BERT like networks.
     *                  "EntropyCalibrator" : This is the legacy entropy calibrator.This is less complicated than a legacy 
     *                  calibrator and produces better results. Calibration happens after Layer fusion by default. See 
     *                  kCALIBRATION_BEFORE_FUSION for enabling calibration before fusion.
     */
    void SetInt8Calibrator(const std::string& calibratorType, const int batchSize,
                           const std::string& dataPath, const std::string& calibrateCachePath);

    /**
     * Set the maximum GPU temporary memory which the engine can use at execution time.
     */
    void SetWorkpaceSize(size_t workspaceSize);

    /**
     * Set dla core
     * @dlaCore dla core index, eg 0,1...
     */
    void SetDLACore(int dlaCore);

    /**
     * Set custom output, this will un-mark the original output
     * @customOutputs custom output node name list
     */
    void SetCustomOutput(const std::vector<std::string>& customOutputs);

    /**
     * Set tensorrt internal log level
     * @level Severity::kINTERNAL_ERROR = 0, Severity::kERROR = 1, Severity::kWARNING = 2, Severity::kINFO = 3,
     *                  Severity::kVERBOSE = 4, default level is <= kINFO.
     */
    void SetLogLevel(int severity);

    /**
     * Add dynamic shape profile
     */
    void AddDynamicShapeProfile(const std::string& inputName,
                                const std::vector<int>& minDimVec,
                                const std::vector<int>& optDimVec,
                                const std::vector<int>& maxDimVec);
    
    /**
     * Create engine from onnx model
     * @onnxModel: path to onnx model
     * @engineFile: path to saved engien file will be save, if it's empty them will not
     *              save engine file
     */
    void BuildEngine(const std::string& onnxModel, const std::string& engineFile);

    /**
     * Deserialize an engine from engineFile
     * Note: If your model has dynamic shapes, you must call AddDynamicShapeProfile
     *       before DesirializeEngine as you did when building engien from onnx.
     * @engineFile: can be create by BuildEngine, or save with trtexec or tiny-exec
     * @dlaCore: dla core to use, you can build engine on dla core 0 and deserialize the
     *           engine to core 1. Only available on jetson platform has DLA support.
     * return false if deserialization failed.
     */
    bool DeserializeEngine(const std::string& engineFile, int dlaCore=-1);


    /**
     * Do inference on engine context, make sure you already copy your data to device memory,
     * return true if success
     */
    bool Forward();

    /**
     * Async inference on engine context, return true if success
     * @stream cuda stream for async inference and data transfer
     */
    bool Forward(const cudaStream_t& stream);

    /**
     * Set input dimentiosn for an inference, call this before forward with dynamic shape mode.
     */
    void SetBindingDimensions(std::vector<int>& inputDims, int bindIndex);

    /**
     * Copy input from host to device
     * @bindIndex binding data index, you can see this in BuildEngine log output.
     */
    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex,const cudaStream_t& stream = 0);

    /**
     * Copy input from device to host
     * @bindIndex binding data index, you can see this in BuildEngine log output.
     */
    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,const cudaStream_t& stream = 0);

    /**
     * Get binding data pointer in device. for example if you want to do some post processing
     * on inference output but want to process them in gpu directly for efficiency, you can
     * use this function to avoid extra data IO, or you can copy inputs from device to
     * binding prt directly so that you don't need to call CopyFromHostToDevice. Hence
     * good for performance.
     * @return: pointer point to device memory.
     */
    void* GetBindingPtr(int bindIndex) const;

    /**
     * Get binding data size in byte, so maybe you need to divide it by sizeof(T) where T is data type
     *               like float.
     * @return: size in byte.
     */
    size_t GetBindingSize(int bindIndex) const;

    /**
     * Get binding dimemsions
     * @return: binding dimemsions, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_dims.html
     */
    nvinfer1::Dims GetBindingDims(int bindIndex) const;

    /**
     * Get binding data type
     * @return: binding data type, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/namespacenvinfer1.html#afec8200293dc7ed40aca48a763592217
     */
    nvinfer1::DataType GetBindingDataType(int bindIndex) const;

    /**
     * Get binding name
     */
    std::string GetBindingName(int bindIndex) const;

    /**
     * Get number of input bindings.
     */
    int GetNbInputBindings() const;

    /**
     * Get number of output bindings.
     */
    int GetNbOutputBindings() const;

protected:
    void CreateDeviceBuffer();

    std::unique_ptr<TrtLogger> mLogger{nullptr};

    TrtUniquePtr<nvinfer1::IBuilder> mBuilder{nullptr};

    TrtUniquePtr<nvinfer1::IBuilderConfig> mConfig{nullptr};

    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine{nullptr};

    TrtUniquePtr<nvinfer1::IExecutionContext> mContext{nullptr};

    nvinfer1::IOptimizationProfile* mProfile = nullptr;

    std::vector<std::string> mCustomOutputs;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<nvinfer1::DataType> mBindingDataType;

    std::vector<std::string> mBindingName;

    int mNbInputBindings = 0;

    int mNbOutputBindings = 0;

    bool mIsDynamicShape = false;
};

#endif
