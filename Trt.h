/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-23 14:26:05
 * @LastEditTime: 2019-08-23 14:38:00
 * @LastEditors: Please set LastEditors
 */

#ifndef TRT_HPP
#define TRT_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>

#include <NvInfer.h>
#include "NvCaffeParser.h"


class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

struct TrtPluginParams {
    // yolo-det layer
    int yoloClassNum = 1; 
    int yolo3NetSize = 416; // 416 or 608

    // upsample layer
    float upsampleScale = 2;
};

class PluginFactory;

class Trt {
public:
    Trt();

    Trt(TrtPluginParams params);

    ~Trt();

    /**
     * description: create engine from caffe prototxt and caffe model
     * @prototxt: caffe prototxt
     * @caffemodel: caffe model contain network parameters
     * @engineFile: serialzed engine file, if it does not exit, will build engine from
     *             prototxt and caffe model, which take about 1 minites, otherwise will
     *             deserialize enfine from engine file, which is very fast.
     * @outputBlobName: specify which layer is network output, find it in caffe prototxt
     * @calibratorData: use for int8 mode, not support now.
     * @maxBatchSize: batch size
     */
    void CreateEngine(const std::string& prototxt, 
                        const std::string& caffeModel,
                        const std::string& engineFile,
                        const std::vector<std::string>& outputBlobName,
                        const std::vector<std::vector<float>>& calibratorData,
                        int maxBatchSize,
                        int mode);
    
    void CreateEngine(const std::string& onnxModelpath,
                      const std::string& engineFile,
                      int maxBatchSize);

    void Forward();

    void Forward(const cudaStream_t& stream);

    /**
     * @description: print layer time, not support now
     */
    void PrintTime();

    void DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice);

    void DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice, cudaStream_t& stream);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex,const cudaStream_t& stream);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,const cudaStream_t& stream);

    int GetMaxBatchSize();

    void* GetBindingPtr(int bindIndex) const;

    size_t GetBindingSize(int bindIndex) const;

    nvinfer1::Dims GetBindingDims(int bindIndex) const;

    nvinfer1::DataType GetBindingDataType(int bindIndex) const;

protected:

    bool DeserializeEngine(const std::string& engineFile);

    bool BuildEngine(const std::string& prototxt, 
                    const std::string& caffeModel,
                    const std::string& engineFile,
                    const std::vector<std::string>& outputBlobName,
                    const std::vector<std::vector<float>>& calibratorData,
                    int maxBatchSize);

    bool BuildEngine(const std::string& onnxModelpath,
                     const std::string& engineFile,
                     int maxBatchSize);
    /**
     * description: Init resource such as device memory, must implement it
     */
    void InitEngine();

    /**
     * description: save engine to engine file
     */
    void SaveEngine(const std::string& fileName);

protected:
    TrtLogger mLogger;

    // tensorrt run mode, see int, only support fp32 now.
    int mRunMode;

    nvinfer1::ICudaEngine* mEngine;

    nvinfer1::IExecutionContext* mContext;

    PluginFactory* mPluginFactory;

    nvinfer1::IRuntime* mRuntime;

    nvinfer1::IProfiler* mProfiler;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<std::string> mBindingName;

    std::vector<nvinfer1::DataType> mBindingDataType;

    int mInputSize = 0;

    // batch size
    int mBatchSize; 
};
#endif