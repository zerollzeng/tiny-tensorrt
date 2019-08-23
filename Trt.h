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

enum class RUN_MODE {
    FLOAT32 = 0,
    FLOAT16 = 1,    
    INT8 = 2
};

struct TrtPluginParams {
    // yolo-det layer
    int yoloClassNum = 1; 
    int yolo3NetSize = 416; // 416 or 608

    // upsample layer
    int upsampleScale = 2;
};

class PluginFactory;

class Trt {
public:
    Trt(TrtPluginParams* param = nullptr);

    ~Trt();

    /**
     * description: create engine from caffe prototxt and caffe model
     * @prototxt: caffe prototxt
     * @caffemodel: caffe model contain network parameters
     * @saveEngine: serialzed engine file, if it does not exit, will build engine from
     *             prototxt and caffe model, which take about 1 minites, otherwise will
     *             deserialize enfine from engine file, which is very fast.
     * @outputBlobName: specify which layer is network output, find it in caffe prototxt
     * @calibratorData: use for int8 mode, not support now.
     * @maxBatchSize: batch size
     */
    void CreateEngine(const std::string& prototxt, 
                        const std::string& caffeModel,
                        const std::string& saveEngine,
                        const std::vector<std::string>& outputBlobName,
                        const std::vector<std::vector<float>>& calibratorData,
                        int maxBatchSize,
                        RUN_MODE mode);

    void Forward(const cudaStream_t& stream);

    /**
     * @description: print layer time, not support now
     */
    void PrintTime();

    void CopyFromHostToDevice(const void* pData, int index,const cudaStream_t& stream);

    void CopyFromDeviceToHost(void* pData, int index,const  cudaStream_t& stream);

    int GetMaxBatchSize();

    void* GetBindingPtr(int index) const;

    size_t GetBindingSize(int index) const;

    nvinfer1::Dims GetBindingDims(int index) const;

    nvinfer1::DataType GetBindingDataType(int index) const;

protected:

    bool DeserializeEngine(const std::string& engineFile);

    bool BuildEngine(const std::string& prototxt, 
                    const std::string& caffeModel,
                    const std::string& engineFile,
                    const std::vector<std::string>& outputBlobName,
                    const std::vector<std::vector<float>>& calibratorData,
                    int maxBatchSize);
    /**
     * description: Init resource such as device memory, must implement it
     */
    void InitEngine();

    /**
     * description: save engine to engine file
     */
    void SaveEngine(std::string fileName);

protected:
    TrtLogger mLogger;

    // tensorrt run mode, see RUN_MODE, only support fp32 now.
    RUN_MODE mRunMode;

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