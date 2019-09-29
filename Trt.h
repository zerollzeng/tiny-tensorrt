/*
 * @Date: 2019-08-29 09:48:01
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-09-29 10:33:23
 */

#ifndef TRT_HPP
#define TRT_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "NvInfer.h"
#include "NvCaffeParser.h"



class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

class Profiler : public nvinfer1::IProfiler
{
public:
    void printLayerTimes(int itrationsTimes)
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / itrationsTimes);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
    }
private:
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
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
    /**
     * @description: default constructor, will initialize plugin factory with default parameters.
     */
    Trt();

    /**
     * @description: if you costomize some parameters, use this.
     */
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
     * @calibratorData: use for int8 mode, calabrator data is a batch of sample input, 
     *                  for classification task you need around 500 sample input. and this
     *                  is for int8 mode
     * @maxBatchSize: max batch size while inference, make sure it do not exceed max batch
     *                size in your model
     * @mode: engine run mode, 0 for float32, 1 for float16, 2 for int8
     */
    void CreateEngine(const std::string& prototxt, 
                        const std::string& caffeModel,
                        const std::string& engineFile,
                        const std::vector<std::string>& outputBlobName,
                        const std::vector<std::vector<float>>& calibratorData,
                        int maxBatchSize,
                        int mode);
    
    /**
     * @description: create engine from onnx model
     * @onnxModelPath: path to onnx model
     * @engineFile: path to saved engien file will be load or save, if it's empty them will not
     *              save engine file
     * @maxBatchSize: max batch size for inference.
     * @return: 
     */
    void CreateEngine(const std::string& onnxModelpath,
                      const std::string& engineFile,
                      int maxBatchSize);

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

    /**
     * @description: print layer time, not support now
     */
    void PrintTime();

    /**
     * @description: data transfer between host and device, for example befor Forward, you need
     *               copy input data from host to device, and after Forward, you need to transfer
     *               output result from device to host.
     * @data data for read and write.
     * @bindIndex binding data index, you can see this in CreateEngine log output.
     * @isHostToDevice 0 for device to host, 1 for host to device (host: cpu memory, device: gpu memory)
     */
    void DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice);

    /**
     * @description: async data tranfer between host and device, see above.
     * @stream cuda stream for async interface and data transfer.
     * @return: 
     */
    void DataTransferAsync(std::vector<float>& data, int bindIndex, bool isHostToDevice, cudaStream_t& stream);

    // this four data method tranfer might be deprecated in future
    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex,const cudaStream_t& stream);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,const cudaStream_t& stream);

    
    void SetDevice(int device);

    int GetDevice() const;

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

    std::vector<std::string> mBindingName;

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
     * description: Init resource such as device memory
     */
    void InitEngine();

    /**
     * description: save engine to engine file
     */
    void SaveEngine(const std::string& fileName);

protected:
    TrtLogger mLogger;

    Profiler mProfiler;

    // tensorrt run mode 0:fp32 1:fp16 2:int8
    int mRunMode;

    nvinfer1::ICudaEngine* mEngine;

    nvinfer1::IExecutionContext* mContext;

    PluginFactory* mPluginFactory;

    nvinfer1::IRuntime* mRuntime;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<nvinfer1::DataType> mBindingDataType;

    int mInputSize = 0;

    // batch size
    int mBatchSize; 
};

#endif