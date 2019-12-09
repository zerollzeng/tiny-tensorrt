/*
 * @Description: yolo-det layer
 * @Author: zerollzeng
 * @Date: 2019-08-23 11:09:26
 * @LastEditTime: 2019-12-06 17:57:41
 * @LastEditors: zerollzeng
 */
#ifndef YOLO_LAYER_PLUGIN_HPP
#define YOLO_LAYER_PLUGIN_HPP


#include <iostream>
#include <map>
#include <cuda_runtime_api.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "plugin_utils.h"

static constexpr int CHECK_COUNT = 3;
static constexpr float IGNORE_THRESH = 0.5f;

struct YoloKernel
{
    int width;
    int height;
    float anchors[CHECK_COUNT*2];
};


class YoloLayerPlugin : public nvinfer1::IPluginV2
{
public:
    YoloLayerPlugin(int classCount, int netSize);

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin(const void *data, size_t length);

    ~YoloLayerPlugin();

    virtual int getNbOutputs() const override {
        return 1;
    }

    virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        return type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW;
    }

    virtual void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override {

    }
    
    virtual int initialize() override;

    virtual void terminate() override {

    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override {
        return 0;
    }
    
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    
    virtual size_t getSerializationSize() const override;

    virtual void serialize(void* buffer) const override;

    virtual const char* getPluginType() const override;
    
    virtual const char* getPluginVersion() const override;
    
    virtual void destroy();
    
    virtual nvinfer1::IPluginV2* clone() const override;

    virtual void setPluginNamespace(const char* pluginNamespace) override {

    }
    
    virtual const char* getPluginNamespace() const override;

    void forwardCpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);

    void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
    
private:
    int mClassCount;
    int mYolo3NetSize;
    int mKernelCount;
    std::vector<YoloKernel> mYoloKernel;

    //cpu
    void* mInputBuffer  {nullptr}; 
    void* mOutputBuffer {nullptr}; 
};


class YoloLayerPluginCreator : public nvinfer1::IPluginCreator {
public:
    // ------------------inherit from IPluginCreator-------------------
    // return the plugin name
    virtual const char* getPluginName() const override;

    // return the plugin version
    virtual const char* getPluginVersion() const override;

    // return a list of fields that needs to be passed to createPlugin
    virtual const nvinfer1::PluginFieldCollection* getFieldNames() override;

    // return nullptr in case of error
    virtual nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) override;

    // Called during deserialization of plugin layer. Return a plugin object.
    virtual nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLenth) override;

    // Set the namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator
    virtual void setPluginNamespace(const char* pluginNamespace) override;

    // Return the namespace of the plugin creator object.
    virtual const char* getPluginNamespace() const override;
};

#endif