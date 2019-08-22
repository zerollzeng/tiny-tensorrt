#ifndef YOLO_LAYER_PLUGIN_HPP
#define YOLO_LAYER_PLUGIN_HPP


#include <iostream>
#include <map>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "PRELUKernel.cuh"
#include "utils.h"

using namespace nvinfer1;
using namespace plugin;


class YoloLayerPlugin : public nvinfer1::IPluginV2
{
public:
    YoloLayerPlugin(int classCount);

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin(const void *data, size_t length);

    ~YoloLayerPlugin();

    virtual int getNbOutputs() const override {
        return 1;
    }

    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(DataType type, PluginFormat format) const override {
        return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
    }

    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {

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
    
    virtual IPluginV2* clone() const override;

    virtual void setPluginNamespace(const char* pluginNamespace) override {

    }
    
    virtual const char* getPluginNamespace() const override;

    void forwardCpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);

    void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
    
private:
    int mClassCount;
    int mKernelCount;
    std::vector<YoloKernel> mYoloKernel;
    int mThreadCount = 512;

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
    virtual const PluginFieldCollection* getFieldNames() override;

    // return nullptr in case of error
    virtual IPluginV2* createPlugin(const char* name, const PluginFieldCollection *fc) override;

    // Called during deserialization of plugin layer. Return a plugin object.
    virtual IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLenth) override;

    // Set the namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator
    virtual void setPluginNamespace(const char* pluginNamespace) override;

    // Return the namespace of the plugin creator object.
    virtual const char* getPluginNamespace() const override;
};

#endif