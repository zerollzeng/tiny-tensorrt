#ifndef UPSAMPLE_PLUGIN_HPP
#define UPSAMPLE_PLUGIN_HPP

#include <iostream>
#include <map>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace plugin;


class UpSamplePlugin : public nvinfer1::IPluginV2
{
public:
    UpSamplePlugin(const float scale);

    // create the plugin at runtime from a byte stream
    UpSamplePlugin(const void *data, size_t length);

    ~UpSamplePlugin();

    virtual int getNbOutputs() const override {
        return 1;
    }

    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(DataType type, PluginFormat format) const override {
        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW;
    }

    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;
    
    virtual int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override {
        return 0;
    }
    
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    
    virtual size_t getSerializationSize() const override {
        return sizeof(nvinfer1::Dims) + sizeof(mDataType) + sizeof(mScale)
               + sizeof(mOutputWidth) + sizeof(mOutputHeight) + sizeof(mThreadCount);
    }

    virtual void serialize(void* buffer) const override;

    virtual const char* getPluginType() const override;
    
    virtual const char* getPluginVersion() const override;
    
    virtual void destroy();
    
    virtual IPluginV2* clone() const override;

    virtual void setPluginNamespace(const char* pluginNamespace) override;
    
    virtual const char* getPluginNamespace() const override;

    template <typename Dtype>
    void forwardGpu(const Dtype* input,Dtype * outputint ,int N,int C,int H ,int W);
    
private:
    nvinfer1::Dims mCHW;
    DataType mDataType{DataType::kFLOAT};
    float mScale;
    int mOutputWidth;
    int mOutputHeight;
    int mThreadCount;
    
    void* mInputBuffer  {nullptr}; 
    void* mOutputBuffer {nullptr};
};


class UpSamplePluginCreator : public nvinfer1::IPluginCreator {
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

#endif // UPSAMPLE_PLUGIN_HPP