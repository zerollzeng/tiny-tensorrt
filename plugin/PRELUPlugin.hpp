#ifndef _PLUGIN_PRELU_H_
#define _PLUGIN_PRELU_H_

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


class PReLUPlugin : public nvinfer1::IPluginV2
{
public:
    PReLUPlugin(const Weights *weights, int nbWeights);

    // create the plugin at runtime from a byte stream
    PReLUPlugin(const void *data, size_t length);

    ~PReLUPlugin();
    virtual int getNbOutputs() const override;

    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(DataType type, PluginFormat format) const override;

    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;
    
    virtual int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override;
    
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    
    virtual size_t getSerializationSize() const override;

    virtual void serialize(void* buffer) const override;

    virtual const char* getPluginType() const override;
    
    virtual const char* getPluginVersion() const override;
    
    virtual void destroy();
    
    virtual IPluginV2* clone() const override;

    virtual void setPluginNamespace(const char* pluginNamespace) override;
    
    virtual const char* getPluginNamespace() const override;
    
private:
    size_t type2size(DataType type) const;

    void *copyToDevice(const void *data, size_t count) const;
    void convertAndCopyToDevice(void *&deviceWeights, const Weights &weights) const;

    void convertAndCopyToBuffer(char *&buffer, const Weights &weights) const;

    void deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size) const;
    
    int mNbInputChannels, mNbInputHeight, mNbInputWidth, mNbInputCount;
    bool mChannelShared = false;
    Weights mWeights;
    DataType mDataType{DataType::kFLOAT};

    void* mDeviceKernel{nullptr};
    char mPluginNamespace;

    void dump(const char* filename, void* memblock, size_t size);
};


class PluginCreator : public nvinfer1::IPluginCreator {
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



#endif // _PLUGIN_PRELU_H_