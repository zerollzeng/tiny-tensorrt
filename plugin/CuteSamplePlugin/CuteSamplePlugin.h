#ifndef CUTE_SAMPLE_PLUGIN_H
#define CUTE_SAMPLE_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class CuteSamplePlugin : public IPluginV2Ext
{
public:
    CuteSamplePlugin();

    CuteSamplePlugin(const void* data, size_t length);

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override;

    int enqueue(int batchSize, void const * const * inputs, void * const * outputs, void* workspace, cudaStream_t stream) noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
};

class CuteSamplePluginCreator : public IPluginCreator
{
public:
    CuteSamplePluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;

    static std::vector<nvinfer1::PluginField> mPluginAttributes;

    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
