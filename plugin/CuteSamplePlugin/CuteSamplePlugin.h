#ifndef CUTE_SAMPLE_PLUGIN_H
#define CUTE_SAMPLE_PLUGIN_H

#include "NvInferPlugin.h"
#include "NvInferVersion.h"
#include <string>
#include <vector>

#if NV_TENSORRT_MAJOR >= 8
    #define IS_NOEXCEPT noexcept
#else
    #define IS_NOEXCEPT
#endif

namespace nvinfer1
{
namespace plugin
{
class CuteSamplePlugin : public IPluginV2Ext
{
public:
    CuteSamplePlugin();

    CuteSamplePlugin(const void* data, size_t length);

    int getNbOutputs() const IS_NOEXCEPT override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) IS_NOEXCEPT override;

    int initialize() IS_NOEXCEPT override;

    void terminate() IS_NOEXCEPT override;

    size_t getWorkspaceSize(int) const IS_NOEXCEPT override;

#if NV_TENSORRT_MAJOR < 8
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
#else
    int enqueue(int batchSize, void const * const * inputs, void * const * outputs, void* workspace, cudaStream_t stream) noexcept override;
#endif

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const IS_NOEXCEPT override;

    size_t getSerializationSize() const IS_NOEXCEPT override;

    void serialize(void* buffer) const IS_NOEXCEPT override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const IS_NOEXCEPT override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const IS_NOEXCEPT override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) IS_NOEXCEPT override;

    bool supportsFormat(DataType type, PluginFormat format) const IS_NOEXCEPT override;

    const char* getPluginType() const IS_NOEXCEPT override;

    const char* getPluginVersion() const IS_NOEXCEPT override;

    void destroy() IS_NOEXCEPT override;

    IPluginV2Ext* clone() const IS_NOEXCEPT override;

    void setPluginNamespace(const char* libNamespace) IS_NOEXCEPT override;

    const char* getPluginNamespace() const IS_NOEXCEPT override;

private:
    std::string mNamespace;
};

class CuteSamplePluginCreator : public IPluginCreator
{
public:
    CuteSamplePluginCreator();

    const char* getPluginName() const IS_NOEXCEPT override;

    const char* getPluginVersion() const IS_NOEXCEPT override;

    const PluginFieldCollection* getFieldNames() IS_NOEXCEPT override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) IS_NOEXCEPT override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT override;

    void setPluginNamespace(const char* libNamespace) IS_NOEXCEPT override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const IS_NOEXCEPT override
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
