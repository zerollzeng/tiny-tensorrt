#ifndef CUTE_SAMPLE_DYNAMIC_EXT_PLUGIN_H
#define CUTE_SAMPLE_DYNAMIC_EXT_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class CuteSampleDynamicExtPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:

    CuteSampleDynamicExtPlugin();

    CuteSampleDynamicExtPlugin(const void* data, size_t length);

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, 
        void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
        int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    int getNbOutputs() const noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
};

class CuteSampleDynamicExtPluginCreator : public nvinfer1::IPluginCreator
{
public:
    CuteSampleDynamicExtPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;

    static std::vector<nvinfer1::PluginField> mPluginAttributes;

    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif