#ifndef CUTE_SAMPLE_DYNAMIC_EXT_PLUGIN_H
#define CUTE_SAMPLE_DYNAMIC_EXT_PLUGIN_H

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

class CuteSampleDynamicExtPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:

    CuteSampleDynamicExtPlugin();

    CuteSampleDynamicExtPlugin(const void* data, size_t length);

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const IS_NOEXCEPT override;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) IS_NOEXCEPT override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) IS_NOEXCEPT override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) IS_NOEXCEPT override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const IS_NOEXCEPT override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, 
        void* workspace, cudaStream_t stream) IS_NOEXCEPT override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
        int nbInputs) const IS_NOEXCEPT override;

    // IPluginV2 Methods
    const char* getPluginType() const IS_NOEXCEPT override;

    const char* getPluginVersion() const IS_NOEXCEPT override;

    int getNbOutputs() const IS_NOEXCEPT override;

    int initialize() IS_NOEXCEPT override;

    void terminate() IS_NOEXCEPT override;

    size_t getSerializationSize() const IS_NOEXCEPT override;

    void serialize(void* buffer) const IS_NOEXCEPT override;

    void destroy() IS_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) IS_NOEXCEPT override;

    const char* getPluginNamespace() const IS_NOEXCEPT override;

private:
    std::string mNamespace;
};

class CuteSampleDynamicExtPluginCreator : public nvinfer1::IPluginCreator
{
public:
    CuteSampleDynamicExtPluginCreator();

    const char* getPluginName() const IS_NOEXCEPT override;

    const char* getPluginVersion() const IS_NOEXCEPT override;

    const nvinfer1::PluginFieldCollection* getFieldNames() IS_NOEXCEPT override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) IS_NOEXCEPT override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) IS_NOEXCEPT override;

    const char* getPluginNamespace() const IS_NOEXCEPT override;

private:
    static nvinfer1::PluginFieldCollection mFC;

    static std::vector<nvinfer1::PluginField> mPluginAttributes;

    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif