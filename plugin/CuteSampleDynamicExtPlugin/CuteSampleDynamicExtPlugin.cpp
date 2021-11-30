/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */

#ifndef CUTEDEBUG
#define CUTEDEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#include "CuteSampleDynamicExtPlugin.h"
#include "plugin_utils.h"
#include <cassert>

using namespace nvinfer1;
using nvinfer1::plugin::CuteSampleDynamicExtPlugin;
using nvinfer1::plugin::CuteSampleDynamicExtPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"CuteSampleDynamicExtPlugin"};

// Static class fields initialization
PluginFieldCollection CuteSampleDynamicExtPluginCreator::mFC{};
std::vector<PluginField> CuteSampleDynamicExtPluginCreator::mPluginAttributes;

CuteSampleDynamicExtPlugin::CuteSampleDynamicExtPlugin()
{
    cutelog("wow I run to here now");
}

CuteSampleDynamicExtPlugin::CuteSampleDynamicExtPlugin(const void* data, size_t length)
{
    cutelog("wow I run to here now");
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* CuteSampleDynamicExtPlugin::clone() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    auto* plugin = new CuteSampleDynamicExtPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs CuteSampleDynamicExtPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return inputs[0];
}

bool CuteSampleDynamicExtPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) IS_NOEXCEPT
{

    cutelog("wow I run to here now");
    return true;
}

void CuteSampleDynamicExtPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
}

size_t CuteSampleDynamicExtPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 0;
}
int CuteSampleDynamicExtPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 1;
}

// IPluginV2Ext Methods
nvinfer1::DataType CuteSampleDynamicExtPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return inputTypes[0];
}

// IPluginV2 Methods

const char* CuteSampleDynamicExtPlugin::getPluginType() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return PLUGIN_NAME;
}

const char* CuteSampleDynamicExtPlugin::getPluginVersion() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return PLUGIN_VERSION;
}

int CuteSampleDynamicExtPlugin::getNbOutputs() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 1;
}

int CuteSampleDynamicExtPlugin::initialize() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 0;
}

void CuteSampleDynamicExtPlugin::terminate() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
}

size_t CuteSampleDynamicExtPlugin::getSerializationSize() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 0;
}

void CuteSampleDynamicExtPlugin::serialize(void* buffer) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
}

void CuteSampleDynamicExtPlugin::destroy() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    delete this;
}

void CuteSampleDynamicExtPlugin::setPluginNamespace(const char* libNamespace) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

const char* CuteSampleDynamicExtPlugin::getPluginNamespace() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

///////////////

CuteSampleDynamicExtPluginCreator::CuteSampleDynamicExtPluginCreator()
{
    cutelog("wow I run to here now");
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CuteSampleDynamicExtPluginCreator::getPluginName() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return PLUGIN_NAME;
}

const char* CuteSampleDynamicExtPluginCreator::getPluginVersion() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return PLUGIN_VERSION;
}

const PluginFieldCollection* CuteSampleDynamicExtPluginCreator::getFieldNames() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return &mFC;
}

IPluginV2* CuteSampleDynamicExtPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return new CuteSampleDynamicExtPlugin();
}

IPluginV2* CuteSampleDynamicExtPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return new CuteSampleDynamicExtPlugin(serialData, serialLength);
}

void CuteSampleDynamicExtPluginCreator::setPluginNamespace(const char* libNamespace) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

const char* CuteSampleDynamicExtPluginCreator::getPluginNamespace() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(CuteSampleDynamicExtPluginCreator);
