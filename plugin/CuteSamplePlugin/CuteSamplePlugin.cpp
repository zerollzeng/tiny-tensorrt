/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */

#ifndef CUTEDEBUG
#define CUTEDEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#include "CuteSamplePlugin.h"
#include "plugin_utils.h"
#include <cassert>

using namespace nvinfer1;
using nvinfer1::plugin::CuteSamplePlugin;
using nvinfer1::plugin::CuteSamplePluginCreator;

static const char* CUTE_PLUGIN_VERSION{"1"};
static const char* CUTE_PLUGIN_NAME{"CuteSamplePlugin"};

PluginFieldCollection CuteSamplePluginCreator::mFC{};
std::vector<PluginField> CuteSamplePluginCreator::mPluginAttributes;

CuteSamplePlugin::CuteSamplePlugin()
{
    cutelog("wow I run to here now");
}

CuteSamplePlugin::CuteSamplePlugin(const void* data, size_t length)
{
    cutelog("wow I run to here now");
}

int CuteSamplePlugin::getNbOutputs() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 1;
}

Dims CuteSamplePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return Dims3(inputs[1].d[1], inputs[1].d[2], inputs[1].d[3]);
}

int CuteSamplePlugin::initialize() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 0;
}

size_t CuteSamplePlugin::getWorkspaceSize(int) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 0;
}

DataType CuteSamplePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return DataType::kFLOAT;
}

#if NV_TENSORRT_MAJOR < 8
int CuteSamplePlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
#else
int CuteSamplePlugin::enqueue(int batchSize, void const * const * inputs, void * const * outputs, void* workspace, cudaStream_t stream) noexcept
#endif
{
    cutelog("wow I run to here now");
    return 0;
}

void CuteSamplePlugin::serialize(void* buffer) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
}

void CuteSamplePlugin::terminate() IS_NOEXCEPT {
    cutelog("wow I run to here now");
}

size_t CuteSamplePlugin::getSerializationSize() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return 0;
}

bool CuteSamplePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return false;
}

bool CuteSamplePlugin::canBroadcastInputAcrossBatch(int inputIndex) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return false;
}

void CuteSamplePlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
}

bool CuteSamplePlugin::supportsFormat(DataType type, PluginFormat format) const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return true;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePlugin::getPluginType() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePlugin::getPluginVersion() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_VERSION;
}

void CuteSamplePlugin::destroy() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    delete this;
}

IPluginV2Ext* CuteSamplePlugin::clone() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    auto* plugin = new CuteSamplePlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/**
 * NO NEED TO MODIFY
 */
void CuteSamplePlugin::setPluginNamespace(const char* libNamespace) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePlugin::getPluginNamespace() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

CuteSamplePluginCreator::CuteSamplePluginCreator()
{
    cutelog("wow I run to here now");
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePluginCreator::getPluginName() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePluginCreator::getPluginVersion() const IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_VERSION;
}

/**
 * NO NEED TO MODIFY
 */
const PluginFieldCollection* CuteSamplePluginCreator::getFieldNames() IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return &mFC;
}

IPluginV2Ext* CuteSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    auto* plugin = new CuteSamplePlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* CuteSamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT
{
    cutelog("wow I run to here now");
    return new CuteSamplePlugin(serialData, serialLength);
}

REGISTER_TENSORRT_PLUGIN(CuteSamplePluginCreator);
