/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */

#include "CuteSamplePlugin.h"
#include <cassert>
#include <iostream>

#ifndef CUTEDEBUG 
#define CUTEDEBUG 1 // set debug mode
#endif

#if CUTEDEBUG
#define cutelog(...) {\
    char str[100];\
    sprintf(str, __VA_ARGS__);\
    std::cout << " (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call " << "[" << __FILE__ << "][" \
              << __FUNCTION__ << "][Line " << __LINE__ << "] " << str << std::endl;\
    }
#else
#define cutelog(...)
#endif

using namespace nvinfer1;
using nvinfer1::plugin::CuteSamplePlugin;
using nvinfer1::plugin::CuteSamplePluginCreator;

static const char* CUTE_PLUGIN_VERSION{"1"};
static const char* CUTE_PLUGIN_NAME{"CuteSamplePlugin"};

PluginFieldCollection CuteSamplePluginCreator::mFC{};

CuteSamplePlugin::CuteSamplePlugin(const std::string name)
    : mLayerName(name)
{
    cutelog("wow I run to here now");
}

CuteSamplePlugin::CuteSamplePlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    cutelog("wow I run to here now");
}

int CuteSamplePlugin::getNbOutputs() const
{
    cutelog("wow I run to here now");
    return 1;
}

Dims CuteSamplePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    cutelog("wow I run to here now");
    return Dims3(inputs[1].d[1], inputs[1].d[2], inputs[1].d[3]);
}

int CuteSamplePlugin::initialize()
{
    cutelog("wow I run to here now");
    return 0;
}

size_t CuteSamplePlugin::getWorkspaceSize(int) const
{
    cutelog("wow I run to here now");
    return 0;
}

DataType CuteSamplePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    cutelog("wow I run to here now");
    return DataType::kFLOAT;
}

int CuteSamplePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    cutelog("wow I run to here now");
    return 0;
}

void CuteSamplePlugin::serialize(void* buffer) const
{
    cutelog("wow I run to here now");
}

void CuteSamplePlugin::terminate() {
    cutelog("wow I run to here now");
}

size_t CuteSamplePlugin::getSerializationSize() const
{
    cutelog("wow I run to here now");
    return 0;
}

bool CuteSamplePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    cutelog("wow I run to here now");
    return false;
}

bool CuteSamplePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    cutelog("wow I run to here now");
    return false;
}

void CuteSamplePlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    cutelog("wow I run to here now");
}

bool CuteSamplePlugin::supportsFormat(DataType type, PluginFormat format) const
{
    cutelog("wow I run to here now");
    return true;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePlugin::getPluginType() const
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePlugin::getPluginVersion() const
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_VERSION;
}

void CuteSamplePlugin::destroy()
{
    cutelog("wow I run to here now");
    delete this;
}

IPluginV2Ext* CuteSamplePlugin::clone() const
{
    auto* plugin = new CuteSamplePlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/**
 * NO NEED TO MODIFY
 */
void CuteSamplePlugin::setPluginNamespace(const char* libNamespace)
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePlugin::getPluginNamespace() const
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
const char* CuteSamplePluginCreator::getPluginName() const
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* CuteSamplePluginCreator::getPluginVersion() const
{
    cutelog("wow I run to here now");
    return CUTE_PLUGIN_VERSION;
}

/**
 * NO NEED TO MODIFY
 */
const PluginFieldCollection* CuteSamplePluginCreator::getFieldNames()
{
    cutelog("wow I run to here now");
    return &mFC;
}

IPluginV2Ext* CuteSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    cutelog("wow I run to here now");
    auto* plugin = new CuteSamplePlugin(name);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* CuteSamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    cutelog("wow I run to here now");
    return new CuteSamplePlugin(name, serialData, serialLength);
}

// if you want to enable the plugin, please uncomment this line
// I comment it for surpress cutelogs
// REGISTER_TENSORRT_PLUGIN(CuteSamplePluginCreator);