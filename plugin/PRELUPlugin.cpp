/*
 * @Author: zerollzeng
 * @Date: 2019-09-06 15:13:19
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-04 19:28:00
 */
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "utils.h"

#include "plugin/PRELUPlugin.hpp"
#include "plugin/PRELUKernel.cuh"
#include "spdlog/spdlog.h"

using namespace nvinfer1;
using namespace plugin;

const char* PRELU_PLUGIN_VERSION = "01";
const char* PRELU_PLUGIN_TYPE = "PReLUPlugin";
const char* PRELU_PLUGIN_NAMESPACE = "_TRT";
const char* PRELU_PLUGIN_NAME = "PReLUPlugin_TRT";

PReLUPlugin::PReLUPlugin(const Weights *weights, int nbWeights)
{
    spdlog::debug("PReLUPlugin(const Weights *weights, int nbWeights: {})", nbWeights);
    assert(nbWeights == 1);
    mWeights = weights[0];
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), weights[0].values, mWeights.count * type2size(mWeights.type));
}

// create the plugin at runtime from a byte stream
PReLUPlugin::PReLUPlugin(const void *data, size_t length)
{
    spdlog::debug("PReLUPlugin(const void *data, size_t length)");
    const char *d = static_cast<const char *>(data), *a = d;
    read<int>(d, mNbInputChannels);
    read<int>(d, mNbInputHeight);
    read<int>(d, mNbInputWidth);
    read<int>(d, mNbInputCount);
    read<bool>(d, mChannelShared);
    read<int64_t>(d, mWeights.count);
    read<DataType>(d, mDataType);

    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mDataType)); //deserializeToDevice(d,mDeviceKernel,mWeights.count);
    memcpy(const_cast<void *>(mWeights.values), d, mWeights.count * type2size(mDataType));
    deserializeToDevice(d, mDeviceKernel, mWeights.count * type2size(mDataType));

    assert(d == a + length);
}

PReLUPlugin::~PReLUPlugin()
{
    spdlog::debug("~PReLUPlugin()...");
    if (mWeights.values) 
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
    if (mDeviceKernel) 
    {
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}

int PReLUPlugin::getNbOutputs() const
{
    spdlog::debug("getNbOutputs()");
    return 1;
}

Dims PReLUPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    spdlog::debug("getOutputDimensions(int index, const Dims *inputs, int nbInputDims)");
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

bool PReLUPlugin::supportsFormat(DataType type, PluginFormat format) const { 
    spdlog::debug("supportsFormat(DataType type, PluginFormat format)");
    return type == DataType::kFLOAT && format == PluginFormat::kNCHW; }

void PReLUPlugin::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
{
    spdlog::debug("configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)");
    assert(type == DataType::kFLOAT && format == PluginFormat::kNCHW);

    mNbInputChannels = inputDims[0].d[0]; 
    mNbInputHeight = inputDims[0].d[1];
    mNbInputWidth = inputDims[0].d[2];
    mNbInputCount = mNbInputChannels * mNbInputHeight * mNbInputWidth;
    mDataType = type;
}

int PReLUPlugin::initialize()
{
    spdlog::debug("initialize()");
    cudaMalloc(&mDeviceKernel, mWeights.count * type2size(mDataType));
    cudaMemcpy(mDeviceKernel, mWeights.values, mWeights.count * type2size(mDataType), cudaMemcpyHostToDevice);
    return 0;
}

void PReLUPlugin::terminate()
{
    spdlog::debug("terminate()");
    if (mWeights.values)
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
    if (mDeviceKernel)
    {
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}

size_t PReLUPlugin::getWorkspaceSize(int maxBatchSize) const
{
    spdlog::debug("getWorkspaceSize(int maxBatchSize)");
    return 0;
}

int PReLUPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    spdlog::debug("enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)");
    #include <iostream>
    const int count = batchSize * mNbInputCount;
    const int channels = mNbInputChannels;
    const int dim = mNbInputWidth * mNbInputHeight;
    const int div_factor = mChannelShared ? mNbInputChannels : 1; // mChannelShared default is false
    if (mDataType == DataType::kFLOAT)
    {
        const float zerof{0.0f};
        CUDA_CHECK(Forward_gpu(count, channels, dim,
                            reinterpret_cast<const float *>(mDeviceKernel),
                            reinterpret_cast<const float *>(inputs[0]),
                            reinterpret_cast<float *>(outputs[0]),
                            zerof,
                            div_factor,
                            stream));
        float out[10];
        cudaMemcpy(out,outputs[0],sizeof(float) * 10,cudaMemcpyDeviceToHost);
        // std::cout << "fp32: debug-----------" << std::endl;
        // for(int i=0;i<10;i++) {
        //     std::cout << (out[i]) << " ";
        // }
        // std::cout << std::endl;
    } else if(mDataType == DataType::kHALF) // DataType::kHALF
    {
        const __half zeroh = __float2half(0.0f);
        CUDA_CHECK(Forward_gpu<half>(count, channels, dim,
                                  reinterpret_cast<const __half *>(mDeviceKernel),
                                  reinterpret_cast<const __half *>(inputs[0]),
                                  reinterpret_cast<__half *>(outputs[0]),
                                  zeroh,
                                  div_factor,
                                  stream));
        __half out[10];
        cudaMemcpy(out,outputs[0],sizeof(__half) * 10,cudaMemcpyDeviceToHost);
        // std::cout << "fp16: debug 11111-----------" << std::endl;
        // for(int i=0;i<10;i++) {
        //     std::cout << __half2float(out[i]) << " ";
        // }
        // std::cout << std::endl;
    } else {
        std::cerr << "unsupport datatype at: " << __FILE__ << " " << __LINE__ << std::endl;
    }

    return 0;
}

size_t PReLUPlugin::getSerializationSize() const
{
    spdlog::debug("getSerializationSize()");
    return 4 * sizeof(int) + sizeof(bool)
            + sizeof(mWeights.count) + sizeof(mDataType)
            + mWeights.count * type2size(mDataType);
}

void PReLUPlugin::serialize(void *buffer) const
{
    spdlog::debug("serialize(void *buffer)");
    char *d = static_cast<char *>(buffer), *a = d;

    write(d, mNbInputChannels);
    write(d, mNbInputHeight);
    write(d, mNbInputWidth);
    write(d, mNbInputCount);
    write(d, mChannelShared);
    write(d, mWeights.count);
    write(d, mDataType);
    convertAndCopyToBuffer(d, mWeights);
    assert(d == a + getSerializationSize());
}

const char *PReLUPlugin::getPluginType() const
{
    spdlog::debug("getPluginType()");
    return PRELU_PLUGIN_TYPE;
}

const char *PReLUPlugin::getPluginVersion() const
{
    spdlog::debug("getPluginVersion()");
    return PRELU_PLUGIN_VERSION;
}

void PReLUPlugin::destroy() { 
    spdlog::debug("destroy()");
    delete this; }

IPluginV2* PReLUPlugin::clone() const
{
    spdlog::debug("clone()");
    return new PReLUPlugin(&mWeights, 1);
}

// should not implement it.
void PReLUPlugin::setPluginNamespace(const char* pluginNamespace)
{
    spdlog::debug("setPluginNamespace(const char* pluginNamespace): {}",pluginNamespace);
}

const char* PReLUPlugin::getPluginNamespace() const
{
    spdlog::debug("getPluginNamespace()");
    return PRELU_PLUGIN_NAMESPACE;
}

size_t PReLUPlugin::type2size(DataType type) const {
    return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); 
}

void *PReLUPlugin::copyToDevice(const void *data, size_t count) const
{
    void *deviceData;
    CUDA_CHECK(cudaMalloc(&deviceData, count));
    CUDA_CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
    return deviceData;
}

void PReLUPlugin::convertAndCopyToDevice(void *&deviceWeights, const Weights &weights) const
{
    if (weights.type != mDataType) // Weights are converted in host memory first, if the type does not match
    {
        size_t size = weights.count * (mWeights.type == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
        void *buffer = malloc(size);
        for (int64_t v = 0; v < weights.count; ++v)
            if (mWeights.type == DataType::kFLOAT)
                static_cast<float *>(buffer)[v] = __half2float(static_cast<const __half *>(weights.values)[v]);
            else
                static_cast<__half *>(buffer)[v] = __float2half(static_cast<const float *>(weights.values)[v]);

        deviceWeights = copyToDevice(buffer, size);
        free(buffer);
    }
    else
        deviceWeights = copyToDevice(weights.values, weights.count * type2size(mWeights.type));
}

void PReLUPlugin::convertAndCopyToBuffer(char *&buffer, const Weights &weights) const
{
    if (weights.type != mDataType) {
        std::cout << "error: weights type not match" << std::endl;
        return;
    }
    for (int64_t v = 0; v < weights.count; ++v) { 
        if (mDataType == DataType::kFLOAT)
            reinterpret_cast<float *>(buffer)[v] = __half2float(static_cast<const __half *>(weights.values)[v]);
        else
            reinterpret_cast<__half *>(buffer)[v] = __float2half(static_cast<const float *>(weights.values)[v]);
    }   
    buffer += weights.count * type2size(mWeights.type);
}

void PReLUPlugin::deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size) const
{
    deviceWeights = copyToDevice(hostBuffer, size);
    hostBuffer += size;
}


// return PRELU_PLUGIN_TYPE + PRELU_PLUGIN_NAMESPACE
const char* PluginCreator::getPluginName() const { 
    spdlog::debug("PluginCreator::getPluginName()");
    return PRELU_PLUGIN_NAME;
}

const char* PluginCreator::getPluginVersion() const { 
    spdlog::debug("PluginCreator::getPluginVersion()");
    return PRELU_PLUGIN_VERSION;
}
const PluginFieldCollection* PluginCreator::getFieldNames() { 
    spdlog::debug("PluginCreator::getFieldNames()");
    return nullptr;
}
IPluginV2* PluginCreator::createPlugin(const char *layerName, const PluginFieldCollection* fc) {
    spdlog::debug("PluginCreator::createPlugin(const char *layerName, const PluginFieldCollection* fc)");
    return nullptr;
}

// deserialization plugin implementation
IPluginV2* PluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    spdlog::debug("PluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength)");
    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

    if (strName.find("prelu") != std::string::npos)
    {
        return (IPluginV2*)(new PReLUPlugin(serialData, serialLength));
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

void PluginCreator::setPluginNamespace(const char* pluginNamespace) {
    spdlog::debug("PluginCreator::setPluginNamespace(const char* pluginNamespace): {}", pluginNamespace);
    // don't implement it 
}

const char* PluginCreator::getPluginNamespace() const {
    spdlog::debug("PluginCreator::getPluginNamespace()");
    return PRELU_PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(PluginCreator);