#include <cstring>
#include <vector>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "plugin_utils.h"
#include "PReLUPlugin.h"

#include "spdlog/spdlog.h"

// for consistency I recommend all plugin have same namesapce and version
// 为了一致性,所有的插件都建议使用相同的namespace和version
const char* G_PLUGIN_NAMESPACE = "_TRT";
const char* G_PLUGIN_VERSION = "1";

const char* G_PRELU_TYPE = "PReLU";
const char* G_PRELU_NAME = "PReLU_TRT"; //plugin_name = plugin_type + plugin_namespace

// CUDA: use 512 threads per block
static const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// /******** PReLU CUDA function ********/
// CUDA kernele for forward
template <typename Ftype>
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const Ftype* slope_data,
    const Ftype* in, Ftype* out,
    const Ftype zero,
    const int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels / div_factor;
        if(in[index] > zero) {
            out[index] = in[index];
        } else {
            out[index] = in[index] * slope_data[c];
        }
    }
}

template <typename Ftype>
cudaError_t Forward_gpu(const int count, const int channels, const int dim,
                const Ftype* mDeviceKernel,
                const Ftype* bottom_data, Ftype* top_data, 
                const Ftype zero,
                const int div_factor, const cudaStream_t stream) {
    PReLUForward<<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
        (count, channels, dim, mDeviceKernel, bottom_data, top_data, zero, div_factor);
    cudaError_t err = cudaGetLastError();
    return err;
}

PReLUPlugin::PReLUPlugin(const nvinfer1::Weights *weights, int nbWeights) {
    mWeights = weights[0];
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), weights[0].values, mWeights.count * type2size(mWeights.type));
}

// create the plugin at runtime from a byte stream
PReLUPlugin::PReLUPlugin(const void *data, size_t length) {
    const char *d = static_cast<const char *>(data), *a = d;
    read<int>(d, mNbInputChannels);
    read<int>(d, mNbInputHeight);
    read<int>(d, mNbInputWidth);
    read<nvinfer1::DataType>(d, mDataType);
    read<int64_t>(d, mWeights.count);
    read<nvinfer1::DataType>(d, mWeights.type);
    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    d = d + mWeights.count * type2size(mWeights.type);
    ASSERT(d == a + length);
}

size_t PReLUPlugin::getSerializationSize() const {
    return sizeof(mNbInputChannels) + sizeof(mNbInputWidth) + sizeof(mNbInputHeight) + sizeof(mDataType) + 
           sizeof(mWeights.count) + sizeof(mWeights.type) + mWeights.count * type2size(mWeights.type);
}

void PReLUPlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer), *a = d;
    write(d, mNbInputChannels);
    write(d, mNbInputHeight);
    write(d, mNbInputWidth);
    write(d, mDataType);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d, mWeights, mWeights.type);
    ASSERT(d == a + getSerializationSize());
}

PReLUPlugin::~PReLUPlugin() {
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

int PReLUPlugin::getNbOutputs() const {
    return 1;
}

nvinfer1::Dims PReLUPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) {
    if(index == 0) {
        return nvinfer1::Dims3(inputs[0].d[0],inputs[0].d[1],inputs[0].d[2]);
    } // else if(index == n) {
        // for other outputs if exists.
    // }
    else {
        ASSERT(false);
    }
}

bool PReLUPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) 
            && format == nvinfer1::PluginFormat::kNCHW;
}

void PReLUPlugin::configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, 
                                      const nvinfer1::Dims* outputDims, int nbOutputs,
                                      nvinfer1::DataType type, nvinfer1::PluginFormat format, 
                                      int maxBatchSize) {
    ASSERT((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
            && format == nvinfer1::PluginFormat::kNCHW);
    mNbInputChannels = inputDims[0].d[0]; 
    mNbInputHeight = inputDims[0].d[1];
    mNbInputWidth = inputDims[0].d[2];
    mDataType = type;
}

int PReLUPlugin::initialize() {
    convertAndCopyToDeivce(mDeviceKernel, mWeights, mDataType);
    return 0;
}

void PReLUPlugin::terminate() {
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
    return 0;
}

int PReLUPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    const int count = batchSize * mNbInputChannels * mNbInputWidth * mNbInputHeight;
    const int channels = mNbInputChannels;
    const int dim = mNbInputWidth * mNbInputHeight;
    const int div_factor = 1;
    if (mDataType == nvinfer1::DataType::kFLOAT)
    {
        const float zerof{0.0f};
        CUDA_CHECK(Forward_gpu(count, channels, dim,
                            reinterpret_cast<const float *>(mDeviceKernel),
                            reinterpret_cast<const float *>(inputs[0]),
                            reinterpret_cast<float *>(outputs[0]),
                            zerof,
                            div_factor,
                            stream));
    }
#ifdef FP16_PRELU
    else 
    {
        const __half zeroh = __half(0.0f);
        CUDA_CHECK(Forward_gpu(count, channels, dim,
                            reinterpret_cast<const __half *>(mDeviceKernel),
                            reinterpret_cast<const __half *>(inputs[0]),
                            reinterpret_cast<__half *>(outputs[0]),
                            zeroh,
                            div_factor,
                            stream));
    }
#else
    else
    {
        spdlog::error("fp16 prelu is unsupported");
        ASSERT(false);
    }
#endif
    return 0;
}

const char *PReLUPlugin::getPluginType() const {
    return G_PRELU_TYPE;
}

const char *PReLUPlugin::getPluginVersion() const {
    return G_PLUGIN_VERSION;
}

void PReLUPlugin::destroy() {
    delete this; 
}

nvinfer1::IPluginV2* PReLUPlugin::clone() const {
    return new PReLUPlugin(&mWeights, 1);
}

const char* PReLUPlugin::getPluginNamespace() const {
    return G_PLUGIN_NAMESPACE;
}

PReLUPluginCreator::PReLUPluginCreator()  {
    mPluginAttributes.emplace_back(nvinfer1::PluginField("weights", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nbWeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

// return PRELU_PLUGIN_TYPE + PRELU_PLUGIN_NAMESPACE
const char* PReLUPluginCreator::getPluginName() const {
    // std::string plugin_type{G_PRELU_TYPE};
    // std::string plugin_namespace{G_PLUGIN_NAMESPACE};
    // return (plugin_type+plugin_namespace).c_str();
    return G_PRELU_NAME;
}

const char* PReLUPluginCreator::getPluginVersion() const {
    return G_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* PReLUPluginCreator::getFieldNames() {
    return &mFC;
}

nvinfer1::IPluginV2* PReLUPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
    int nbWeights;
    std::vector<float> weightValues;
    const nvinfer1::PluginField* fields = fc->fields;
    for (int i=0; i<fc->nbFields; i++) {
        const char* attrName = fields[i].name;
        if(strcmp(attrName, "nbWeights")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            nbWeights = *(static_cast<const int*>(fields[i].data));
        }
        if(strcmp(attrName, "weights")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            weightValues.reserve(fields[i].length);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < weightValues.size(); j++)
            {
                weightValues.push_back(*w);
                w++;
            }
        }
    }
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, weightValues.data(), (int64_t)weightValues.size()};
    return new PReLUPlugin(&weights,nbWeights);
}

// deserialization plugin implementation
nvinfer1::IPluginV2* PReLUPluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength) {
    return new PReLUPlugin(serialData, serialLength);
}

const char* PReLUPluginCreator::getPluginNamespace() const {
    return G_PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(PReLUPluginCreator); // DO NOT FORGET THIS
                                              // 别忘了这个
