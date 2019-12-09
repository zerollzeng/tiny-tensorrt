#include "UpSamplePlugin.hpp"
#include "plugin_utils.h"
#include "spdlog/spdlog.h"

#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <cassert>

const int NUM_THREADS = 512;

const char* UPSAMPLE_PLUGIN_VERSION = "01";
const char* UPSAMPLE_PLUGIN_TYPE = "UpSamplePlugin";
const char* UPSAMPLE_PLUGIN_NAMESPACE = "_TRT";
const char* UPSAMPLE_PLUGIN_NAME = "UpSamplePlugin_TRT";

UpSamplePlugin::UpSamplePlugin(const float scale) : mScale{scale} {
}

UpSamplePlugin::UpSamplePlugin(const void* data, size_t length) {
    const char *d = reinterpret_cast<const char *>(data), *a = d;
      read(d, mCHW);
      read(d, mDataType);
      read(d, mScale);
      read(d, mOutputWidth);
      read(d, mOutputHeight);
      read(d, mThreadCount);
  
      //std::cout << "read:" << a << " " << mOutputWidth<< " " <<mOutputHeight<<std::endl;
      assert(d == a + length);
}

UpSamplePlugin::~UpSamplePlugin() {
}

Dims UpSamplePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    // spdlog::info("getOutputDimensions...");
    // spdlog::info("input dimemsion: {},{},{}",inputs[0].d[0],inputs[0].d[1],inputs[0].d[2]);
    // spdlog::info("index:{}, nbInputDims:{}, mScale:{}",index,nbInputDims,mScale);

    // mCHW = inputs[0];
    // mOutputHeight = (int)(inputs[0].d[1]* mScale);
    // mOutputWidth = (int)(inputs[0].d[2]* mScale);
    //std::cout << "ouputDims:" << mCHW.d[0] << " " << mOutputHeight << " " << mOutputWidth << std::endl;
    return Dims3(inputs[0].d[0], (int)(inputs[0].d[1]* mScale), (int)(inputs[0].d[2]* mScale));
}

void UpSamplePlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) {
    //std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
    assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW);

    mCHW = inputDims[0];
    mOutputWidth = outputDims[0].d[1];
    mOutputHeight = outputDims[0].d[2];
    mDataType = type;
    
    //std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " " <<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
}

int UpSamplePlugin::initialize() {
    int inputHeight = mCHW.d[1];
    int inputWidth = mCHW.d[2];
    
    mOutputHeight = (int)(inputHeight * mScale);
    mOutputWidth = (int)(inputWidth * mScale);

    return 0;
}

void UpSamplePlugin::terminate() {
    // WARNING: not implement?
}

void UpSamplePlugin::serialize(void* buffer) const{
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, mCHW);
    write(d, mDataType);
    write(d, mScale);
    write(d, mOutputWidth);
    write(d, mOutputHeight);
    write(d, mThreadCount);

    //std::cout << "write:" << a << " " << mOutputHeight<< " " <<mOutputWidth<<std::endl;
    assert(d == a + getSerializationSize());
}

const char* UpSamplePlugin::getPluginType() const{
    return UPSAMPLE_PLUGIN_TYPE;
}

const char* UpSamplePlugin::getPluginVersion() const{
    return UPSAMPLE_PLUGIN_VERSION;
}

void UpSamplePlugin::destroy() { 
    delete this;
}

IPluginV2* UpSamplePlugin::clone() const{
    return new UpSamplePlugin(mScale);
}

void UpSamplePlugin::setPluginNamespace(const char* pluginNamespace) {

}

const char* UpSamplePlugin::getPluginNamespace() const{
    return UPSAMPLE_PLUGIN_NAMESPACE;
}


// return UPSAMPLE_PLUGIN_TYPE + UPSAMPLE_PLUGIN_NAMESPACE
const char* UpSamplePluginCreator::getPluginName() const { 
    return UPSAMPLE_PLUGIN_NAME;
}

const char* UpSamplePluginCreator::getPluginVersion() const { 
    return UPSAMPLE_PLUGIN_VERSION;
}
const PluginFieldCollection* UpSamplePluginCreator::getFieldNames() { 
    return nullptr;
}
IPluginV2* UpSamplePluginCreator::createPlugin(const char *layerName, const PluginFieldCollection* fc) {
    return nullptr;
}

// deserialization plugin implementation
IPluginV2* UpSamplePluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

    if (strName.find("upsample") != std::string::npos)
    {
        return (IPluginV2*)(new UpSamplePlugin(serialData, serialLength));
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

void UpSamplePluginCreator::setPluginNamespace(const char* pluginNamespace) {
    // don't implement it 
}

const char* UpSamplePluginCreator::getPluginNamespace() const {
    return UPSAMPLE_PLUGIN_NAMESPACE;
}

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
int x, y, z, w;
w = ii % d3;
ii = ii/d3;
z = ii % d2;
ii = ii/d2;
y = ii % d1;
ii = ii/d1;
x = ii;
w = w/scale_factor;
z = z/scale_factor;
d2 /= scale_factor;
d3 /= scale_factor;
return (((x*d1+y)*d2)+z)*d3+w;
}

template <typename Dtype>
__global__ void upscale(const Dtype *input, Dtype *output,
        int no_elements, int scale_factor, int d1, int d2, int d3) {
    int ii = threadIdx.x + blockDim.x * blockIdx.x;
    if (ii >= no_elements) return;
    int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
    output[ii]=input[ipidx];
}

template <typename Dtype>
void UpSamplePlugin::forwardGpu(const Dtype* input,Dtype * output,
    int N,int C,int H ,int W) {

int numElem = N*C*H*W;
upscale<<<(numElem + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(input,output, numElem, mScale, C, H, W);
}

int UpSamplePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const int channels = mCHW.d[0];
    const int64_t in_height = mCHW.d[1];
    const int64_t in_width = mCHW.d[2];
    const int64_t out_height = mOutputHeight;
    const int64_t out_width = mOutputWidth;
    int totalElems = batchSize * in_height * in_width * channels;
    
    // Handle no-op resizes efficiently.
    if (out_height == in_height && out_width == in_width) {
        CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], totalElems * type2size(mDataType), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return 0;
    }
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    
    switch (mDataType)
    {
        case DataType::kFLOAT :
            forwardGpu<float>((const float *)inputs[0],(float *)outputs[0],batchSize,mCHW.d[0],mOutputHeight,mOutputWidth);
            break;
        case DataType::kHALF:
            forwardGpu<__half>((const __half *)inputs[0],(__half *)outputs[0],batchSize,mCHW.d[0],mOutputHeight,mOutputWidth);
            break;
        case DataType::kINT8:
            forwardGpu<u_int8_t>((const u_int8_t *)inputs[0],(u_int8_t *)outputs[0],batchSize,mCHW.d[0],mOutputHeight,mOutputWidth);
            break;
        default:
            std::cerr << "error data type" << std::endl;
    }
    return 0;    
};

REGISTER_TENSORRT_PLUGIN(UpSamplePluginCreator);