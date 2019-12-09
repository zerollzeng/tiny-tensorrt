#include "YoloLayerPlugin.hpp"
#include "plugin_utils.h"
#include "spdlog/spdlog.h"
#include <cassert>

const char* YOLOLAYER_PLUGIN_VERSION = "01";
const char* YOLOLAYER_PLUGIN_TYPE = "YoloLayerPlugin";
const char* YOLOLAYER_PLUGIN_NAMESPACE = "_TRT";
const char* YOLOLAYER_PLUGIN_NAME = "YoloLayerPlugin_TRT";

//YOLO 608
YoloKernel yolo1_608 = {
    19,
    19,
    {116,90,  156,198,  373,326}
};
YoloKernel yolo2_608 = {
    38,
    38,
    {30,61,  62,45,  59,119}
};
YoloKernel yolo3_608 = {
    76,
    76,
    {10,13,  16,30,  33,23}
};

// YOLO 416
YoloKernel yolo1_416 = {
    13,
    13,
    {116,90,  156,198,  373,326}
};
YoloKernel yolo2_416 = {
    26,
    26,
    {30,61,  62,45,  59,119}
};
YoloKernel yolo3_416 = {
    52,
    52,
    {10,13,  16,30,  33,23}
};

using namespace nvinfer1;

YoloLayerPlugin::YoloLayerPlugin(int classCount, int netSize)
{
    mClassCount = classCount;
    mYolo3NetSize = netSize;
    mYoloKernel.clear();
    switch(netSize) {
    case 608:
        mYoloKernel.push_back(yolo1_608);
        mYoloKernel.push_back(yolo2_608);
        mYoloKernel.push_back(yolo3_608);
        break;
    case 416:
        mYoloKernel.push_back(yolo1_416);
        mYoloKernel.push_back(yolo2_416);
        mYoloKernel.push_back(yolo3_416);
        break;
    default:
        spdlog::error("error: unsupport netSize, make sure it's 416 or 608");
    }
    mKernelCount = mYoloKernel.size();
}

// create the plugin at runtime from a byte stream
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    read(d, mClassCount);
    read(d, mYolo3NetSize);
    read(d, mKernelCount);
    mYoloKernel.resize(mKernelCount);
    auto kernelSize = mKernelCount*sizeof(YoloKernel);
    memcpy(mYoloKernel.data(),d,kernelSize);
    d += kernelSize;

    assert(d == a + length);
}

YoloLayerPlugin::~YoloLayerPlugin()
{
    if(mInputBuffer)
        CUDA_CHECK(cudaFreeHost(mInputBuffer));

    if(mOutputBuffer)
        CUDA_CHECK(cudaFreeHost(mOutputBuffer));
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    //output the result to channel
    int totalCount = 0;
    for(const auto& yolo : mYoloKernel)
        totalCount += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);

    return Dims3(totalCount + 1, 1, 1);
}

int YoloLayerPlugin::initialize()
{ 
    int totalCount = 0;
    for(const auto& yolo : mYoloKernel)
        totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
    CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

    totalCount = 0;//detection count
    for(const auto& yolo : mYoloKernel)
        totalCount += yolo.width*yolo.height * CHECK_COUNT;
    CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection), cudaHostAllocDefault));
    return 0;
}

int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    //assert(batchSize == 1);
    //GPU
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    forwardGpu((const float *const *)inputs,(float *)outputs[0],stream,batchSize);

    //CPU
    // forwardCpu((const float *const *)inputs,(float *)outputs[0],stream,batchSize);
    return 0;
}

size_t YoloLayerPlugin::getSerializationSize() const
{  
    return sizeof(mClassCount) + sizeof(mYolo3NetSize) + sizeof(mKernelCount) + sizeof(YoloKernel) * mYoloKernel.size();
}

void YoloLayerPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mYolo3NetSize);
    write(d, mKernelCount);
    auto kernelSize = mKernelCount*sizeof(YoloKernel);
    memcpy(d,mYoloKernel.data(),kernelSize);
    d += kernelSize;

    assert(d == a + getSerializationSize());
}

const char* YoloLayerPlugin::getPluginType() const {
    return YOLOLAYER_PLUGIN_TYPE;
}

const char* YoloLayerPlugin::getPluginVersion() const {
    return YOLOLAYER_PLUGIN_VERSION;
}

void YoloLayerPlugin::destroy() {
    delete this;
}

IPluginV2* YoloLayerPlugin::clone() const {
    return new YoloLayerPlugin(mClassCount,mYolo3NetSize);
}

const char* YoloLayerPlugin::getPluginNamespace() const {
    return YOLOLAYER_PLUGIN_NAMESPACE;
}

void YoloLayerPlugin::forwardCpu(const float*const * inputs, float* outputs, cudaStream_t stream,int batchSize)
{
    auto Logist = [=](float data){
        return 1./(1. + exp(-data));
    };

    int totalOutputCount = 0;
        int i = 0;
    int totalCount = 0;
        for(const auto& yolo : mYoloKernel)
        {
        totalOutputCount += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);
        totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
        ++ i;
    }

    for (int idx = 0; idx < batchSize;idx++)
    {
        i = 0;
        float* inputData = (float *)mInputBuffer;// + idx *totalCount; //if create more batch size
        for(const auto& yolo : mYoloKernel)
        {
            int size = (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
            CUDA_CHECK(cudaMemcpyAsync(inputData, (float *)inputs[i] + idx * size, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
            inputData += size;
            ++ i;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        inputData = (float *)mInputBuffer ;//+ idx *totalCount; //if create more batch size
        std::vector <Detection> result;
        for (const auto& yolo : mYoloKernel)
        {
            int stride = yolo.width*yolo.height;
            for (int j = 0;j < stride ;++j)
            {
                for (int k = 0;k < CHECK_COUNT; ++k )
                {
                    int beginIdx = (LOCATIONS + 1 + mClassCount)* stride *k + j;
                    int objIndex = beginIdx + LOCATIONS*stride;
                    
                    //check obj
                    float objProb = Logist(inputData[objIndex]);   
                    if(objProb <= IGNORE_THRESH)
                        continue;

                    //classes
                    int classId = -1;
                    float maxProb = IGNORE_THRESH;
                    for (int c = 0;c< mClassCount;++c){
                        float cProb =  Logist(inputData[beginIdx + (5 + c) * stride]) * objProb;
                        if(cProb > maxProb){
                            maxProb = cProb;
                            classId = c;
                        }
                    }
        
                    if(classId >= 0) {
                        Detection det;
                        int row = j / yolo.width;
                        int cols = j % yolo.width;

                        //Location
                        det.bbox[0] = (cols + Logist(inputData[beginIdx]))/ yolo.width;
                        det.bbox[1] = (row + Logist(inputData[beginIdx+stride]))/ yolo.height;
                        det.bbox[2] = exp(inputData[beginIdx+2*stride]) * yolo.anchors[2*k];
                        det.bbox[3] = exp(inputData[beginIdx+3*stride]) * yolo.anchors[2*k + 1];
                        det.classId = classId;
                        det.prob = maxProb;

                        result.emplace_back(det);
                    }
                }
            }

            inputData += (LOCATIONS + 1 + mClassCount) * stride * CHECK_COUNT;
        }

        
        int detCount =result.size();
        auto data = (float *)mOutputBuffer;// + idx*(totalOutputCount + 1); //if create more batch size
        float * begin = data;
        //copy count;
        data[0] = (float)detCount;
        data++;
        //copy result
        memcpy(data,result.data(),result.size()*sizeof(Detection));

        //(count + det result)
        CUDA_CHECK(cudaMemcpyAsync(outputs, begin,sizeof(float) + result.size()*sizeof(Detection), cudaMemcpyHostToDevice, stream));

        outputs += totalOutputCount + 1;
    }
}

__device__ float Logist(float data){ return 1./(1. + exp(-data)); };

__global__ void CalDetection(const float *input, float *output,int noElements, 
        int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= noElements) return;

    int stride = yoloWidth*yoloHeight;
    int bnIdx =  idx / stride;

    int curIdx = idx - stride*bnIdx;

    const float* curInput = input + bnIdx* ((LOCATIONS + 1 + classes) * stride * CHECK_COUNT);

    for (int k = 0;k < CHECK_COUNT; ++k )
    {
        int beginIdx = (LOCATIONS + 1 + classes)* stride *k + curIdx;
        int objIndex = beginIdx + LOCATIONS*stride;
        
        //check objectness
        float objProb = Logist(curInput[objIndex]);
        if(objProb <= IGNORE_THRESH)
            continue;

        int row = curIdx / yoloWidth;
        int cols = curIdx % yoloWidth;
        
        //classes
        int classId = -1;
        float maxProb = IGNORE_THRESH;
        for (int c = 0;c<classes;++c){
            float cProb =  Logist(curInput[beginIdx + (5 + c) * stride]) * objProb;
            if(cProb > maxProb){
                maxProb = cProb;
                classId = c;
            }
        }

        if(classId >= 0) {
            float *curOutput = output + bnIdx*outputElem;
            int resCount = (int)atomicAdd(curOutput,1);
            char* data = (char * )curOutput + sizeof(float) + resCount*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            //Location
            det->bbox[0] = (cols + Logist(curInput[beginIdx]))/ yoloWidth;
            det->bbox[1] = (row + Logist(curInput[beginIdx+stride]))/ yoloHeight;
            det->bbox[2] = exp(curInput[beginIdx+2*stride]) * anchors[2*k];
            det->bbox[3] = exp(curInput[beginIdx+3*stride]) * anchors[2*k + 1];
            det->classId = classId;
            det->prob = maxProb;
        }
    }
}

void YoloLayerPlugin::forwardGpu(const float *const * inputs,float * output,cudaStream_t stream,int batchSize) {
    void* devAnchor;
    size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
    CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

    int outputElem = 1;
    for (unsigned int i = 0;i< mYoloKernel.size();++i)
    {
        const auto& yolo = mYoloKernel[i];
        outputElem += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);
    }

    for(int idx = 0 ;idx < batchSize;++idx)
        CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));

    int numElem = 0;
    for (unsigned int i = 0;i< mYoloKernel.size();++i)
    {
        const auto& yolo = mYoloKernel[i];
        numElem = yolo.width*yolo.height*batchSize;
        CUDA_CHECK(cudaMemcpy(devAnchor,yolo.anchors,AnchorLen,cudaMemcpyHostToDevice));
        CalDetection<<< (yolo.width*yolo.height*batchSize + 512 - 1) / 512, 512>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount ,outputElem);
    }

    CUDA_CHECK(cudaFree(devAnchor));
}

// return UPSAMPLE_PLUGIN_TYPE + UPSAMPLE_PLUGIN_NAMESPACE
const char* YoloLayerPluginCreator::getPluginName() const { 
    return YOLOLAYER_PLUGIN_NAME;
}

const char* YoloLayerPluginCreator::getPluginVersion() const { 
    return YOLOLAYER_PLUGIN_VERSION;
}
const PluginFieldCollection* YoloLayerPluginCreator::getFieldNames() { 
    return nullptr;
}
IPluginV2* YoloLayerPluginCreator::createPlugin(const char *layerName, const PluginFieldCollection* fc) {
    return nullptr;
}

// deserialization plugin implementation
IPluginV2* YoloLayerPluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

    if (strName.find("yolo-det") != std::string::npos)
    {
        return (IPluginV2*)(new YoloLayerPlugin(serialData, serialLength));
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

void YoloLayerPluginCreator::setPluginNamespace(const char* pluginNamespace) {
    // don't implement it 
}

const char* YoloLayerPluginCreator::getPluginNamespace() const {
    return YOLOLAYER_PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
