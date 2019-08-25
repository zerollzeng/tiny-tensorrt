/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 14:06:38
 * @LastEditTime: 2019-08-23 14:33:38
 * @LastEditors: Please set LastEditors
 */
#include "Trt.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "Int8EntropyCalibrator.h"
#include "plugin/PluginFactory.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include "NvInfer.h"
#include "NvCaffeParser.h"



Trt::Trt(TrtPluginParams* params /*= nullptr */) {
    if(params == nullptr) {
        TrtPluginParams p;
        mPluginFactory = new PluginFactory(p);
    } else {
        mPluginFactory = new PluginFactory(*params);
    }
    
}

Trt::~Trt() {
    if(mPluginFactory != nullptr) {
        delete mPluginFactory;
        mPluginFactory = nullptr;
    }
}

void Trt::CreateEngine(const std::string& prototxt, 
                       const std::string& caffeModel,
                       const std::string& engineFile,
                       const std::vector<std::string>& outputBlobName,
                       const std::vector<std::vector<float>>& calibratorData,
                       int maxBatchSize,
                       RUN_MODE mode) {
    mRunMode = mode;
    if(!DeserializeEngine(engineFile)) {
        BuildEngine(prototxt,caffeModel,engineFile,outputBlobName,calibratorData,maxBatchSize);
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
    // Notice: close profiler
    //mContext->setProfiler(mProfiler);
}

void Trt::Forward(const cudaStream_t& stream) {
    if(stream != 0) {
        mContext->enqueue(mBatchSize, &mBinding[0], stream, nullptr);
    } else {
        mContext->execute(mBatchSize,&mBinding[0]);
    }
}

void Trt::PrintTime() {
    
}

void Trt::CopyFromHostToDevice(const void* pData, int index, const cudaStream_t& stream) {
    if(stream != 0) {
        CUDA_CHECK(cudaMemcpyAsync(mBinding[index], pData, mBindingSize[index], cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(mBinding[index], pData, mBindingSize[index], cudaMemcpyHostToDevice));
    }
}

void Trt::CopyFromDeviceToHost(void* pData, int index, const cudaStream_t& stream) {
    if(stream != 0) {
        CUDA_CHECK(cudaMemcpyAsync(pData, mBinding[index], mBindingSize[index], cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(pData, mBinding[index], mBindingSize[index], cudaMemcpyDeviceToHost));
    }
}

int Trt::GetMaxBatchSize() {
    return mBatchSize;
}

void* Trt::GetBindingPtr(int index) const {
    return mBinding[index];
}

size_t Trt::GetBindingSize(int index) const {
    return mBindingSize[index];
}

nvinfer1::Dims Trt::GetBindingDims(int index) const {
    return mBindingDims[index];
}

nvinfer1::DataType Trt::GetBindingDataType(int index) const {
    return mBindingDataType[index];
}

void Trt::SaveEngine(std::string fileName) {
    if(mEngine) {
        nvinfer1::IHostMemory* data = mEngine->serialize();
        std::ofstream file;
        file.open(fileName,std::ios::binary | std::ios::out);
        if(!file.is_open()) {
            std::cout << "read create engine file" << fileName <<" failed" << std::endl;
            return;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        data->destroy();
        std::cout << "save engine to: " << fileName << " done" << std::endl;
    } else {
        std::cout << "save engine failed" << std::endl;
    }
}

bool Trt::DeserializeEngine(const std::string& engineFile) {
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    if(in.is_open()) {
        spdlog::info("deserialize engine from {}",engineFile);
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);
        initLibNvInferPlugins(&mLogger, "");
        mRuntime = nvinfer1::createInferRuntime(mLogger);
        mEngine = mRuntime->deserializeCudaEngine((void*)engineBuf.get(), bufCount, nullptr);
        mBatchSize = mEngine->getMaxBatchSize();
        spdlog::info("max batch size of deserialized engine: {}",mEngine->getMaxBatchSize());
        return true;
    }
    return false;
}

bool Trt::BuildEngine(const std::string& prototxt, 
                        const std::string& caffeModel,
                        const std::string& engineFile,
                        const std::vector<std::string>& outputBlobName,
                        const std::vector<std::vector<float>>& calibratorData,
                        int maxBatchSize) {
        mBatchSize = maxBatchSize;
        spdlog::info("create engine by build...");
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
        assert(builder != nullptr);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();
        assert(network != nullptr);
        nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
        if(mPluginFactory != nullptr) {
            parser->setPluginFactoryV2(mPluginFactory);
        }
        // Notice: change here to costom data type
        const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(prototxt.c_str(),caffeModel.c_str(),*network,nvinfer1::DataType::kFLOAT);
        for(auto& s : outputBlobName) {
            network->markOutput(*blobNameToTensor->find(s.c_str()));
        }
        spdlog::info("Number of network layers: {}",network->getNbLayers());
        spdlog::info("Number of input: ", network->getNbInputs());
        std::cout << "Input layer: " << std::endl;
        for(int i = 0; i < network->getNbInputs(); i++) {
            std::cout << network->getInput(i)->getName() << " : ";
            Dims dims = network->getInput(i)->getDimensions();
            for(int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j] << "x"; 
            }
            std::cout << "\b "  << std::endl;
        }
        spdlog::info("Number of output: {}",network->getNbOutputs());
        std::cout << "Output layer: " << std::endl;
        for(int i = 0; i < network->getNbOutputs(); i++) {
            std::cout << network->getOutput(i)->getName() << " : ";
            Dims dims = network->getOutput(i)->getDimensions();
            for(int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j] << "x"; 
            }
            std::cout << "\b " << std::endl;
        }
        spdlog::info("parse network done");
        Int8EntropyCalibrator* calibrator = nullptr; // NOTE: memory leak here
        if (mRunMode == RUN_MODE::INT8)
        {
            spdlog::info("set int8 inference mode");
            if (!builder->platformHasFastInt8())
                spdlog::warn("Warning: current platform doesn't support int8 inference");
            builder->setInt8Mode(true);
            
            if (calibratorData.size() > 0 ){
                auto endPos= prototxt.find_last_of(".");
                auto beginPos= prototxt.find_last_of('/') + 1; // NOTE: optimize here
                std::string calibratorName = prototxt.substr(beginPos,endPos - beginPos);
                std::cout << "create calibrator,Named:" << calibratorName << std::endl;
                calibrator = new Int8EntropyCalibrator(maxBatchSize,calibratorData,calibratorName,false);
            }
            builder->setInt8Calibrator(calibrator);
        }
        
        if (mRunMode == RUN_MODE::FLOAT16)
        {
            spdlog::info("setFp16Mode");
            if (!builder->platformHasFastFp16()) {
                spdlog::warn("the platform do not has fast for fp16");
            }
            builder->setFp16Mode(true);
        }
        builder->setMaxBatchSize(mBatchSize);
        builder->setMaxWorkspaceSize(10 << 20); // Warning: here might have bug
        spdlog::info("fp16 support: {}",builder->platformHasFastFp16 ());
        spdlog::info("int8 support: {}",builder->platformHasFastInt8 ());
        spdlog::info("Max batchsize: {}",builder->getMaxBatchSize());
        spdlog::info("Max workspace size: {}",builder->getMaxWorkspaceSize());
        spdlog::info("Number of DLA core: {}",builder->getNbDLACores());
        spdlog::info("Max DLA batchsize: {}",builder->getMaxDLABatchSize());
        spdlog::info("Current use DLA core: {}",builder->getDLACore());
        spdlog::info("Half2 mode: {}",builder->getHalf2Mode());
        spdlog::info("INT8 mode: {}",builder->getInt8Mode());
        spdlog::info("FP16 mode: {}",builder->getFp16Mode());
        spdlog::info("build engine...");
        mEngine = builder -> buildCudaEngine(*network);
        assert(mEngine != nullptr);
        spdlog::info("serialize engine to file...");
        SaveEngine(engineFile);
        
        builder->destroy();
        network->destroy();
        parser->destroy();
        if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
        return true;
}

void Trt::InitEngine() {
    spdlog::info("init engine...");
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    spdlog::info("malloc device memory");
    int nbBindings = mEngine->getNbBindings();
    std::cout << "nbBingdings: " << nbBindings << std::endl;
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for(int i=0; i< nbBindings; i++) {
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);
        int64_t totalSize = volume(dims) * mBatchSize * getElementSize(dtype);
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;
        spdlog::info("binding index: {}, name: {}, size in byte: {}",i,name,totalSize);
        mBinding[i] = safeCudaMalloc(totalSize);
        if(mEngine->bindingIsInput(i)) {
            mInputSize++;
        }
    }
}

