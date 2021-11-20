/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 14:06:38
 * @LastEditTime: 2020-06-10 11:51:09
 * @LastEditors: zerollzeng
 */
#include "Trt.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "Int8Calibrator.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

using NDCFLAG = nvinfer1::NetworkDefinitionCreationFlag;

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= mSeverity) {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            spdlog::critical("[F] [TRT] {}", msg);
            break;
        case Severity::kERROR:
            spdlog::error("[E] [TRT] {}", msg);
            break;
        case Severity::kWARNING:
            spdlog::warn("[W] [TRT] {}", msg);
            break;
        case Severity::kINFO:
            spdlog::info("[I] [TRT] {}", msg);
            break;
        case Severity::kVERBOSE:
            spdlog::info("[V] [TRT] {}", msg);
            break;
        default:
            assert(false && "invalid log level");
            break;
        }
    }
}

void TrtLogger::setLogSeverity(Severity severity) {
    mSeverity = severity;
}


Trt::Trt() {
    spdlog::info("create Trt instance");
    mLogger = new TrtLogger();
    mBuilder = nvinfer1::createInferBuilder(*mLogger);
    mConfig = mBuilder->createBuilderConfig();
    mProfile = mBuilder->createOptimizationProfile();
}

Trt::~Trt() {
    spdlog::info("destroy Trt instance");
    if(mContext != nullptr) {
        delete mContext;
        mContext = nullptr;
    }
    if(mEngine !=nullptr) {
        delete mEngine;
        mEngine = nullptr;
    }
    if(mConfig !=nullptr) {
        delete mConfig;
        mConfig = nullptr;
    }
    if(mBuilder !=nullptr) {
        delete mBuilder;
        mBuilder = nullptr;
    }
    if(mNetwork !=nullptr) {
        delete mNetwork;
        mNetwork = nullptr;
    }
    for(size_t i=0;i<mBinding.size();i++) {
        safeCudaFree(mBinding[i]);
    }
}

void Trt::CreateEngine(
        const std::string& onnxModel,
        const std::string& engineFile,
        int maxBatchSize,
        int mode) {
    mBatchSize = maxBatchSize;
    mRunMode = mode;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithOnnx(onnxModel,engineFile,mCustomOutputs)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    InitEngine();
}

void Trt::Forward() {
    if(mFlags == 1U << static_cast<uint32_t>(NDCFLAG::kEXPLICIT_BATCH)) {
        mContext->executeV2(&mBinding[0]);
    } else {
        mContext->execute(mBatchSize, &mBinding[0]);
    }
}

void Trt::ForwardAsync(const cudaStream_t& stream) {
    if(mFlags == 1U << static_cast<uint32_t>(NDCFLAG::kEXPLICIT_BATCH)) {
        mContext->enqueueV2(&mBinding[0], stream, nullptr);
    } else {
        mContext->enqueue(mBatchSize, &mBinding[0], stream, nullptr);
    }
}

void Trt::SetBindingDimensions(std::vector<int>& inputDims, int bindIndex) {
    nvinfer1::Dims dims;
    int nbDims = inputDims.size();
    dims.nbDims = nbDims;
    for(int i=0; i< nbDims; i++) {
        dims.d[i] = inputDims[i];
    }
    mContext->setBindingDimensions(bindIndex, dims);
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex) {
    assert(input.size()*sizeof(float) <= mBindingSize[bindIndex]);
    CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], input.data(),
        input.size()*sizeof(float), cudaMemcpyHostToDevice));
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input,
                               int bindIndex, const cudaStream_t& stream) {
    assert(input.size()*sizeof(float) <= mBindingSize[bindIndex]);
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(),
        input.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex) {
    output.resize(mBindingSize[bindIndex]/sizeof(float));
    CUDA_CHECK(cudaMemcpy(output.data(), mBinding[bindIndex],
        mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,
                               const cudaStream_t& stream) {
    output.resize(mBindingSize[bindIndex]/sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex],
        mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
}

void Trt::SetDevice(int device) {
    spdlog::info("set device {}", device);
    CUDA_CHECK(cudaSetDevice(device));
}

int Trt::GetDevice() const {
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if(device != -1) {
        return device;
    } else {
        spdlog::error("Get Device Error");
        return -1;
    }
}

void Trt::SetInt8Calibrator(const std::string& calibratorType, const int batchSize,
                            const std::string& dataPath, const std::string& calibrateCachePath) {
    spdlog::info("set int8 inference mode");
    mRunMode = 2;
    nvinfer1::IInt8Calibrator* calibrator = GetInt8Calibrator(
        calibratorType, batchSize, dataPath, calibrateCachePath);
    if (!mBuilder->platformHasFastInt8()) {
        spdlog::warn("Warning: current platform doesn't support int8 inference");
    }
    // enum class BuilderFlag : int
    // {
    //     kFP16 = 0,         //!< Enable FP16 layer selection.
    //     kINT8 = 1,         //!< Enable Int8 layer selection.
    //     kDEBUG = 2,        //!< Enable debugging of layers via synchronizing after every layer.
    //     kGPU_FALLBACK = 3, //!< Enable layers marked to execute on GPU if layer cannot execute on DLA.
    //     kSTRICT_TYPES = 4, //!< Enables strict type constraints.
    //     kREFIT = 5,        //!< Enable building a refittable engine.
    // };
    mConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
    mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    mConfig->setInt8Calibrator(calibrator);
}

void Trt::SetDLACore(int dlaCore) {
    spdlog::info("set dla core {}", dlaCore);
    if(dlaCore >= 0) {
        mConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        mConfig->setDLACore(dlaCore);
        mConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }
}

void Trt::SetCustomOutput(const std::vector<std::string>& customOutputs) {
    spdlog::info("set custom output");
    mCustomOutputs = customOutputs;
}

void Trt::SetLogLevel(int severity) {
    spdlog::info("set log level {}", severity);
    mLogger->setLogSeverity(static_cast<nvinfer1::ILogger::Severity>(severity));
}

void Trt::AddDynamicShapeProfile(const std::string& inputName,
                                const std::vector<int>& minDimVec,
                                const std::vector<int>& optDimVec,
                                const std::vector<int>& maxDimVec) {
    spdlog::info("add mProfile for {}", inputName);
    nvinfer1::Dims minDim, optDim, maxDim;
    int nbDims = optDimVec.size();
    minDim.nbDims = nbDims;
    optDim.nbDims = nbDims;
    maxDim.nbDims = nbDims;
    for(int i=0; i< nbDims; i++) {
        minDim.d[i] = minDimVec[i];
        optDim.d[i] = optDimVec[i];
        maxDim.d[i] = maxDimVec[i];
    }
    mProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, minDim);
    mProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, optDim);
    mProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, maxDim);
    assert(mProfile->isValid());
    mConfig->addOptimizationProfile(mProfile);
}

int Trt::GetMaxBatchSize() const{
    return mBatchSize;
}

void* Trt::GetBindingPtr(int bindIndex) const {
    return mBinding[bindIndex];
}

size_t Trt::GetBindingSize(int bindIndex) const {
    return mBindingSize[bindIndex];
}

nvinfer1::Dims Trt::GetBindingDims(int bindIndex) const {
    return mBindingDims[bindIndex];
}

nvinfer1::DataType Trt::GetBindingDataType(int bindIndex) const {
    return mBindingDataType[bindIndex];
}

std::string Trt::GetBindingName(int bindIndex) const{
    return mBindingName[bindIndex];
}

int Trt::GetNbInputBindings() const {
    return mNbInputBindings;
}

int Trt::GetNbOutputBindings() const {
    return mNbOutputBindings;
}

void Trt::SaveEngine(const std::string& fileName) {
    if(fileName == "") {
        spdlog::warn("empty engine file name, skip save");
        return;
    }
    if(mEngine != nullptr) {
        spdlog::info("save engine to {}...",fileName);
        nvinfer1::IHostMemory* data = mEngine->serialize();
        std::ofstream file;
        file.open(fileName,std::ios::binary | std::ios::out);
        if(!file.is_open()) {
            spdlog::error("read create engine file {} failed",fileName);
            return;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        delete data;
    } else {
        spdlog::error("engine is empty, save engine failed");
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
        initLibNvInferPlugins(mLogger, "");
        mRuntime = nvinfer1::createInferRuntime(*mLogger);
        mEngine = mRuntime->deserializeCudaEngine((void*)engineBuf.get(), bufCount);
        assert(mEngine != nullptr);
        mBatchSize = mEngine->getMaxBatchSize();
        spdlog::info("max batch size of deserialized engine: {}",mEngine->getMaxBatchSize());
        delete mRuntime;
        return true;
    }
    return false;
}

void Trt::BuildEngine() {
    if (mRunMode == 1)
    {
        spdlog::info("setFp16Mode");
        if (!mBuilder->platformHasFastFp16()) {
            spdlog::warn("the platform do not has fast for fp16");
        }
        mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    mBuilder->setMaxBatchSize(mBatchSize);
    // set the maximum GPU temporary memory which the engine can use at execution time.
    mConfig->setMaxWorkspaceSize(10 << 20);

    spdlog::info("Max batchsize: {}",mBuilder->getMaxBatchSize());
    spdlog::info("Max workspace size: {}",mConfig->getMaxWorkspaceSize());
    spdlog::info("build engine...");
    mEngine = mBuilder -> buildEngineWithConfig(*mNetwork, *mConfig);
    assert(mEngine != nullptr);
}

bool Trt::BuildEngineWithOnnx(const std::string& onnxModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& customOutput) {
    spdlog::info("build onnx engine from {}...",onnxModel);
    assert(mBuilder != nullptr);
    mFlags = 1U << static_cast<uint32_t>(NDCFLAG::kEXPLICIT_BATCH);
    mNetwork = mBuilder->createNetworkV2(mFlags);
    assert(mNetwork != nullptr);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*mNetwork, *mLogger);
    if(!parser->parseFromFile(onnxModel.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        spdlog::error("error: could not parse onnx engine");
        return false;
    }
#ifdef DEBUG
    for(int i=0;i<mNetwork->getNbLayers();i++) {
        nvinfer1::ILayer* custom_output = mNetwork->getLayer(i);
        for(int j=0;j<custom_output->getNbInputs();j++) {
            nvinfer1::ITensor* input_tensor = custom_output->getInput(j);
            std::cout << input_tensor->getName() << " ";
        }
        std::cout << " -------> ";
        for(int j=0;j<custom_output->getNbOutputs();j++) {
            nvinfer1::ITensor* output_tensor = custom_output->getOutput(j);
            std::cout << output_tensor->getName() << " ";
        }
        std::cout << std::endl;
    }
#endif
    if(customOutput.size() > 0) {
        spdlog::info("unmark original output...");
        for(int i=0;i<mNetwork->getNbOutputs();i++) {
            nvinfer1::ITensor* origin_output = mNetwork->getOutput(i);
            mNetwork->unmarkOutput(*origin_output);
        }
        spdlog::info("mark custom output...");
        for(int i=0;i<mNetwork->getNbLayers();i++) {
            nvinfer1::ILayer* custom_output = mNetwork->getLayer(i);
            for(int j=0;j<custom_output->getNbOutputs();j++) {
                nvinfer1::ITensor* output_tensor = custom_output->getOutput(j);
                for(size_t k=0; k<customOutput.size();k++) {
                    std::string layer_name(output_tensor->getName());
                    if(layer_name == customOutput[k]) {
                        mNetwork->markOutput(*output_tensor);
                        break;
                    }
                }
            }

        }
    }
    BuildEngine();

    SaveEngine(engineFile);

    delete parser;

    return true;
}

void Trt::InitEngine() {
    spdlog::info("init engine...");
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    spdlog::info("malloc device memory");
    int nbBindings = mEngine->getNbBindings();
    if(mConfig->getNbOptimizationProfiles() > 0) {
        spdlog::info("malloc memory with max dims when use dynamic shape");
        nbBindings = nbBindings / 2;
    }
    std::cout << "nbBingdings: " << nbBindings << std::endl;
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for(int i=0; i< nbBindings; i++) {
        const char* name = mEngine->getBindingName(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        nvinfer1::Dims dims;
        if(mConfig->getNbOptimizationProfiles() > 0) {
            if(mEngine->bindingIsInput(i)) {
                dims = mProfile->getDimensions(name, nvinfer1::OptProfileSelector::kMAX);
                mContext->setBindingDimensions(i, dims);
            } else {
                assert(mContext->allInputDimensionsSpecified());
                dims = mContext->getBindingDimensions(i);
            }
        } else {
            dims = mEngine->getBindingDimensions(i);
        }
        int64_t totalSize = volume(dims) * mBatchSize * getElementSize(dtype);
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;
        if(mEngine->bindingIsInput(i)) {
            spdlog::info("input: ");
        } else {
            spdlog::info("output: ");
        }
        spdlog::info("binding bindIndex: {}, name: {}, size in byte: {}",i,name,totalSize);
        spdlog::info("binding dims with {} dimemsion",dims.nbDims);
        for(int j=0;j<dims.nbDims;j++) {
            std::cout << dims.d[j] << " x ";
        }
        std::cout << "\b\b  "<< std::endl;
        mBinding[i] = safeCudaMalloc(totalSize);
        if(mEngine->bindingIsInput(i)) {
            mNbInputBindings++;
        } else {
            mNbOutputBindings++;
        }
    }
}
