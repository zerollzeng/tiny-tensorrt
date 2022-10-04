#include "Trt.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "Int8Calibrator.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "NvInferVersion.h"

void SetDevice(int device) {
    spdlog::info("set device {}", device);
    CUDA_CHECK(cudaSetDevice(device));
}

int GetDevice() {
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if(device != -1) {
        return device;
    } else {
        spdlog::error("Get Device Error");
        return -1;
    }
}

void SaveEngine(const std::string& fileName, TrtUniquePtr<nvinfer1::IHostMemory>& plan) {
    if(fileName == "") {
        spdlog::warn("empty engine file name, skip save");
        return;
    }
    assert(plan != nullptr);
    spdlog::info("save engine to {}...",fileName);
    std::ofstream file;
    file.open(fileName,std::ios::binary | std::ios::out);
    if(!file.is_open()) {
        spdlog::error("read create engine file {} failed",fileName);
        return;
    }
    file.write((const char*)plan->data(), plan->size());
    file.close();
}

bool setTensorDynamicRange(const nvinfer1::INetworkDefinition& network, float inRange, float outRange)
{
    // Ensure that all layer inputs have a dynamic range.
    for (int l = 0; l < network.getNbLayers(); l++)
    {
        auto* layer = network.getLayer(l);
        for (int i = 0; i < layer->getNbInputs(); i++)
        {
            nvinfer1::ITensor* input{layer->getInput(i)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input && !input->dynamicRangeIsSet())
            {
                if (!input->setDynamicRange(-inRange, inRange))
                {
                    return false;
                }
            }
        }
        for (int o = 0; o < layer->getNbOutputs(); o++)
        {
            nvinfer1::ITensor* output{layer->getOutput(o)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output dynamic range.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    if (!output->setDynamicRange(-inRange, inRange))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!output->setDynamicRange(-outRange, outRange))
                    {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

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
#if NV_TENSORRT_MAJOR < 7
    assert(false && "tiny-tensorrt only support TRT version greater than 7");
#endif
    spdlog::info("create Trt instance");
    mLogger.reset(new TrtLogger());
    initLibNvInferPlugins(mLogger.get(), "");
    mBuilder.reset(nvinfer1::createInferBuilder(*mLogger));
    assert(mBuilder != nullptr && "create trt builder failed");
    mConfig.reset(mBuilder->createBuilderConfig());
    assert(mConfig != nullptr && "create trt builder config failed");
#if !(NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >=4)
    mConfig->setMaxWorkspaceSize(1 << 30); // 1GB
#endif
    mProfile = mBuilder->createOptimizationProfile();
    assert(mProfile != nullptr && "create trt builder optimazation profile failed");
}

Trt::~Trt() {
    spdlog::info("destroy Trt instance");
    mProfile = nullptr;
    for(size_t i=0;i<mBinding.size();i++) {
        safeCudaFree(mBinding[i]);
    }
}

void Trt::EnableFP16() {
    assert(mBuilder != nullptr && mConfig !=nullptr && "Please set config before build engine");
    spdlog::info("enable FP16");
    if (!mBuilder->platformHasFastFp16()) {
        spdlog::warn("the platform doesn't have native fp16 support");
    }
    mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
}

void Trt::EnableINT8() {
    assert(mBuilder != nullptr && mConfig !=nullptr && "Please set config before build engine");
    spdlog::info("enable int8, call SetInt8Calibrator to set int8 calibrator");
    if (!mBuilder->platformHasFastInt8()) {
        spdlog::warn("the platform doesn't have native int8 support");
    }
    mConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
}

void Trt::SetInt8Calibrator(const std::string& calibratorType, const int batchSize,
                            const std::string& dataPath, const std::string& calibrateCachePath) {
    assert(mBuilder != nullptr && mConfig !=nullptr && "Please set config before build engine");
    spdlog::info("set int8 inference mode");
    nvinfer1::IInt8Calibrator* calibrator = GetInt8Calibrator(
        calibratorType, batchSize, dataPath, calibrateCachePath);
    mConfig->setInt8Calibrator(calibrator);
}

// depricated at TRT 8.4
#if !(NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >=4)
void Trt::SetWorkpaceSize(size_t workspaceSize) {
    assert(mBuilder != nullptr && mConfig !=nullptr && "Please set config before build engine");
    mConfig->setMaxWorkspaceSize(workspaceSize);
    spdlog::info("set max workspace size: {}", mConfig->getMaxWorkspaceSize());
}
#endif // NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >=4

void Trt::SetDLACore(int dlaCore) {
    assert(mBuilder != nullptr && mConfig !=nullptr && "Please set config before build engine");
    spdlog::info("set dla core {}", dlaCore);
    if(dlaCore >= 0) {
        mConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        mConfig->setDLACore(dlaCore);
        mConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }
}

void Trt::SetCustomOutput(const std::vector<std::string>& customOutputs) {
    assert(mBuilder != nullptr && mConfig !=nullptr && "Please set config before build engine");
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
    mIsDynamicShape = true;
}

void Trt::BuildEngine(
        const std::string& onnxModel,
        const std::string& engineFile) {
    spdlog::info("build onnx engine from {}...",onnxModel);
    TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*mLogger)};
    assert(runtime != nullptr && "create trt runtime failed");
    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network{mBuilder->createNetworkV2(flag)};
    assert(network != nullptr && "create trt network failed");
    TrtUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, *mLogger)};
    assert(network != nullptr && "create trt onnx parser failed");
    bool parse_success = parser->parseFromFile(onnxModel.c_str(),
        static_cast<int>(Severity::kWARNING));
    assert(parse_success && "parse onnx file failed");
    if(mCustomOutputs.size() > 0) {
        spdlog::info("unmark original output...");
        for(int i=0;i<network->getNbOutputs();i++) {
            nvinfer1::ITensor* origin_output = network->getOutput(i);
            network->unmarkOutput(*origin_output);
        }
        spdlog::info("mark custom output...");
        for(int i=0;i<network->getNbLayers();i++) {
            nvinfer1::ILayer* custom_output = network->getLayer(i);
            for(int j=0;j<custom_output->getNbOutputs();j++) {
                nvinfer1::ITensor* output_tensor = custom_output->getOutput(j);
                for(size_t k=0; k<mCustomOutputs.size();k++) {
                    std::string layer_name(output_tensor->getName());
                    if(layer_name == mCustomOutputs[k]) {
                        network->markOutput(*output_tensor);
                        break;
                    }
                }
            }

        }
    }
    if(mConfig->getFlag(nvinfer1::BuilderFlag::kINT8) && mConfig->getInt8Calibrator() == nullptr) {
        spdlog::warn("No calibrator found, using fake scale");
        setTensorDynamicRange(*network, 2.0f, 4.0f);
    }
    if(mIsDynamicShape) {
        assert(mProfile->isValid() && "Invalid dynamic shape profile");
        mConfig->addOptimizationProfile(mProfile);
    }
#if NV_TENSORRT_MAJOR < 8
    mEngine.reset(mBuilder -> buildEngineWithConfig(*network, *mConfig));
    TrtUniquePtr<nvinfer1::IHostMemory> plan{mEngine ->serialize()};
#else
    TrtUniquePtr<nvinfer1::IHostMemory> plan{mBuilder -> buildSerializedNetwork(*network, *mConfig)};
    mEngine.reset(runtime -> deserializeCudaEngine(plan->data(), plan->size()));
#endif
    assert(mEngine != nullptr && "build trt engine failed");
    SaveEngine(engineFile, plan);
    mContext.reset(mEngine->createExecutionContext());
    assert(mContext != nullptr);
    CreateDeviceBuffer();
    mBuilder.reset(nullptr);
    mConfig.reset(nullptr);
}

bool Trt::DeserializeEngine(const std::string& engineFile, int dlaCore) {
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    if(in.is_open()) {
        spdlog::info("deserialize engine from {}",engineFile);
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);
        TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*mLogger)};
        if(dlaCore >= 0) {
            runtime->setDLACore(dlaCore);
        }
        mEngine.reset(runtime->deserializeCudaEngine((void*)engineBuf.get(), bufCount));
        assert(mEngine != nullptr);
        mContext.reset(mEngine->createExecutionContext());
        assert(mContext != nullptr);
        if(mIsDynamicShape) {
            assert(mProfile->isValid() && "Invalid dynamic shape profile");
            mConfig->addOptimizationProfile(mProfile);
        }
        CreateDeviceBuffer();
        return true;
    }
    return false;
}

bool Trt::Forward() {
    return mContext->executeV2(&mBinding[0]);
}

bool Trt::Forward(const cudaStream_t& stream) {
    return mContext->enqueueV2(&mBinding[0], stream, nullptr);
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

void Trt::CopyFromHostToDevice(const std::vector<float>& input,
                               int bindIndex, const cudaStream_t& stream) {
#ifdef DEBUG
    spdlog::info("input size: {}, binding size: {}", input.size()*sizeof(float), mBindingSize[bindIndex]);
#endif
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(),
        input.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,
                               const cudaStream_t& stream) {
#ifdef DEBUG
    spdlog::info("output size: {}, binding size: {}", output.size()*sizeof(float), mBindingSize[bindIndex]);
#endif
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex],
        output.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
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

void Trt::CreateDeviceBuffer() {
    spdlog::info("malloc device memory");
    int nbBindings = mEngine->getNbBindings();
    spdlog::info("nbBingdings: {}", nbBindings);
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for(int i=0; i< nbBindings; i++) {
        const char* name = mEngine->getBindingName(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        nvinfer1::Dims dims;
        if(mIsDynamicShape) {
            // specify max input dimensions to get max output dimensions
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
        int64_t totalSize = volume(dims) * getElementSize(dtype);
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
