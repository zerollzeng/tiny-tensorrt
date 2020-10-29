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
#include "PluginFactory.h"
// #include "tensorflow/graph.pb.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

Trt::Trt() {
    TrtPluginParams params;
    mPluginFactory = new PluginFactory(params);
    mBuilder = nvinfer1::createInferBuilder(mLogger);
    mConfig = mBuilder->createBuilderConfig();
}

Trt::Trt(TrtPluginParams params) {
    mPluginFactory = new PluginFactory(params);
    mBuilder = nvinfer1::createInferBuilder(mLogger);
    mConfig = mBuilder->createBuilderConfig();
}

Trt::~Trt() {
    if(mPluginFactory != nullptr) {
        delete mPluginFactory;
        mPluginFactory = nullptr;
    }
    if(mContext != nullptr) {
        mContext->destroy();
        mContext = nullptr;
    }
    if(mEngine !=nullptr) {
        mEngine->destroy();
        mEngine = nullptr;
    }
    if(mConfig !=nullptr) {
        mConfig->destroy();
        mConfig = nullptr;
    }
    for(size_t i=0;i<mBinding.size();i++) {
        safeCudaFree(mBinding[i]);
    }
}

void Trt::CreateEngine(
        const std::string& prototxt,
        const std::string& caffeModel,
        const std::string& engineFile,
        const std::vector<std::string>& outputBlobName,
        int maxBatchSize,
        int mode) {
    mBatchSize = maxBatchSize;
    mRunMode = mode;
    spdlog::info("prototxt: {}",prototxt);
    spdlog::info("caffeModel: {}",caffeModel);
    spdlog::info("engineFile: {}",engineFile);
    spdlog::info("outputBlobName: ");
    for(size_t i=0;i<outputBlobName.size();i++) {
        std::cout << outputBlobName[i] << " ";
    }
    std::cout << std::endl;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithCaffe(prototxt,caffeModel,engineFile,outputBlobName)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
    // Notice: close profiler
    //mContext->setProfiler(mProfiler);
}

void Trt::CreateEngine(
        const std::string& onnxModel,
        const std::string& engineFile,
        const std::vector<std::string>& customOutput,
        int maxBatchSize,
        int mode) {
    mBatchSize = maxBatchSize;
    mRunMode = mode;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithOnnx(onnxModel,engineFile,customOutput)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
}

void Trt::CreateEngine(
        const std::string& uffModel,
        const std::string& engineFile,
        const std::vector<std::string>& inputTensorNames,
        const std::vector<std::vector<int>>& inputDims,
        const std::vector<std::string>& outputTensorNames,
        int maxBatchSize,
        int mode) {
    mBatchSize = maxBatchSize;
    mRunMode = mode;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngineWithUff(uffModel,engineFile,inputTensorNames,inputDims, outputTensorNames)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
}

void Trt::Forward() {
    if(mFlags == 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)) {
        mContext->executeV2(&mBinding[0]);
    } else {
        mContext->execute(mBatchSize, &mBinding[0]);
    }
}

void Trt::ForwardAsync(const cudaStream_t& stream) {
    if(mFlags == 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)) {
        mContext->enqueueV2(&mBinding[0], stream, nullptr);
    } else {
        mContext->enqueue(mBatchSize, &mBinding[0], stream, nullptr);
    }
}

void Trt::SetBindingDimensions(std::vector<int>& inputDims, int bindIndex) {
    const nvinfer1::Dims3& dims{inputDims[0],inputDims[1],inputDims[2]};
    mContext->setBindingDimensions(bindIndex, dims);
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex) {
    assert(input.size()*sizeof(float) <= mBindingSize[bindIndex]);
    CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex, const cudaStream_t& stream) {
    assert(input.size()*sizeof(float) <= mBindingSize[bindIndex]);
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, stream));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex) {
    output.resize(mBindingSize[bindIndex]/sizeof(float));
    CUDA_CHECK(cudaMemcpy(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex, const cudaStream_t& stream) {
    output.resize(mBindingSize[bindIndex]/sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
}

void Trt::SetDevice(int device) {
    spdlog::warn("warning: make sure save engine file match choosed device");
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

void Trt::SetInt8Calibrator(const std::string& calibratorType, const std::vector<std::vector<float>>& calibratorData) {
    mRunMode = 2;
    spdlog::warn("INT8 inference is available only on GPUs with compute capability equal or greater than 6.1");
    nvinfer1::IInt8Calibrator* calibrator = GetInt8Calibrator(calibratorType, mBatchSize, calibratorData, "calibrator", false);
    spdlog::info("set int8 inference mode");
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
    mConfig->setInt8Calibrator(calibrator);
}

void Trt::AddDynamicShapeProfile(int batchSize,
                                const std::string& inputName,
                                const std::vector<int>& minDimVec,
                                const std::vector<int>& optDimVec,
                                const std::vector<int>& maxDimVec) {
    const nvinfer1::Dims4& minDim{batchSize, minDimVec[0],minDimVec[1],minDimVec[2]};
    const nvinfer1::Dims4& optDim{batchSize, optDimVec[0],optDimVec[1],optDimVec[2]};
    const nvinfer1::Dims4& maxDim{batchSize, maxDimVec[0],maxDimVec[1],maxDimVec[2]};
    IOptimizationProfile* profile = mBuilder->createOptimizationProfile();
    profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, minDim);
    profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, optDim);
    profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, maxDim);
    assert(profile->isValid());
    mConfig->addOptimizationProfile(profile);
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
        data->destroy();
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
        initLibNvInferPlugins(&mLogger, "");
        mRuntime = nvinfer1::createInferRuntime(mLogger);
        mEngine = mRuntime->deserializeCudaEngine((void*)engineBuf.get(), bufCount, nullptr);
        assert(mEngine != nullptr);
        mBatchSize = mEngine->getMaxBatchSize();
        spdlog::info("max batch size of deserialized engine: {}",mEngine->getMaxBatchSize());
        mRuntime->destroy();
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
    
    spdlog::info("fp16 support: {}",mBuilder->platformHasFastFp16 ());
    spdlog::info("int8 support: {}",mBuilder->platformHasFastInt8 ());
    spdlog::info("Max batchsize: {}",mBuilder->getMaxBatchSize());
    spdlog::info("Max workspace size: {}",mConfig->getMaxWorkspaceSize());
    spdlog::info("Number of DLA core: {}",mBuilder->getNbDLACores());
    spdlog::info("Max DLA batchsize: {}",mBuilder->getMaxDLABatchSize());
    spdlog::info("Current use DLA core: {}",mConfig->getDLACore()); // TODO: set DLA core
    spdlog::info("build engine...");
    mEngine = mBuilder -> buildEngineWithConfig(*mNetwork, *mConfig);
    assert(mEngine != nullptr);
}

bool Trt::BuildEngineWithCaffe(const std::string& prototxt, 
                        const std::string& caffeModel,
                        const std::string& engineFile,
                        const std::vector<std::string>& outputBlobName) {
    spdlog::info("build caffe engine with {} and {}", prototxt, caffeModel);
    assert(mBuilder != nullptr);
    mNetwork = mBuilder->createNetworkV2(mFlags);
    assert(mNetwork != nullptr);
    nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
    if(mPluginFactory != nullptr) {
        parser->setPluginFactoryV2(mPluginFactory);
    }
    // Notice: change here to costom data type
    nvinfer1::DataType type = mRunMode==1 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(prototxt.c_str(),caffeModel.c_str(),
                                                                            *mNetwork,type);
    
    for(auto& s : outputBlobName) {
        mNetwork->markOutput(*blobNameToTensor->find(s.c_str()));
    }
    spdlog::info("Number of mNetwork layers: {}",mNetwork->getNbLayers());
    spdlog::info("Number of input: ", mNetwork->getNbInputs());
    std::cout << "Input layer: " << std::endl;
    for(int i = 0; i < mNetwork->getNbInputs(); i++) {
        std::cout << mNetwork->getInput(i)->getName() << " : ";
        Dims dims = mNetwork->getInput(i)->getDimensions();
        for(int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j] << "x"; 
        }
        std::cout << "\b "  << std::endl;
    }
    spdlog::info("Number of output: {}",mNetwork->getNbOutputs());
    std::cout << "Output layer: " << std::endl;
    for(int i = 0; i < mNetwork->getNbOutputs(); i++) {
        std::cout << mNetwork->getOutput(i)->getName() << " : ";
        Dims dims = mNetwork->getOutput(i)->getDimensions();
        for(int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j] << "x"; 
        }
        std::cout << "\b " << std::endl;
    }
    spdlog::info("parse mNetwork done");

    BuildEngine();

    spdlog::info("serialize engine to {}", engineFile);
    SaveEngine(engineFile);
    
    mBuilder->destroy();
    mNetwork->destroy();
    parser->destroy();
    return true;
}

bool Trt::BuildEngineWithOnnx(const std::string& onnxModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& customOutput) {
    spdlog::info("build onnx engine from {}...",onnxModel);
    assert(mBuilder != nullptr);
    mFlags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    mNetwork = mBuilder->createNetworkV2(mFlags);
    assert(mNetwork != nullptr);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*mNetwork, mLogger);
    if(!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        spdlog::error("error: could not parse onnx engine");
        return false;
    }
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
    if(customOutput.size() > 0) {
        spdlog::info("unmark original output...");
        for(int i=0;i<mNetwork->getNbOutputs();i++) {
            nvinfer1::ITensor* origin_output = mNetwork->getOutput(i);
            mNetwork->unmarkOutput(*origin_output);
        }
        spdlog::info("mark custom output...");
        for(int i=0;i<mNetwork->getNbLayers();i++) {
            nvinfer1::ILayer* custom_output = mNetwork->getLayer(i);
            nvinfer1::ITensor* output_tensor = custom_output->getOutput(0);
            for(size_t j=0; j<customOutput.size();j++) {
                std::string layer_name(output_tensor->getName());
                if(layer_name == customOutput[j]) {
                    mNetwork->markOutput(*output_tensor);
                    break;
                }
            }
        }    
    }
    BuildEngine();

    spdlog::info("serialize engine to {}", engineFile);
    SaveEngine(engineFile);

    mBuilder->destroy();
    mNetwork->destroy();
    parser->destroy();
    return true;
}

bool Trt::BuildEngineWithUff(const std::string& uffModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& inputTensorNames,
                      const std::vector<std::vector<int>>& inputDims,
                      const std::vector<std::string>& outputTensorNames) {
    spdlog::info("build uff engine with {}...", uffModel);
    assert(mBuilder != nullptr);
    mNetwork = mBuilder->createNetworkV2(mFlags);
    assert(mNetwork != nullptr);
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    assert(parser != nullptr);
    assert(inputTensorNames.size() == inputDims.size());
    //parse input
    for(size_t i=0;i<inputTensorNames.size();i++) {
        nvinfer1::Dims dim;
        dim.nbDims = inputDims[i].size();
        for(int j=0;j<dim.nbDims;j++) {
            dim.d[j] = inputDims[i][j];
        }
        parser->registerInput(inputTensorNames[i].c_str(), dim, nvuffparser::UffInputOrder::kNCHW);
    }
    //parse output
    for(size_t i=0;i<outputTensorNames.size();i++) {
        parser->registerOutput(outputTensorNames[i].c_str());
    }
    if(!parser->parse(uffModel.c_str(), *mNetwork, nvinfer1::DataType::kFLOAT)) {
        spdlog::error("error: parse model failed");
    }
    BuildEngine();
    spdlog::info("serialize engine to {}", engineFile);
    SaveEngine(engineFile);
    
    mBuilder->destroy();
    mNetwork->destroy();
    parser->destroy();
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
            mInputSize++;
        }
    }
}
