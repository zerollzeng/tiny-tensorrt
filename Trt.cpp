/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 14:06:38
 * @LastEditTime: 2019-10-23 13:58:06
 * @LastEditors: zerollzeng
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
#include "NvOnnxParser.h"
#include "NvUffParser.h"

Trt::Trt() {
    TrtPluginParams params;
    mPluginFactory = new PluginFactory(params);
}

Trt::Trt(TrtPluginParams params) {
    mPluginFactory = new PluginFactory(params);
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
                       int mode) {
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
        if(!BuildEngine(prototxt,caffeModel,engineFile,outputBlobName,calibratorData,maxBatchSize)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
    // Notice: close profiler
    //mContext->setProfiler(mProfiler);
}

void Trt::CreateEngine(const std::string& onnxModel,
                       const std::string& engineFile,
                       const std::vector<std::string>& customOutput,
                       int maxBatchSize) {
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngine(onnxModel,engineFile,customOutput,maxBatchSize)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
}

void Trt::CreateEngine(const std::string& uffModel,
                       const std::string& engineFile,
                       const std::vector<std::string>& inputTensorName,
                       const std::vector<std::string>& outputTensorName,
                       int maxBatchSize) {
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngine(uffModel,engineFile,inputTensorName,outputTensorName,maxBatchSize)) {
            spdlog::error("error: could not deserialize or build engine");
            return;
        }
    }
    spdlog::info("create execute context and malloc device memory...");
    InitEngine();
}

void Trt::Forward() {
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    mContext->execute(mBatchSize, &mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    spdlog::info("net forward takes {} ms", elapsedTime);
}

void Trt::ForwardAsync(const cudaStream_t& stream) {
    mContext->enqueue(mBatchSize, &mBinding[0], stream, nullptr);
}

void Trt::PrintTime() {
    mProfiler.printLayerTimes(1);
}

void Trt::DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice) {
    if(isHostToDevice) {
        assert(data.size()*sizeof(float) == mBindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], data.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
    } else {
        data.resize(mBindingSize[bindIndex]/sizeof(float));
        CUDA_CHECK(cudaMemcpy(data.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
    }
}

void Trt::DataTransferAsync(std::vector<float>& data, int bindIndex, bool isHostToDevice, cudaStream_t& stream) {
    if(isHostToDevice) {
        assert(data.size()*sizeof(float) == mBindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], data.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, stream));
    } else {
        data.resize(mBindingSize[bindIndex]/sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(data.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
    }
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex, const cudaStream_t& stream) {
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, stream));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex, const cudaStream_t& stream) {
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
}

void Trt::SetDevice(int device) {
    spdlog::warn("warning: make sure save engine file match choosed device");
    CUDA_CHECK(cudaSetDevice(device));
}

int Trt::GetDevice() const { 
    int* device = nullptr; //NOTE: memory leaks here
    CUDA_CHECK(cudaGetDevice(device));
    if(device != nullptr) {
        return device[0];
    } else {
        spdlog::error("Get Device Error");
        return -1;
    }
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
        spdlog::info("build caffe engine with {} and {}", prototxt, caffeModel);
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
        assert(builder != nullptr);
        // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
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
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        Int8EntropyCalibrator* calibrator = nullptr;
        if (mRunMode == 2)
        {
            spdlog::info("set int8 inference mode");
            if (!builder->platformHasFastInt8()) {
                spdlog::warn("Warning: current platform doesn't support int8 inference");
            }
            if (calibratorData.size() > 0 ){
                auto endPos= prototxt.find_last_of(".");
                auto beginPos= prototxt.find_last_of('/') + 1;
                if(prototxt.find("/") == std::string::npos) {
                    beginPos = 0;
                }
                std::string calibratorName = prototxt.substr(beginPos,endPos - beginPos);
                std::cout << "create calibrator,Named:" << calibratorName << std::endl;
                calibrator = new Int8EntropyCalibrator(maxBatchSize,calibratorData,calibratorName,false);
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
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setInt8Calibrator(calibrator);
        }
        
        if (mRunMode == 1)
        {
            spdlog::info("setFp16Mode");
            if (!builder->platformHasFastFp16()) {
                spdlog::warn("the platform do not has fast for fp16");
            }
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        builder->setMaxBatchSize(mBatchSize);
        // set the maximum GPU temporary memory which the engine can use at execution time.
        config->setMaxWorkspaceSize(10 << 20);
        spdlog::info("fp16 support: {}",builder->platformHasFastFp16 ());
        spdlog::info("int8 support: {}",builder->platformHasFastInt8 ());
        spdlog::info("Max batchsize: {}",builder->getMaxBatchSize());
        spdlog::info("Max workspace size: {}",config->getMaxWorkspaceSize());
        spdlog::info("Number of DLA core: {}",builder->getNbDLACores());
        spdlog::info("Max DLA batchsize: {}",builder->getMaxDLABatchSize());
        spdlog::info("Current use DLA core: {}",config->getDLACore()); // TODO: set DLA core
        spdlog::info("build engine...");
        mEngine = builder -> buildEngineWithConfig(*network, *config);
        assert(mEngine != nullptr);
        spdlog::info("serialize engine to {}", engineFile);
        SaveEngine(engineFile);
        
        builder->destroy();
        config->destroy();
        network->destroy();
        parser->destroy();
        if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
        return true;
}

bool Trt::BuildEngine(const std::string& onnxModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& customOutput,
                      int maxBatchSize) {
    spdlog::warn("The ONNX Parser shipped with TensorRT 5.1.x+ supports ONNX IR (Intermediate Representation) version 0.0.3, opset version 9");
    mBatchSize = maxBatchSize;
    spdlog::info("build onnx engine from {}...",onnxModel);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, mLogger);
    if(!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        spdlog::error("error: could not parse onnx engine");
        return false;
    }
    for(int i=0;i<network->getNbLayers();i++) {
        nvinfer1::ILayer* custom_output = network->getLayer(i);
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
        for(int i=0;i<network->getNbOutputs();i++) {
            nvinfer1::ITensor* origin_output = network->getOutput(i);
            network->unmarkOutput(*origin_output);
        }
        spdlog::info("mark custom output...");
        for(int i=0;i<network->getNbLayers();i++) {
            nvinfer1::ILayer* custom_output = network->getLayer(i);
            nvinfer1::ITensor* output_tensor = custom_output->getOutput(0);
            for(size_t j=0; j<customOutput.size();j++) {
                std::string layer_name(output_tensor->getName());
                if(layer_name == customOutput[j]) {
                    network->markOutput(*output_tensor);
                    break;
                }
            }
        }    
    }
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    builder->setMaxBatchSize(mBatchSize);
    config->setMaxWorkspaceSize(10 << 20);
    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);
    spdlog::info("serialize engine to {}", engineFile);
    SaveEngine(engineFile);

    builder->destroy();
    network->destroy();
    parser->destroy();
    return true;
}

bool Trt::BuildEngine(const std::string& uffModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& inputTensorName,
                      const std::vector<std::string>& outputTensorName,
                      int maxBatchSize) {
    mBatchSize = maxBatchSize;
    spdlog::info("build uff engine with {}...", uffModel);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    assert(parser != nullptr);
    nvinfer1::Dims inputDim;
    inputDim.nbDims = 3;
    inputDim.d[0] = 3;
    inputDim.d[1] = 272;
    inputDim.d[2] = 480;
    parser->registerInput(inputTensorName[0].c_str(), inputDim, nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(outputTensorName[0].c_str());
    parser->registerOutput(outputTensorName[1].c_str());
    parser->registerOutput(outputTensorName[2].c_str());
    if(!parser->parse(uffModel.c_str(), *network, nvinfer1::DataType::kFLOAT)) {
        spdlog::error("error: parse model failed");
    }
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(10 << 20);
    builder->setMaxBatchSize(mBatchSize);
    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);
    spdlog::info("serialize engine to {}", engineFile);
    SaveEngine(engineFile);

    builder->destroy();
    network->destroy();
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
