/*
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 14:50:04
 * @LastEditTime: 2019-08-23 14:50:04
 * @LastEditors: Please set LastEditors
 */
#include "YoloV3.h"
#include "utils.h"
#include "spdlog/spdlog.h"

#include <NvInfer.h>
#include <NvCaffeParser.h>

#include <chrono>
#include <vector>
#include <cstring>

YoloV3::YoloV3(const std::string& prototxt, 
                const std::string& caffeModel,
                const std::string& saveEngine,
                const std::vector<std::string>& outputBlobName,
                const std::vector<std::vector<float>>& calibratorData,
                int maxBatchSize,
                RUN_MODE mode,
                int yoloClassNum,
                int netSize) {
    TrtPluginParams* params = new TrtPluginParams();
    params->yoloClassNum = yoloClassNum;
    params->yolo3NetSize = netSize;
    mNet = new Trt(params);
    mNet->CreateEngine(prototxt, caffeModel, saveEngine, outputBlobName, calibratorData, maxBatchSize, mode);
    mYoloClassNum = yoloClassNum;
    mpDetCpu = new float[63883];
}

YoloV3::~YoloV3() {
    if(mNet != nullptr) {
        delete mNet;
        mNet = nullptr;
    }

}


void DoNms(std::vector<Detection>& detections,int classes ,float nmsThresh)
{
    using namespace std;
    auto t_start = chrono::high_resolution_clock::now();

    std::vector<std::vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };
        
        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    std::vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i]; 
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    cout << "Time taken for nms is " << total << " ms." << endl;
}

void YoloV3::DoInference(void* inputContext,void* outputContext) {
    YoloInDataSt* in = (YoloInDataSt*)inputContext;

    // // debug
    // for(int i=0;i<10;i++) {
    //     std::cout << in->data[i] << " ";
    // }
    // std::cout << std::endl;
    // for(int i=259584;i<259594;i++) {
    //     std::cout << in->data[i] << " ";
    // }
    // std::cout << std::endl;

    YoloOutDataSt* out = (YoloOutDataSt*)outputContext;

    mNet->CopyFromHostToDevice(in->data, 0, 0);
    mNet->Forward(0);
    mNet->CopyFromDeviceToHost(mpDetCpu, 1, 0);
    

    // // debug
    // float* testf = new float[10];
    // CUDA_CHECK(cudaMemcpy(testf,mBinding[1],10*sizeof(float),cudaMemcpyDeviceToHost));
    // std::cout << "-----------test-----------" << std::endl;
    // for(int i=0;i<10;i++) {
    //     std::cout << testf[i] << " ";
    // }
    // std::cout << std::endl << "-----------" << std::endl;
    // debug
    // nvinfer1::Dims dims = mEngine->getBindingDimensions(1);
    // size_t outputSize = volume(dims) * mBatchSize;
    // float* output = new float[outputSize];
    // std::cout << "-----------output----------" << std::endl;
    // for(int i=0;i<10;i++) {
    //     std::cout << output[i] << " ";
    // }
    // std::cout << std::endl;
    int detCount = (int)mpDetCpu[0];
    std::cout << "detCount: " << detCount << std::endl;
    for(int i=1;i<71;i++) {
        if((i-1)%6 == 0) {
            std::cout << std::endl;
        }
        std::cout << mpDetCpu[i] << " ";
    }

    std::vector<Detection> result;
    result.resize(detCount);
    memcpy(result.data(), &mpDetCpu[1], detCount*sizeof(Detection));

    //scale bbox to img
    int width = in->originalWidth;
    int height = in->originalHeight;
    float scale = std::min(float(mNetWidth)/width,float(mNetHeight)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : result)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * mNetWidth - (mNetWidth - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * mNetHeight - (mNetHeight - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }
    DoNms(result,mYoloClassNum,0.5);
    std::cout << "number of people: " << result.size() << std::endl;
    // for(const auto& item : result) {
    //     Bbox bbox;
    //     auto& b= item.bbox;
    //     bbox.left = std::max(int((b[0]-b[2]/2.)*width),0);
    //     bbox.right = std::max(int((b[0]+b[2]/2.)*width),width);
    //     bbox.top = std::min(int((b[1]-b[3]/2.)*height),0);
    //     bbox.bottom = std::min(int((b[1]+b[3]/2.)*height),height);
    //     bbox.score = item.prob;
    //     std::cout << "left: " << bbox.left << ", top: " << bbox.top << ", right: " << bbox.right << ", bottom: " << bbox.bottom << ", score: " << bbox.score << std::endl;
    //     out->result.push_back(bbox);
    // }
    for(const auto& item : result) {
        Bbox bbox;
        auto& b= item.bbox;
        bbox.left = std::max(int((b[0]-b[2]/2.)*width),0);
        bbox.right = std::min(int((b[0]+b[2]/2.)*width),width);
        bbox.top = std::max(int((b[1]-b[3]/2.)*height),0);
        bbox.bottom = std::min(int((b[1]+b[3]/2.)*height),height);
        bbox.score = item.prob;
        spdlog::info("object in {},{},{},{}",bbox.left,bbox.top,bbox.right,bbox.bottom);
        out->result.push_back(bbox);
    }
}

// void YoloV3::MallocExtraMemory() {
//     spdlog::info("malloc det memory...")
//     nvinfer1::Dims detDims = mEngine->getBindingDimensions(1);
//     mDetDims = nvinfer1::Dims3(detDims.d[0],detDims.d[1],detDims.d[2]);
//     std::cout << mEngine->getBindingName(1) << " size: " << mBatchSize << " "  << mDetDims.d[0] << " " << mDetDims.d[1] << " " << mDetDims.d[2] << std::endl;
//     mDetSize = mBatchSize * mDetDims.d[0] * mDetDims.d[1] * mDetDims.d[2] * getElementSize(mInputDataType);
//     mpDetGpu = safeCudaMalloc(mDetSize);
//     mpDetCpu = new float[mDetSize / getElementSize(mInputDataType)];
// }