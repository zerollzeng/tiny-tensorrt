#ifndef YOLOV3_HPP
#define YOLOV3_HPP

#include "Trt.h"

struct YoloInDataSt{
    float data[1*3*416*416];
    int originalWidth;
    int originalHeight;
};

struct Bbox {
    int left,right,top,bottom;
    float score;
};

struct YoloOutDataSt{
    std::vector<Bbox> result;
};

class YoloV3 {
public:    
    YoloV3(const std::string& prototxt, 
            const std::string& caffeModel,
            const std::string& saveEngine,
            const std::vector<std::string>& outputBlobName,
            const std::vector<std::vector<float>>& calibratorData,
            int maxBatchSize,
            RUN_MODE mode,
            int yoloClassNum,
            int netSize);

    ~YoloV3();

    void DoInference(void* inputContext, void* outputContext);

protected:
    Trt* mNet;
    
    int mYoloClassNum;

    const int mNetWidth = 416;
    const int mNetHeight = 416;

    float* mpDetCpu;
};

#endif // YOLOV3_HPP