#include "YoloV3.h"
#include "spdlog/spdlog.h"

#include <vector>
#include <string>
#include <ctime>

#include "opencv2/opencv.hpp"

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

int main(int argc, char** argv) {
    InputParser cmdparams(argc, argv);
    const std::string& prototxt = cmdparams.getCmdOption("-prototxt");
    const std::string& caffemodel = cmdparams.getCmdOption("-caffemodel");
    const std::string& save_engine = cmdparams.getCmdOption("-save_engine");
    const std::string& img_name = cmdparams.getCmdOption("-input");
    
    cv::Mat img = cv::imread(img_name);

    std::vector<std::string> outputBlobname{"yolo-det"};
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int yoloClassNum = 80;
    int netSize = 416;
    YoloV3 yolo3(prototxt,
                caffemodel,
                save_engine,
                outputBlobname,
                calibratorData,
                maxBatchSize,
                RUN_MODE::FLOAT32,
                yoloClassNum,
                netSize);

    

    int c = 3;
    int h = 416;   //net h
    int w = 416;   //net w

    float scale = std::min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,cv::INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    cv::Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height);
    resized.copyTo(cropped(rect));

    // cv::imwrite("cropped.jpg",cropped);

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(c);
    cv::split(img_float, input_channels);

    YoloInDataSt* input = new YoloInDataSt();
    input->originalWidth = img.cols;
    input->originalHeight = img.rows;
    float* data = input->data;
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    // cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    // cv::resize(img,img,cv::Size(416,416));
    // unsigned char* data = img.data;
    // YoloInDataSt* input = new YoloInDataSt();
    // for(int n=0; n<1;n++) {
    //     for(int c=0;c<3;c++) {
    //         for(int i=0;i<416*416;i++) {
    //             input->data[i+c*416*416+n*3*416*416] = (float)data[i*3+c];
    //         }
    //     }
    // }
    YoloOutDataSt* output = new YoloOutDataSt();
    clock_t start = clock();
    yolo3.DoInference((void*)input, (void*)output);
    clock_t end = clock();
    std::cout << "inference Time : " <<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " ms" << std::endl;
    spdlog::info("------------------------");
    for(int i=0;i<output->result.size();i++) {
        Bbox bbox = output->result[i];
        spdlog::info("object in {},{},{},{}",bbox.left,bbox.top,bbox.right,bbox.bottom);
        cv::rectangle(img,cv::Point(bbox.left,bbox.top),cv::Point(bbox.right,bbox.bottom),cv::Scalar(0,255,0),1);
    }

    cv::imwrite("result.jpg",img);
    
    return 0;
}