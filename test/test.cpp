/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2020-03-02 15:16:08
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-05-22 11:49:13
 */

#include "Trt.h"

#include <string>
#include <vector>

void test_caffe(
        const std::string& prototxt, 
        const std::string& caffeModel,
        const std::vector<std::string>& outputBlobName) {
    std::string engineFile = "";
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = 0;
    Trt* caffe_net = new Trt();
    caffe_net->CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, maxBatchSize, mode, calibratorData);
    caffe_net->Forward();
}

void test_onnx(const std::string& onnxModelpath) {
    std::string engineFile = "";
    const std::vector<std::string> customOutput;
    std::vector<std::vector<float>> calibratorData;
    int maxBatchSize = 1;
    int mode = 0;
    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode, calibratorData);
    onnx_net->Forward();
}

void test_uff(const std::string& uffModelpath) {
    std::string engineFile = "";
    std::vector<std::vector<float>> calibratorData;
    std::vector<std::string> input{"normalized_input_image_tensor"};
    std::vector<int> input_dims{1,320,320,3};
    std::vector<std::vector<int>> inputDims{input_dims};
    std::vector<std::string> output{"raw_outputs/class_predictions","raw_outputs/box_encodings"};
    int maxBatchSize = 1;
    int mode = 0;
    Trt* uff_net = new Trt();
    uff_net->CreateEngine(uffModelpath, engineFile, input, inputDims, output, maxBatchSize, mode, calibratorData);
    uff_net->Forward();
}

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

    // const std::string& onnx_path = cmdparams.getCmdOption("--onnx_path");
    // test_onnx(onnx_path);

    const std::string& prototxt = cmdparams.getCmdOption("--prototxt");         
    const std::string& caffemodel = cmdparams.getCmdOption("--caffemodel");     
    const std::string& output_blob = cmdparams.getCmdOption("--output_blob");   
    std::vector<std::string> outputBlobName;
    outputBlobName.push_back(output_blob);
    test_caffe(prototxt,caffemodel,outputBlobName);

    // test_uff("../models/frozen_inference_graph.uff");
    
    return 0;
}