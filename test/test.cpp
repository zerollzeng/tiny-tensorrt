/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2020-03-02 15:16:08
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-05-22 11:49:13
 */

#include "Trt.h"

#include <string>
#include <sstream>
#include <vector>
#include "time.h"

void test_onnx_forward(
    const std::string& onnxModelpath,
    int maxBatchSize,
    int mode,
    std::string engineFile,
    const std::vector<std::string> customOutput) {
    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize, mode);
    onnx_net->Forward();
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
    const std::string& onnx_path = cmdparams.getCmdOption("--onnx");
    int batch_size = 1;
    int run_mode = 0;
    std::vector<std::string> custom_outputs;

    const std::string& custom_outputs_string = cmdparams.getCmdOption("--custom_outputs");
    std::istringstream stream(custom_outputs_string);
    if(custom_outputs_string != "") {
        std::string s;
        while (std::getline(stream, s, ',')) {
            custom_outputs.push_back(s);
        }
    }
    const std::string& run_mode_string = cmdparams.getCmdOption("--mode");
    if(run_mode_string != "") {
        run_mode = std::stoi(run_mode_string);
    }
    const std::string& engine_file = cmdparams.getCmdOption("--engine_file");
    const std::string& batch_size_string = cmdparams.getCmdOption("--batch_size");
    if(batch_size_string != "") {
        batch_size = std::stoi(batch_size_string);
    }
    std::cout << "**********************custom outputs: " << std::endl;
    for(size_t i=0;i<custom_outputs.size();i++) {
        std::cout << custom_outputs[i] << " ";
    }
    std::cout << std::endl;
    test_onnx_forward(onnx_path, batch_size, run_mode, engine_file, custom_outputs);
    
    return 0;
}