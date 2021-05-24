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
#include <chrono>

class InputParser{                                                              
    public:                                                                     
        InputParser (int &argc, char **argv){                                   
            for (int i=1; i < argc; ++i)                                        
                this->tokens.push_back(std::string(argv[i]));                   
        }                                                                       
        const std::string& getCmdOption(const std::string &option) const{       
            std::vector<std::string>::const_iterator itr;                       
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option); 
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){      
                return *itr;                                                    
            }                                                                   
            static const std::string empty_string("");                          
            return empty_string;                                                
        }                                                                       
        bool cmdOptionExists(const std::string &option) const{                  
            return std::find(this->tokens.begin(), this->tokens.end(), option)  
                   != this->tokens.end();                                       
        }                                                                       
    private:                                                                    
        std::vector <std::string> tokens;                                       
};  

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t--onnx\t\tinput onnx model, must specify\n"
              << "\t--batch_size\t\tdefault is 1\n"
              << "\t--mode\t\t0 for fp32 1 for fp16 2 for int8, default is 0\n"
              << "\t--engine\t\tsaved path for engine file, if path exists,"
              << "will load the engine file, otherwise will create the engine file"
              <<  "after build engine. dafault is empty\n"
              << "\t--calibrate_data\t\tdata path for calibrate data which contain npz files"
              << "default is empty\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
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
    const std::string& engine_file = cmdparams.getCmdOption("--engine");
    const std::string& batch_size_string = cmdparams.getCmdOption("--batch_size");
    if(batch_size_string != "") {
        batch_size = std::stoi(batch_size_string);
    }
    const std::string& calibrateDataDir = cmdparams.getCmdOption("--calibrate_data");


    Trt* onnx_net = new Trt();
    if(calibrateDataDir != "") {
        onnx_net->SetInt8Calibrator("Int8EntropyCalibrator2", batch_size, calibrateDataDir);
    }
    onnx_net->CreateEngine(onnx_path, engine_file, custom_outputs, batch_size, run_mode);
    auto time1 = std::chrono::steady_clock::now();
    onnx_net->Forward();
    auto time2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
    
    return 0;
}