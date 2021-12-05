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
#include <cassert>

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

static void split_string(std::vector<std::string>& output, std::string input_str, char c) {
    std::stringstream str_stream(input_str);
    std::string segment;
    while(std::getline(str_stream, segment, c))
    {
        output.push_back(segment);
    }
}

static void parse_specs(const std::string& input_specs_str, 
        std::vector<std::string>& input_names, std::vector<std::vector<int>>& min_shapes,
        std::vector<std::vector<int>>& opt_shapes, std::vector<std::vector<int>>& max_shapes) {
    std::vector<std::string> profiles;
    split_string(profiles, input_specs_str, ',');
    for(size_t i=0;i<profiles.size();i++) {
        std::vector<std::string> spec;
        split_string(spec, profiles[i], ':');
        for(size_t j=0;j<spec.size();j++) {
            // "input_1"
            if(j == 0) {
                input_names.push_back(spec[j]);
                continue;
            // "1x3x64x64"
            } else if(j == 1) {
                std::vector<std::string> shape_str;
                split_string(shape_str, spec[j], 'x');
                std::vector<int> shape;
                for(size_t k=0;k<shape_str.size();k++) {
                    shape.push_back(std::stoi(shape_str[k]));
                }
                min_shapes.push_back(shape);
                continue;
            } else if(j == 2) {
                std::vector<std::string> shape_str;
                split_string(shape_str, spec[j], 'x');
                std::vector<int> shape;
                for(size_t k=0;k<shape_str.size();k++) {
                    shape.push_back(std::stoi(shape_str[k]));
                }
                opt_shapes.push_back(shape);
                continue;
            } else if(j == 3) {
                std::vector<std::string> shape_str;
                split_string(shape_str, spec[j], 'x');
                std::vector<int> shape;
                for(size_t k=0;k<shape_str.size();k++) {
                    shape.push_back(std::stoi(shape_str[k]));
                }
                max_shapes.push_back(shape);
                continue;
            } else {
                assert(false && "should not be here");
            }

        }
    }
}

static void show_usage(std::string name) {
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t--onnx\t\tinput onnx model, must specify\n"
              << "\t--batch_size\t\tdefault is 1\n"
              << "\t--mode\t\t0 for fp32 1 for fp16 2 for int8, default is 0\n"
              << "\t--engine\t\tsaved path for engine file, if path exists, "
                  "will load the engine file, otherwise will create the engine file "
                  "after build engine. dafault is empty\n"
              << "\t--calibrate_data\t\tdata path for calibrate data which contain "
                 "npz files, default is empty\n"
              << "\t--gpu\t\tchoose your device, default is 0\n"
              << "\t--dla\t\tset dla core if you want with 0,1..., default is -1(not enable)\n"
              << "\t--input_specs\t\tset input shape when running model with dynamic shape\n"
                 "eg: --input_specs data_1:1x3x16x16:1x3x32x32:1x3x64x64,data_2:1x3x16x16:1x3x32x32:1x3x64x64"
              << "\t--log_level\t\tSeverity::kINTERNAL_ERROR = 0, Severity::kERROR = 1, Severity::kWARNING = 2, Severity::kINFO = 3,"
              << "Severity::kVERBOSE = 4, default level is <= kINFO.\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // parse args
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    InputParser cmdparams(argc, argv);

    const std::string& onnx_path = cmdparams.getCmdOption("--onnx");
    
    std::vector<std::string> custom_outputs;
    const std::string& custom_outputs_string = cmdparams.getCmdOption("--custom_outputs");
    std::istringstream stream(custom_outputs_string);
    if(custom_outputs_string != "") {
        std::string s;
        while (std::getline(stream, s, ',')) {
            custom_outputs.push_back(s);
        }
    }

    int run_mode = 0;
    const std::string& run_mode_string = cmdparams.getCmdOption("--mode");
    if(run_mode_string != "") {
        run_mode = std::stoi(run_mode_string);
    }

    const std::string& engine_file = cmdparams.getCmdOption("--engine");

    int batch_size = 1;
    const std::string& batch_size_string = cmdparams.getCmdOption("--batch_size");
    if(batch_size_string != "") {
        batch_size = std::stoi(batch_size_string);
    }

    const std::string& calibrateDataDir = cmdparams.getCmdOption("--calibrate_data");
    const std::string& calibrateCache = cmdparams.getCmdOption("--calibrate_cache");

    int device = 0;
    const std::string& device_string = cmdparams.getCmdOption("--gpu");
    if(device_string != "") {
        device = std::stoi(device_string);
    }

    int dla_core = -1;
    const std::string& dla_core_string = cmdparams.getCmdOption("--dla");
    if(dla_core_string != "") {
        dla_core = std::stoi(dla_core_string);
    }

    // --input_specs input_1:1x3x16x16:1x3x32x32:1x3x64x64,input_2:1x3x16x16:1x3x32x32:1x3x64x64
    const std::string& input_specs_str = cmdparams.getCmdOption("--input_specs");
    std::vector<std::string> input_names;
    std::vector<std::vector<int>> min_shapes;
    std::vector<std::vector<int>> opt_shapes;
    std::vector<std::vector<int>> max_shapes;
    parse_specs(input_specs_str, input_names, min_shapes, opt_shapes, max_shapes);

    // build engine
    Trt* onnx_net = new Trt();

    if(input_specs_str != "") {
        for(size_t i=0;i<input_names.size();i++) {
            std::cout << "Add profile for: " << input_names[i] << std::endl;
            onnx_net->AddDynamicShapeProfile(input_names[i],min_shapes[i], opt_shapes[i], max_shapes[i]);
        }
    }
    if(custom_outputs.size() > 0) {
        onnx_net->SetCustomOutput(custom_outputs);
    }
    onnx_net->SetDevice(device);
    onnx_net->SetDLACore(dla_core);
    if(calibrateDataDir != "" || calibrateCache != "") {
        onnx_net->SetInt8Calibrator("Int8EntropyCalibrator2", batch_size, calibrateDataDir, calibrateCache);
    }
    const std::string& log_level_string = cmdparams.getCmdOption("--log_level");
    if(log_level_string != "") {
        int log_level = std::stoi(log_level_string);
        onnx_net->SetLogLevel(log_level);
    }

    onnx_net->CreateEngine(onnx_path, engine_file, batch_size, run_mode);

    // do inference
    if(input_specs_str != "") {
        for(size_t i=0;i<input_names.size();i++) {
            onnx_net->SetBindingDimensions(opt_shapes[i], i);
        }
    }

    for(int i=0; i<10; i++) {
        auto time1 = std::chrono::steady_clock::now();
        onnx_net->Forward();
        auto time2 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
        std::cout << "TRT enqueue done, time: " << ((float)duration / 1000) << " ms." << std::endl;
    }

    delete onnx_net;
    
    return 0;
}