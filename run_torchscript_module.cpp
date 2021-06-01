#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <math.h>
#include"cnpy.h"
#include <string>
#include <map>

using namespace cv;
using namespace std;

// function for load model
torch::jit::script::Module load_model(const char* model_path){
  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module;
  module = torch::jit::load(model_path); 
  return module;
}


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    cerr << "usage: run_torchscript_module <path-to-torchscript-module>>" << endl;
    return -1;
  }

  torch::jit::script::Module model;
  cout << "Torchscript model path: " << argv[1] << endl;

  try {
    // Load torchscript model
    model = load_model(argv[1]);
    cout << "Model loaded. " << endl;
    }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    std::cerr << e.msg() << std::endl;
    return -1;
  }

  // Access model parameters
  cout << endl << "Model named parameters" << endl;
  for(auto p : model.named_parameters()){
    cout << p.name << ":" << p.value<< endl;
  }

  // Access member variable
  cout << endl << "Model member variable: " << endl;
  cout << model.attr("var") << endl;

  // Input 
  at::Tensor sample_input = torch::randint(/*high=*/10, {1, 17, 2});    
  std::vector<torch::jit::IValue> input_vec;
  input_vec.push_back(sample_input);  // Input: 1 x 17 x 2

  // Mode prediction
  at::Tensor sample_out = model.forward(input_vec).toTensor();
  cout << "Model output:\n" << sample_out << endl;

  cout << "Done\n";

  return 0;
}

