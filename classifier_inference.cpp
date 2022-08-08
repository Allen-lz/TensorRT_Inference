#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
/*
using namespace cv;
using namespace dnn;
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
// 推理的值也是正确的

class Logger : public ILogger {
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    // suppress info-level messages
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
} gLogger;

int main(int argc, char** argv) {
  std::string onnx_filename = "weights/fixed_male_hair_classifier.onnx";
  // 创建网络
  IBuilder* builder = createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto parser = nvonnxparser::createParser(*network, gLogger);

  // 解析ONNX模型
  parser->parseFromFile(onnx_filename.c_str(), 2);
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  printf("tensorRT load onnx mnist model...\n");


  // 创建推理引擎
  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 20);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  IExecutionContext* context = engine->createExecutionContext();


  // 获取输入与输出名称，格式
  const char* input_blob_name = network->getInput(0)->getName();
  const char* output_blob_name = network->getOutput(0)->getName();
  printf("input_blob_name : %s \n", input_blob_name);
  printf("output_blob_name : %s \n", output_blob_name);

  const int inputH = network->getInput(0)->getDimensions().d[2];
  const int inputW = network->getInput(0)->getDimensions().d[3];
  printf("inputH : %d, inputW: %d \n", inputH, inputW);

  // 预处理输入数据
  cv::Mat image = cv::imread("images/male_test_img.jpg");
  // imshow("输入图像", image);

  // cv::cvtColor(image, image, CV_BGR2RGB);
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int input_batch = 1;
  int input_channel = 3;
  int input_height = 224;
  int input_width = 224;

  // 准备好input_data_host和input_data_device，分别表示内存中的数据指针和显存中的数据指针
  // 一会儿将预处理过的图像数据搬运到GPU
  int input_numel = input_batch * input_channel * input_height * input_width;
  float* input_data_host = nullptr;
  float* input_data_device = nullptr;

  cudaMallocHost(&input_data_host, input_numel * sizeof(float));
  cudaMalloc(&input_data_device, input_numel * sizeof(float));

  // 图片读取与预处理，与之前python中的预处理方式一致：
  // BGR->RGB、归一化/除均值减标准差
  float mean[] = {0.406, 0.456, 0.485};
  float std[] = {0.225, 0.224, 0.229};

  cv::resize(image, image, cv::Size(input_width, input_height));

  int image_area = image.cols * image.rows;
  unsigned char* pimage = image.data;
  float* phost_b = input_data_host + image_area * 0;
  float* phost_g = input_data_host + image_area * 1;
  float* phost_r = input_data_host + image_area * 2;
  for (int i = 0; i < image_area; ++i, pimage += 3) {
    *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
    *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
    *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
  }


  // img2 = img2 / 255;


  // 创建GPU显存输入/输出缓冲区
  void* buffers[2] = {NULL, NULL};
  int nBatchSize = 1;
  int nOutputSize = 6;
  cudaMalloc(&buffers[0], nBatchSize * inputH * inputW * 3 * sizeof(float));
  cudaMalloc(&buffers[1], nBatchSize * nOutputSize * sizeof(float));

  // 创建cuda流
  
  void* data = malloc(nBatchSize * inputH * inputW * 3 * sizeof(float));
  // memcpy(data, img2.ptr<float>(0), inputH * inputW * 3 * sizeof(float));


  // 内存到GPU显存
  cudaMemcpyAsync(buffers[0], data,
                  nBatchSize * inputH * inputW * 3 * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  std::cout << "start to infer image..." << std::endl;

  // 推理
  context->enqueueV2(buffers, stream, nullptr);

  // 显存到内存
  float prob[6];
  cudaMemcpyAsync(prob, buffers[1], 1 * nOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

  // 同步结束，释放资源
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  // 解析输出
  std::cout << "image inference finished!" << std::endl;

 
  cv::Mat result = cv::Mat(1, 6, CV_32F, (float*)prob);
  float max = result.at<float>(0, 0);
  int index = 0;
  for (int i = 0; i < 6; i++) {
    if (max < result.at<float>(0, i)) {
      max = result.at<float>(0, i);
      index = i;
    }
  }
  std::cout << prob[0] << " " << prob[1] << " " << prob[2] << index << std::endl;
  std::cout << prob[3] << " " << prob[4] << " " << prob[5] << index
            << std::endl;

  context->destroy();
  engine->destroy();
  network->destroy();
  parser->destroy();
}
*/