#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"

// 将关键点的给用上的话会报莫名其妙的错误

/*
using namespace nvinfer1;
using namespace nvonnxparser;


class Logger : public ILogger {
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    // suppress info-level messages
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
} gLogger;


int main(int argc, char** argv) {
  std::string onnx_filename = "weights/scrfd_500m_kps.onnx";
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

  const char* output_1 = network->getOutput(0)->getName();
  const char* output_2 = network->getOutput(1)->getName();
  const char* output_3 = network->getOutput(2)->getName();

  const char* output_4 = network->getOutput(3)->getName();
  const char* output_5 = network->getOutput(4)->getName();
  const char* output_6 = network->getOutput(5)->getName();

  // 关键点的暂时用不上
  // const char* output_7 = network->getOutput(6)->getName();
  // const char* output_8 = network->getOutput(7)->getName();
  // const char* output_9 = network->getOutput(8)->getName();

  // 打印每个输入输出的节点对应的名字
  // printf("input_blob_name : %s \n", input_blob_name);
  // printf("output_1 : %s \n", output_1);
  // printf("output_2 : %s \n", output_2);
  // printf("output_3 : %s \n", output_3);
  // printf("output_4 : %s \n", output_4);
  // printf("output_5 : %s \n", output_5);
  // printf("output_6 : %s \n", output_6);
  // printf("output_7 : %s \n", output_7);
  // printf("output_8 : %s \n", output_8);
  // printf("output_9 : %s \n", output_9);

  // output_1 : out0
  // output_2 : out1
  // output_3 : out2

  // output_4 : out3
  // output_5 : out4
  // output_6 : out5

  // output_7 : out6
  // output_8 : out7
  // output_9 : out8

  // 获取输入的shape
  const int inputH = network->getInput(0)->getDimensions().d[2];
  const int inputW = network->getInput(0)->getDimensions().d[3];
  printf("inputH : %d, inputW: %d \n", inputH, inputW);


  // =====================================Inference======================================
  // 预处理输入数据
  cv::Mat image = cv::imread("images/male_test_img.jpg");
  // imshow("输入图像", image);


  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::Mat img2;
  image.convertTo(img2, CV_32F);
  img2 = (img2 - 127.5) / 128.0;

  // 创建GPU显存输入/输出缓冲区
  void* buffers[7] = {NULL, NULL, NULL, NULL, NULL,
                       NULL, NULL};
  int nBatchSize = 1;
  cudaMalloc(&buffers[0], nBatchSize * inputH * inputW * sizeof(float));

  cudaMalloc(&buffers[1], nBatchSize * 12800 * sizeof(float));
  cudaMalloc(&buffers[2], nBatchSize * 3200 * sizeof(float));
  cudaMalloc(&buffers[3], nBatchSize * 800 * sizeof(float));

  cudaMalloc(&buffers[4], nBatchSize * 12800 * 4 * sizeof(float));
  cudaMalloc(&buffers[5], nBatchSize * 3200 * 4 * sizeof(float));
  cudaMalloc(&buffers[6], nBatchSize * 800 * 4 * sizeof(float));

  // cudaMalloc(&buffers[7], nBatchSize * 12800 * 10 * sizeof(float));
  // cudaMalloc(&buffers[8], nBatchSize * 3200 * 10 * sizeof(float));
  // cudaMalloc(&buffers[9], nBatchSize * 800 * 10 * sizeof(float));

  // 创建cuda流
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  void* data = malloc(nBatchSize * inputH * inputW * sizeof(float));

  // 将待测数据拷贝到内存中
  memcpy(data, img2.ptr<float>(0), inputH * inputW * sizeof(float));
  // 内存到GPU显存
  cudaMemcpyAsync(buffers[0], data,
                  nBatchSize * inputH * inputW * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  std::cout << "start to infer image..." << std::endl;

  // 推理
  context->enqueueV2(buffers, stream, nullptr);

  printf("推理成功!!! \n");

  // 显存到内存

  float prob_1[12800];
  cudaMemcpyAsync(prob_1, buffers[1], nBatchSize * 12800 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  
  
  float prob_2[3200];
  cudaMemcpyAsync(prob_2, buffers[2], nBatchSize * 3200 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  

  float prob_3[800];
  cudaMemcpyAsync(prob_3, buffers[3], nBatchSize * 800 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  

  float prob_4[12800 * 4];
  cudaMemcpyAsync(prob_4, buffers[4], nBatchSize * 12800 * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  

  float prob_5[3200 * 4];
  cudaMemcpyAsync(prob_5, buffers[5], nBatchSize * 3200 * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  

  float prob_6[800 * 4];
  cudaMemcpyAsync(prob_6, buffers[6], nBatchSize * 800 * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
 
  printf("转存成功!!! \n");

  
  // 同步结束，释放资源
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cv::Mat result_1 = cv::Mat(12800, 1, CV_32F, (float*)prob_1);
  cv::Mat result_2 = cv::Mat(3200, 1, CV_32F, (float*)prob_2);
  cv::Mat result_3 = cv::Mat(800, 1, CV_32F, (float*)prob_3);
  cv::Mat result_4 = cv::Mat(12800, 4, CV_32F, (float*)prob_4);
  cv::Mat result_5 = cv::Mat(3200, 4, CV_32F, (float*)prob_5);
  cv::Mat result_6 = cv::Mat(800, 4, CV_32F, (float*)prob_6);





  context->destroy();
  engine->destroy();
  network->destroy();
  parser->destroy();

}

*/


