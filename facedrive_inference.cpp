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


/*
using namespace nvinfer1;
using namespace nvonnxparser;
// 面部驱动推理的值也是正确的

class Logger : public ILogger {
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    // suppress info-level messages
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
} gLogger;

int main(int argc, char** argv) {
  std::string onnx_filename = "weights/simple_merge_finetune_tongue_face_drive_v2.onnx";
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
  printf("tensorRT load onnx model...\n");


  // 创建推理引擎
  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 20);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  IExecutionContext* context = engine->createExecutionContext();


  // 获取输入格式
  const int inputH = network->getInput(0)->getDimensions().d[2];
  const int inputW = network->getInput(0)->getDimensions().d[3];
  printf("inputH : %d, inputW: %d \n", inputH, inputW);



  // 预处理输入数据
  cv::Mat image = cv::imread("images/male_test_img.jpg");
  // imshow("输入图像", image);

  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::Mat img2;
  image.convertTo(img2, CV_32F);
  img2 = img2 / 255;


  // 创建GPU显存输入/输出缓冲区
  void* buffers[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  int nBatchSize = 1;
  cudaMalloc(&buffers[0], nBatchSize * inputH * inputW * sizeof(float));
  cudaMalloc(&buffers[1], nBatchSize * 34 * sizeof(float));
  cudaMalloc(&buffers[2], nBatchSize * 3 * sizeof(float));
  cudaMalloc(&buffers[3], nBatchSize * 136 * sizeof(float));
  cudaMalloc(&buffers[4], nBatchSize * 1 * sizeof(float));
  cudaMalloc(&buffers[5], nBatchSize * 1 * sizeof(float));

  // 创建cuda流
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  void* data = malloc(nBatchSize * inputH * inputW * sizeof(float));
  memcpy(data, img2.ptr<float>(0), inputH * inputW * sizeof(float));


  // 内存到GPU显存
  cudaMemcpyAsync(buffers[0], data,
                  nBatchSize * inputH * inputW * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  std::cout << "start to infer image..." << std::endl;

  // 推理
  context->enqueueV2(buffers, stream, nullptr);

  // 显存到内存
  float bs[34];
  cudaMemcpyAsync(bs, buffers[1], 1 * 34 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  float hp[3];
  cudaMemcpyAsync(hp, buffers[2], 1 * 3 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  float lm[136];
  cudaMemcpyAsync(lm, buffers[3], 1 * 136 * sizeof(float), 
                  cudaMemcpyDeviceToHost, stream);
  float cf[1];
  cudaMemcpyAsync(cf, buffers[4], 1 * 1 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  float tg[1];
  cudaMemcpyAsync(tg, buffers[5], 1 * 1 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  // 同步结束，释放资源
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);


  cv::Mat result_1 = cv::Mat(1, 34, CV_32F, (float*)bs);
  cv::Mat result_2 = cv::Mat(1, 3 , CV_32F, (float*)hp);
  cv::Mat result_3 = cv::Mat(1, 136, CV_32F, (float*)lm);
  cv::Mat result_4 = cv::Mat(1, 1, CV_32F, (float*)cf);
  cv::Mat result_5 = cv::Mat(1, 1, CV_32F, (float*)tg);
  

  // 解析输出
  std::cout << "image inference finished!" << std::endl;

  context->destroy();
  engine->destroy();
  network->destroy();
  parser->destroy();
}

*/
