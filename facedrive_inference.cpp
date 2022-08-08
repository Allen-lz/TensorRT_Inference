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
// �沿���������ֵҲ����ȷ��

class Logger : public ILogger {
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    // suppress info-level messages
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
} gLogger;

int main(int argc, char** argv) {
  std::string onnx_filename = "weights/simple_merge_finetune_tongue_face_drive_v2.onnx";
  // ��������
  IBuilder* builder = createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto parser = nvonnxparser::createParser(*network, gLogger);

  // ����ONNXģ��
  parser->parseFromFile(onnx_filename.c_str(), 2);
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  printf("tensorRT load onnx model...\n");


  // ������������
  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 20);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  IExecutionContext* context = engine->createExecutionContext();


  // ��ȡ�����ʽ
  const int inputH = network->getInput(0)->getDimensions().d[2];
  const int inputW = network->getInput(0)->getDimensions().d[3];
  printf("inputH : %d, inputW: %d \n", inputH, inputW);



  // Ԥ������������
  cv::Mat image = cv::imread("images/male_test_img.jpg");
  // imshow("����ͼ��", image);

  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::Mat img2;
  image.convertTo(img2, CV_32F);
  img2 = img2 / 255;


  // ����GPU�Դ�����/���������
  void* buffers[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  int nBatchSize = 1;
  cudaMalloc(&buffers[0], nBatchSize * inputH * inputW * sizeof(float));
  cudaMalloc(&buffers[1], nBatchSize * 34 * sizeof(float));
  cudaMalloc(&buffers[2], nBatchSize * 3 * sizeof(float));
  cudaMalloc(&buffers[3], nBatchSize * 136 * sizeof(float));
  cudaMalloc(&buffers[4], nBatchSize * 1 * sizeof(float));
  cudaMalloc(&buffers[5], nBatchSize * 1 * sizeof(float));

  // ����cuda��
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  void* data = malloc(nBatchSize * inputH * inputW * sizeof(float));
  memcpy(data, img2.ptr<float>(0), inputH * inputW * sizeof(float));


  // �ڴ浽GPU�Դ�
  cudaMemcpyAsync(buffers[0], data,
                  nBatchSize * inputH * inputW * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  std::cout << "start to infer image..." << std::endl;

  // ����
  context->enqueueV2(buffers, stream, nullptr);

  // �Դ浽�ڴ�
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

  // ͬ���������ͷ���Դ
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);


  cv::Mat result_1 = cv::Mat(1, 34, CV_32F, (float*)bs);
  cv::Mat result_2 = cv::Mat(1, 3 , CV_32F, (float*)hp);
  cv::Mat result_3 = cv::Mat(1, 136, CV_32F, (float*)lm);
  cv::Mat result_4 = cv::Mat(1, 1, CV_32F, (float*)cf);
  cv::Mat result_5 = cv::Mat(1, 1, CV_32F, (float*)tg);
  

  // �������
  std::cout << "image inference finished!" << std::endl;

  context->destroy();
  engine->destroy();
  network->destroy();
  parser->destroy();
}

*/
