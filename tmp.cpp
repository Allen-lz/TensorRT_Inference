#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <sstream>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "NvInfer.h"
#include "NvOnnxParser.h"


/*
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;
using namespace dnn;
using namespace std;


class Logger : public ILogger {
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    // suppress info-level messages
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
} gLogger;


// 配置类
struct Net_config {
  float confThreshold;  // class Confidence threshold
  float nmsThreshold;   // Non-maximum suppression threshold
  int inpWidth;
  int inpHeight;
  string modelfile;
};

class SCRFD {
 public:
  SCRFD(Net_config config);
  void detect(Mat& frame);

 private:
  const float stride[3] = {8.0, 16.0, 32.0};
  int inpWidth;
  int inpHeight;
  float confThreshold;
  float nmsThreshold;
  const bool keep_ratio = true;

  int inputH;
  int inputW;
  IParser* parser = nullptr;
  nvinfer1::INetworkDefinition* network = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  void* buffers[7] = {NULL, NULL, NULL, NULL, NULL, 
                       NULL, NULL};
  cudaStream_t stream;
  void* data = nullptr;
  const int nBatchSize = 1;

  vector<cv::Mat> res;

  Net net;
  Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
};

SCRFD::SCRFD(Net_config config) {
  this->confThreshold = config.confThreshold;
  this->nmsThreshold = config.nmsThreshold;
  this->inpWidth = config.inpWidth;
  this->inpHeight = config.inpHeight;


  // 将这个Net换成TensorRT的
  std::string onnx_filename = config.modelfile;

  // 创建网络
  IBuilder* builder = createInferBuilder(gLogger);
  this->network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  this->parser = nvonnxparser::createParser(*(this->network), gLogger);

  // 解析ONNX模型
  parser->parseFromFile(onnx_filename.c_str(), 2);
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  printf("tensorRT load onnx mnist model...\n");

  // 创建推理引擎
  IBuilderConfig* trt_config = builder->createBuilderConfig();
  trt_config->setMaxWorkspaceSize(1 << 20);
  trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  this->engine = builder->buildEngineWithConfig(*network, *trt_config);
  this->context = this->engine->createExecutionContext();


  // 获取输入与输出名称，格式 
  const char* input_blob_name = network->getInput(0)->getName();
  // cfg
  // const char* output_1 = network->getOutput(0)->getName();
  // const char* output_2 = network->getOutput(1)->getName();
  // const char* output_3 = network->getOutput(2)->getName();
  // bboxes
  // const char* output_4 = network->getOutput(3)->getName();
  // const char* output_5 = network->getOutput(4)->getName();
  // const char* output_6 = network->getOutput(5)->getName();
  // landmarks
  // const char* output_7 = network->getOutput(6)->getName();
  // const char* output_8 = network->getOutput(7)->getName();
  // const char* output_9 = network->getOutput(8)->getName();
  // 获得输入的H和W
  this->inputH = network->getInput(0)->getDimensions().d[2];
  this->inputW = network->getInput(0)->getDimensions().d[3];


  // 创建GPU显存输入/输出缓冲区, 为缓冲区分配内存
  cudaMalloc(&(this->buffers[0]), this->nBatchSize * this->inputH * this->inputW * sizeof(float));

  cudaMalloc(&(this->buffers[1]), this->nBatchSize * 12800 * sizeof(float));
  cudaMalloc(&(this->buffers[2]), this->nBatchSize * 3200 * sizeof(float));
  cudaMalloc(&(this->buffers[3]), this->nBatchSize * 800 * sizeof(float));

  cudaMalloc(&(this->buffers[4]), this->nBatchSize * 12800 * 4 * sizeof(float));
  cudaMalloc(&(this->buffers[5]), this->nBatchSize * 3200 * 4 * sizeof(float));
  cudaMalloc(&(this->buffers[6]), this->nBatchSize * 800 * 4 * sizeof(float));

  // cudaMalloc(&(this->buffers[7]), this->nBatchSize * 12800 * 10 * sizeof(float));
  // cudaMalloc(&(this->buffers[8]), this->nBatchSize * 3200 * 10 * sizeof(float));
  // cudaMalloc(&(this->buffers[9]), this->nBatchSize * 800 * 10 * sizeof(float));


  // 创建cuda流
  cudaStreamCreate(&(this->stream));
  this->data = malloc(this->nBatchSize * this->inputH * this->inputW * sizeof(float));


  this->net = readNet(config.modelfile);
}

Mat SCRFD::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left) {
  int srch = srcimg.rows, srcw = srcimg.cols;
  *newh = this->inpHeight;
  *neww = this->inpWidth;
  Mat dstimg;
  if (this->keep_ratio && srch != srcw) {
    float hw_scale = (float)srch / srcw;
    if (hw_scale > 1) {
      *newh = this->inpHeight;
      *neww = int(this->inpWidth / hw_scale);
      resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
      *left = int((this->inpWidth - *neww) * 0.5);
      copyMakeBorder(dstimg, dstimg, 0, 0, *left,
                     this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
    } else {
      *newh = (int)this->inpHeight * hw_scale;
      *neww = this->inpWidth;
      resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
      *top = (int)(this->inpHeight - *newh) * 0.5;
      copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0,
                     BORDER_CONSTANT, 0);
    }
  } else {
    resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
  }
  return dstimg;
}

void SCRFD::detect(Mat& frame) {
  int newh = 0, neww = 0, padh = 0, padw = 0;
  Mat img = this->resize_image(frame, &newh, &neww, &padh, &padw);
  Mat blob;
  // 对输入的数据进行预处理
  blobFromImage(img, blob, 1 / 128.0, Size(this->inpWidth, this->inpHeight),
                Scalar(127.5, 127.5, 127.5), true, false);

  // =======================================在这里插入TensorRT的推理==================================
  // 将待测数据拷贝到内存中
  cv::Mat img2;
  blob.convertTo(img2, CV_32F);
  memcpy(this->data, img2.ptr<float>(0), this->inputH * this->inputW * sizeof(float));
  // 内存到GPU显存
  cudaMemcpyAsync(this->buffers[0], this->data,
                  this->nBatchSize * this->inputH * this->inputW * sizeof(float),
                  cudaMemcpyHostToDevice, this->stream);
  std::cout << "start to infer image..." << std::endl;

  // 推理
  this->context->enqueueV2(this->buffers, this->stream, nullptr);

  // 显存到内存
  float prob_1[12800];
  cudaMemcpyAsync(prob_1, this->buffers[1], 
                  this->nBatchSize * 12800 * sizeof(float),
                  cudaMemcpyDeviceToHost, this->stream);
  

  float prob_2[3200];
  cudaMemcpyAsync(prob_2, this->buffers[2], 
                  this->nBatchSize * 3200 * sizeof(float),
                  cudaMemcpyDeviceToHost, this->stream);
  

  float prob_3[800];
  cudaMemcpyAsync(prob_3, this->buffers[3], 
                  this->nBatchSize * 800 * sizeof(float),
                  cudaMemcpyDeviceToHost, this->stream);
  

  float prob_4[12800 * 4];
  cudaMemcpyAsync(prob_4, this->buffers[4], 
                  this->nBatchSize * 12800 * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost, this->stream);
  

  float prob_5[3200 * 4];
  cudaMemcpyAsync(prob_5, this->buffers[5],
                  this->nBatchSize * 3200 * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost, this->stream);
  

  float prob_6[800 * 4];
  cudaMemcpyAsync(prob_6, this->buffers[6],
                  this->nBatchSize * 800 * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost, this->stream);


  cudaStreamSynchronize(this->stream);
  cudaStreamDestroy(this->stream);

  cv::Mat result_1 = cv::Mat(12800, 1, CV_32F, (float*)prob_1);
  this->res.push_back(result_1);
  cv::Mat result_2 = cv::Mat(3200, 1, CV_32F, (float*)prob_2);
  this->res.push_back(result_2);
  cv::Mat result_3 = cv::Mat(800, 1, CV_32F, (float*)prob_3);
  this->res.push_back(result_3);
  cv::Mat result_4 = cv::Mat(12800, 4, CV_32F, (float*)prob_4);
  this->res.push_back(result_4);
  cv::Mat result_5 = cv::Mat(3200, 4, CV_32F, (float*)prob_5);
  this->res.push_back(result_5);
  cv::Mat result_6 = cv::Mat(800, 4, CV_32F, (float*)prob_6);
  this->res.push_back(result_6);
  // ====================================================================================

  this->net.setInput(blob);
  vector<Mat> outs;
  this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

  /////generate proposals
  vector<float> confidences;
  vector<Rect> boxes;
  vector<vector<int>> landmarks;
  float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
  int n = 0, i = 0, j = 0, k = 0, l = 0;
  for (n = 0; n < 3; n++) {
    int num_grid_x = (int)(this->inpWidth / this->stride[n]);
    int num_grid_y = (int)(this->inpHeight / this->stride[n]);
    

    float* pdata_score = (float*)(this->res)[n].data;  /// score
    float* pdata_bbox = (float*)(this->res)[n + 3].data;  /// bounding box


    // float* pdata_score = (float*)outs[n * 3].data;     /// score
    // float* pdata_bbox = (float*)outs[n * 3 + 1].data;  /// bounding box

    // [12800 x 1]
    // [3200 x 1]
    // [800 x 1]


    float* pdata_kps = (float*)outs[n * 3 + 2].data;   /// face landmark

    for (i = 0; i < num_grid_y; i++) {
      for (j = 0; j < num_grid_x; j++) {
        for (k = 0; k < 2; k++) {
          if (pdata_score[0] > this->confThreshold) {
            const int xmin =
                (int)(((j - pdata_bbox[0]) * this->stride[n] - padw) * ratiow);
            const int ymin =
                (int)(((i - pdata_bbox[1]) * this->stride[n] - padh) * ratioh);
            const int width = (int)((pdata_bbox[2] + pdata_bbox[0]) *
                                    this->stride[n] * ratiow);
            const int height = (int)((pdata_bbox[3] + pdata_bbox[1]) *
                                     this->stride[n] * ratioh);
            confidences.push_back(pdata_score[0]);

            std::cout << pdata_score[0] << " " << xmin << " " << ymin << " " << width << " " << height << std::endl;
            boxes.push_back(Rect(xmin, ymin, width, height));
            vector<int> landmark(10, 0);
            for (l = 0; l < 10; l += 2) {
              landmark[l] =
                  (int)(((j + pdata_kps[l]) * this->stride[n] - padw) * ratiow);
              landmark[l + 1] =
                  (int)(((i + pdata_kps[l + 1]) * this->stride[n] - padh) *
                        ratioh);
            }
            landmarks.push_back(landmark);
          }
          pdata_score++;
          pdata_bbox += 4;
          pdata_kps += 10;
        }
      }
    }
  }

  // Perform non maximum suppression to eliminate redundant overlapping boxes
  // with lower confidences
  vector<int> indices;
  dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold,
                indices);
  for (i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    Rect box = boxes[idx];
    rectangle(frame, Point(box.x, box.y),
              Point(box.x + box.width, box.y + box.height), Scalar(0, 0, 255),
              2);
    for (k = 0; k < 10; k += 2) {
      circle(frame, Point(landmarks[idx][k], landmarks[idx][k + 1]), 1,
             Scalar(0, 255, 0), -1);
    }

    // Get the label for the class name and its confidence
    string label = format("%.2f", confidences[idx]);
    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize =
        getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = max(box.y, labelSize.height);
    // rectangle(frame, Point(left, top - int(1.5 * labelSize.height)),
    // Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255,
    // 0), FILLED);
    putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.75,
            Scalar(0, 255, 0), 1);
  }

  this->res.clear();
}

int main() {
  Net_config cfg = {
      0.5, 0.5, 640, 640,
      "weights/scrfd_500m_kps.onnx"};  /// choices =
                                       /// ["weights/scrfd_500m_kps.onnx",
                                       /// "weights/scrfd_2.5g_kps.onnx",
                                       /// "weights/scrfd_10g_kps.onnx"]
  SCRFD mynet(cfg);
  string imgpath = "images/male_test_img.jpg";
  Mat srcimg = imread(imgpath);
  mynet.detect(srcimg);

  static const string kWinName = "Deep learning object detection in OpenCV";
  namedWindow(kWinName, WINDOW_NORMAL);
  imshow(kWinName, srcimg);
  waitKey(0);
  destroyAllWindows();
}
*/