#include "E:/vscode/TensorRT_Inference/code/includes/alexnet.h"
// 这个就是直接继承的分类模型没有什么好解释的
AlexNet::AlexNet(const YAML::Node &config) : Classification(config) {}