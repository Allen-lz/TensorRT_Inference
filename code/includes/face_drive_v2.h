#ifndef TENSORRT_INFERENCE_ALEXNET_H
#define TENSORRT_INFERENCE_ALEXNET_H

#include "classification.h"

class FaceDriveV2 : public Classification {
public:
  explicit FaceDriveV2(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_ALEXNET_H
