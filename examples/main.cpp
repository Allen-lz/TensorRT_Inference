#include "../code/includes/scrfd.h"
#include "../code/includes/face_drive_v2.h"
#include "time.h"

int main()
{   
    // 测试图片
    std::string image_name = "E:/vscode/TensorRT_Inference/images/samples/faces_recognition/male_test_img.jpg";

    // 创建人脸检测网络
    std::string config_detection = "configs/scrfd/config.yaml";
    YAML::Node root1 = YAML::LoadFile(config_detection);
    scrfd scrfd(root1["scrfd"]);
    scrfd.LoadEngine();

    // 创建面部驱动网络
    std::string config_face_drive = "configs/face_drive_v2/config.yaml";
    YAML::Node root2 = YAML::LoadFile(config_face_drive);
    FaceDriveV2 FaceDriveV2(root2["face_drive_v2"]);
    FaceDriveV2.LoadEngine();

    // 人脸检测
    cv::Mat src_img = cv::imread(image_name);

    int kTestCount = 1000;
    double total_cost_time = 0;
    
    for (int i = 0; i < kTestCount; i++) {

      clock_t start = clock();
      std::vector<float> bbox;
      scrfd.InferenceImage(src_img, bbox);
      cv::Mat crop_img =
          src_img(cv::Range(bbox[1], bbox[3]), cv::Range(bbox[0], bbox[2]));
      // 将crop出来的图片padding到正方形
      if (crop_img.cols > crop_img.rows) {
        int padding = (crop_img.cols - crop_img.rows) / 2;
        cv::copyMakeBorder(crop_img, crop_img, padding, padding, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
      } else {
        int padding = (crop_img.rows - crop_img.cols) / 2;
        cv::copyMakeBorder(crop_img, crop_img, 0, 0, padding, padding,
                           cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
      }

      // cv::imshow("demo", crop_img);
      // cv::waitKey(0);

      // 面部驱动
      std::vector<float> face_drive_res;
      FaceDriveV2.InferenceImage(crop_img, face_drive_res);

      std::vector<float> blendershape;
      std::vector<float> headpose;
      for (int i = 0; i < 175; ++i) {
        if (i < 34)
          blendershape.push_back(face_drive_res[i]);
        else if (i < 37)
          headpose.push_back(face_drive_res[i]);
      }
      blendershape[33] = face_drive_res[174];
      clock_t end = clock();
      if (i > 0) total_cost_time += (end - start);
    }
    

    std::cout << "mean cost time:" << total_cost_time / (kTestCount - 1)
              << std::endl;

    return 0;
   
}

