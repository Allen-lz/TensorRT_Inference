#include <E:/vscode/TensorRT_Inference/code/includes/scrfd.h>
// #include <E:/vscode/TensorRT_Inference/code/includes/CenterFace.h>
// #include <E:/vscode/TensorRT_Inference/code/includes/RetinaFace.h>
// #include <E:/vscode/TensorRT_Inference/code/includes/alexnet.h>
// #include <E:/vscode/TensorRT_Inference/code/includes/ghostnet.h>



int main(int argc, char **argv)
{   
    // std::string config_file = "configs/RetinaFace/config_anti.yaml";

    // std::string config_file = "configs/CenterFace/config.yaml";

    std::string config_file = "configs/scrfd/config.yaml";

    // std::string config_file = "configs/alexnet/config.yaml";

    // std::string config_file = "configs/ghostnet/config.yaml";

    // std::string folder_name = "E:/vscode/TensorRT_Inference/images";

    std::string folder_name = "E:/vscode/TensorRT_Inference/images/samples/faces_detection";

    YAML::Node root = YAML::LoadFile(config_file);

    scrfd scrfd(root["scrfd"]);
    scrfd.LoadEngine();
    scrfd.InferenceFolder(folder_name);

    // CenterFace CenterFace(root["CenterFace"]);
    // CenterFace.LoadEngine();
    // CenterFace.InferenceFolder(folder_name);

    // RetinaFace RetinaFace(root["RetinaFace"]);
    // RetinaFace.LoadEngine();
    // RetinaFace.InferenceFolder(folder_name);

    // AlexNet AlexNet(root["alexnet"]);
    // AlexNet.LoadEngine();
    // AlexNet.InferenceFolder(folder_name);

    // GhostNet GhostNet(root["ghostnet"]);
    // GhostNet.LoadEngine();
    // GhostNet.InferenceFolder(folder_name);


    return 0;
}

