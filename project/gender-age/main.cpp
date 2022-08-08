#include "gender-age.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Please design config file and folder name!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string folder_name = argv[2];
    YAML::Node root = YAML::LoadFile(config_file);
    GenderAge GenderAge(root["gender-age"]);
    GenderAge.LoadEngine();
    GenderAge.InferenceFolder(folder_name);
    return 0;
}
