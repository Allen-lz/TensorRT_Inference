#include <fstream>
#include <iostream>

#include "yaml-cpp/yaml.h"
/*
int main() {
  YAML::Emitter out;
  out << "Hello, World!";

  std::cout << "Here's the output YAML:\n" << out.c_str();

  YAML::Node config = YAML::LoadFile("configs/config.yaml");

  if (config["lastLogin"]) {
    std::cout << "Last logged in: " << config["lastLogin"].as<std::string>()
              << std::endl;
  }

  const std::string username = config["username"].as<std::string>();
  const std::string password = config["password"].as<std::string>();

  // login(username, password);
  // config["lastLogin"] = getCurrentDateTime();
  config["lastLogin"] = "2021-01-21 10:26:10";

  std::cout << "username: " << username << ", password: " << password
            << std::endl;

  std::ofstream fout("config.yaml");
  fout << config;

  return 0;
}

*/
