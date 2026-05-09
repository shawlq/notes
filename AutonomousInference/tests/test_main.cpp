#include <exception>
#include <iostream>

void testConfigLoader();
void testPipelineRuns();

int main() {
  try {
    testConfigLoader();
    testPipelineRuns();
  } catch (const std::exception& e) {
    std::cerr << "test failed: " << e.what() << std::endl;
    return 1;
  }
  std::cout << "all tests passed" << std::endl;
  return 0;
}
