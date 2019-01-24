#include <iostream>

#include "task_graph.hpp"

void hello_world(void*, void*) noexcept {
  std::cout << "hello world" << std::endl;
}

int main() {
  run_generation_task(&hello_world, nullptr);
}
