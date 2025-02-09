#include <cstdlib>
#include <exception>
#include <iostream>

#define CGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define VMA_IMPLEMENTATION

#include <cgltf.h>
#include <stb_image.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_static_assertions.hpp>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

import game;

namespace {

constexpr std::string_view kDefaultErrorMessage = "An unknown error occurred\n";

std::ostream& operator<<(std::ostream& ostream, const std::exception& exception) {
  if (const auto* const system_error = dynamic_cast<const std::system_error*>(&exception)) {
    ostream << '[' << system_error->code() << "] ";
  }
  ostream << exception.what() << '\n';
  try {
    std::rethrow_if_nested(exception);
  } catch (const std::exception& nested_exception) {
    ostream << nested_exception;
  } catch (...) {
    ostream << kDefaultErrorMessage;
  }
  return ostream;
}

}  // namespace

int main() {
  try {
    std::ios_base::sync_with_stdio(false);  // avoid synchronizing with stdio because only standard C++ streams are used
    game::Start();
  } catch (const std::exception& exception) {
    std::cerr << exception;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << kDefaultErrorMessage;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
