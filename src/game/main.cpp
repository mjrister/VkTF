#include <cstdlib>
#include <exception>
#include <iostream>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif
#include <vulkan/vulkan_static_assertions.hpp>

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

int main() {  // NOLINT(bugprone-exception-escape): std::cerr is not configured to throw exceptions
  try {
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
