#include <cstdlib>
#include <exception>
#include <iostream>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include "game/game.h"

int main() {  // NOLINT(bugprone-exception-escape)
  try {
    gfx::Game game;
    game.Run();
  } catch (const std::system_error& e) {
    std::cerr << '[' << e.code() << "] " << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "An unknown error occurred\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
