#include <cstdlib>
#include <exception>
#include <iostream>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

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
