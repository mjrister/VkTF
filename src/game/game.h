#ifndef GAME_GAME_H_
#define GAME_GAME_H_

#include "graphics/camera.h"
#include "graphics/engine.h"
#include "graphics/model.h"
#include "graphics/window.h"

namespace gfx {

class Game {
public:
  Game();

  void Run();

private:
  Window window_;
  Engine engine_;
  Camera camera_;
  Model model_;
};

}  // namespace gfx

#endif  // GAME_GAME_H_
