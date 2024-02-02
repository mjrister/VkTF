#ifndef SRC_GAME_INCLUDE_GAME_GAME_H_
#define SRC_GAME_INCLUDE_GAME_GAME_H_

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
  void HandleKeyEvent(int key, int action) const;
  void HandleCursorEvent(float x, float y);
  void HandleScrollEvent(float y);

  Window window_;
  Engine engine_;
  Camera camera_;
  Model model_;
};

}  // namespace gfx

#endif  // SRC_GAME_INCLUDE_GAME_GAME_H_
