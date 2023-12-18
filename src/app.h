#ifndef SRC_APP_H_
#define SRC_APP_H_

#include "graphics/camera.h"
#include "graphics/engine.h"
#include "graphics/model.h"
#include "graphics/window.h"

namespace gfx {

class App {
public:
  App();

  void Run();

private:
  void HandleKeyEvent(const int key, const int action) const;

  Window window_;
  Engine engine_;
  Camera camera_;
  Model model_;
};

}  // namespace gfx

#endif  // SRC_APP_H_
