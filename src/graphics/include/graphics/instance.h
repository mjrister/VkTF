#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_INSTANCE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_INSTANCE_H_

#include <vulkan/vulkan.hpp>

#include "graphics/window.h"

namespace gfx {

class Instance {
public:
  static constexpr auto kApiVersion = vk::ApiVersion13;

  Instance();

  [[nodiscard]] vk::Instance operator*() const noexcept { return *instance_; }

private:
  vk::UniqueInstance instance_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_INSTANCE_H_
