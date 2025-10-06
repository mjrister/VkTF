
# VkTF

A physically based Vulkan glTF renderer in C++23.

![VkTF Demo](vktf.png)

## Features

* Modern, cross-platform foundation built with C++23, Vulkan, CMake, and vcpkg
* Physically based rendering (PBR) metallic-roughness workflow
* Data oriented glTF 2.0 asset loading pipeline
* Adaptive texture compression with Basis Universal and KTX 2.0
* Flexible shader system supporting runtime GLSL shader compilation and precompiled SPIR-V binaries
* Efficient memory management with Vulkan Memory Allocator (VMA)
* View frustum culling
* Normal mapping
* Quaternion based first-person camera
* Multisample anti-aliasing (MSAA)
* Configurable thread-safe logging

## Quickstart

This example demonstrates the high-level API for loading and rendering a scene composed of multiple glTF assets.

```C++
const vktf::Window window{"VkTF"};
vktf::Engine engine{window};

if (auto scene = engine.Load({"path/to/asset0.gltf", "path/to/asset1.gltf"})) {
  engine.Run(window, [&](const auto delta_time) mutable {
    HandleInputEvents(window, *scene, delta_time);
    engine.Render(*scene);
  });
}
```

## Requirements

This project requires CMake 3.31 and a compiler that supports the C++23 language standard. To assist with CMake configuration, building, and testing, [CMake Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) are used with [ninja](https://ninja-build.org/) as a build generator.

### Vulkan

This project requires a graphics driver with Vulkan 1.3 support. It's also built with [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp) which uses dynamic loading to avoid linking against `vulkan-1.lib`. The [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) is therefore only required for debug builds which enable [validation layers](https://vulkan.lunarg.com/doc/view/latest/windows/validation_layers.html) by default.

### Package Management

This project uses [`vcpkg`](https://vcpkg.io) to manage external dependencies. To get started, run `git submodule update --init` to clone `vcpkg` as a git submodule. Upon completion, CMake will integrate with `vcpkg` to download, compile, and link external libraries specified in the [vcpkg.json](vcpkg.json) manifest when building the project.

### Address Sanitizer

This project enables [Address Sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) (ASan) for debug builds. On Linux, this should already be available when using a version of GCC or Clang with C++23 support. On Windows, ASan needs to be installed separately which is documented [here](https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170#install-addresssanitizer).

## Build

The simplest way to build the project is to use an IDE with CMake integration. Alternatively, the project can be built from the command line using CMake presets. To use the `windows-release` preset, run:

```bash
cmake --preset windows-release
cmake --build --preset windows-release
```

A list of available configuration and build presets can be displayed by running  `cmake --list-presets` and `cmake --build --list-presets` respectively. At this time, only x64 builds are supported. Note that on Windows, `cl` and `ninja` are expected to be available in your environment path which are available by default when using the Developer Command Prompt for Visual Studio.

## Test

This project uses [Google Test](https://github.com/google/googletest) for unit testing which can be run after building the project with [CTest](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html). To use the `windows-release` preset, run:

```bash
ctest --preset windows-release
```

To see what test presets are available, run `ctest --list-presets`. Alternatively, the executable for running unit tests can be found under the `out/build/<cmake-preset>/tests` directory.

## Run

After building the project, the executable for a sample glTF viewer can be found under `out/build/<cmake-preset>/src/game` which features a first-person camera that can be translated with `WASD` keys and rotated by dragging the mouse while holding the left-click button. To close the application, press the `ESC` button.
