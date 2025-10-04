
# VkTF

A cross-platform, physically based Vulkan glTF renderer written in C++23.

## Features

* Cross-platform Vulkan renderer written in C++23 with a focus on performance, type safety, and a modular architecture
* Physically based rendering (PBR) based on the metallic-roughness workflow
* Data-oriented [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) asset loading pipeline with support for loading multiple models to create a combined scene
* Multithreaded [KTX 2.0](https://www.khronos.org/ktx/) loading with support for transcoding [Basis Universal](https://github.com/BinomialLLC/basis_universal) supercompressed textures at runtime
* Runtime GLSL shader compilation using [glslang](https://github.com/KhronosGroup/glslang) with support for loading precompiled SPIR-V binaries
* Robust memory management with [Vulkan Memory Allocator (VMA)](https://gpuopen.com/vulkan-memory-allocator/)
* Quaternion based first-person camera implementation
* View frustum culling
* Normal mapping
* Multisample anti-aliasing (MSAA)
* Configurable and thread-safe logging implementation

## Requirements

This project requires CMake 3.31 and a compiler that supports the C++23 language standard. To assist with CMake configuration, building, and testing, [CMake Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) are used with [ninja](https://ninja-build.org/) as a build generator.

### Vulkan

This project requires a graphics driver with Vulkan 1.3 support and is built with [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp) which uses a dynamic loading implementation to avoid statically linking against `vulkan-1.lib`. Therefore, it's not required to install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) for release builds; however, it's still required for debug builds which enable [validation layers](https://vulkan.lunarg.com/doc/view/latest/windows/validation_layers.html) by default.

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

To see what test presets are available, run `ctest --list-presets`.  Alternatively, tests can be run from the separate `tests` executable which is built with the project.
