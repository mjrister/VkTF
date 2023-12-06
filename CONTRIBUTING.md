# CONTRIBUTING

Welcome 👋 This document outlines standards and guidelines for contributring to this project.

## Getting Started

In addition to the dependencies listed in the [prerequisites](README.md#Prerequisites) section of the README, contributing to this project will require:

- [python3](https://www.python.org/downloads/)
  - [pre-commit](https://pre-commit.com)
  - [cpplint](https://github.com/cpplint/cpplint)
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)

## Pre-commit

This project uses [pre-commit](https://pre-commit.com) to integrate essential [git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) that will validate project requirements prior to checking in. After following the installation instructions, run `pre-commit install` from the root directory which enable pre-commit hooks defined in [.pre-commit-config.yaml](.pre-commit-config.yaml).

## Code Style

This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with the following exceptions:

1. 120-character limit per line.
2. Exceptions are allowed.
3. Forward declarations are allowed.
4. C++ implementation files use `.cpp` as a file extension.
5. Public APIs are documented with [Doxygen](https://www.doxygen.nl/).

Code style is enforced with `clang-format` and `cpplint` which are run prior to check-in using pre-commit hooks.

## Commits Message Format

Commit messages follow the [semantic commit messages](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716) format.
