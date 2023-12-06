#include "graphics/mesh.cpp"  // NOLINT(build/include)

#include <array>

#include <gtest/gtest.h>

namespace {

TEST(MeshTest, InitializationWithInvalidNumberOfPositionsThrowsAnException) {
  for (auto i = 0; i <= 4; ++i) {
    if (i != 3) {
      EXPECT_THROW((qem::Mesh{std::vector<glm::vec3>(i), {}, {}, {}}), std::invalid_argument);
    }
  }
}

TEST(MeshTest, InitializationWithInvalidNumberOfTextureCoordinatesThrowsAnException) {
  constexpr std::array<glm::vec3, 3> kPositions{};
  for (auto i = 1; i <= 4; ++i) {
    if (i != 3) {
      EXPECT_THROW((qem::Mesh{kPositions, std::vector<glm::vec2>(i), {}, {}}), std::invalid_argument);
    }
  }
}

TEST(MeshTest, InitializationWithInvalidNumberOfNormalsThrowsAnException) {
  constexpr std::array<glm::vec3, 3> kPositions{};
  for (auto i = 1; i <= 4; ++i) {
    if (i != 3) {
      EXPECT_THROW((qem::Mesh{kPositions, {}, std::vector<glm::vec3>(i), {}}), std::invalid_argument);
    }
  }
}

TEST(MeshTest, InitializationWithInvalidIndicesThrowsAnException) {
  constexpr std::array<glm::vec3, 3> kPositions{};
  for (auto i = 1; i <= 4; ++i) {
    if (i != 3) {
      EXPECT_THROW((qem::Mesh{kPositions, {}, {}, std::vector<GLuint>(i)}), std::invalid_argument);
    }
  }
}

TEST(MeshTest, InitializationWithCorrectNumberOfPositionsTextureCoordinatesAndNormalsDoesNotThrowAnException) {
  const std::vector<glm::vec3> positions(3);
  const std::vector<glm::vec2> texture_coordinates(3);
  const std::vector<glm::vec3> normals(3);
  EXPECT_NO_THROW((qem::Mesh{positions, texture_coordinates, normals, {}}));
}

TEST(MeshTest, InitializationWithCorrectNumberOfPositionsTextureCoordinatesNormalsAndIndicesDoesNotThrowAnException) {
  constexpr std::array<glm::vec3, 4> kPositions{};
  constexpr std::array<glm::vec2, 2> kTextureCoordinates{};
  constexpr std::array<glm::vec3, 5> kNormals{};
  constexpr std::array<GLuint, 3> kIndices{};
  EXPECT_NO_THROW((qem::Mesh{kPositions, kTextureCoordinates, kNormals, kIndices}));
}
}  // namespace
