#include "graphics/obj_loader.cpp"  // NOLINT(build/include)

#include <gtest/gtest.h>

namespace {

TEST(ObjLoaderTest, TrimStringWithOnlyWhitespaceReturnsTheEmptyString) {
  static constexpr auto* kLine = "     ";
  static_assert(Trim(kLine).empty());
}

TEST(ObjLoaderTest, TrimStringRemovesWhitespaceFromBothEndsOfTheString) {
  static constexpr auto* kLine = "\t  Hello, World!  \t";
  static_assert(Trim(kLine) == "Hello, World!");
}

TEST(ObjLoaderTest, SplitEmptyStringReturnsAnEmptyList) {
  static constexpr auto* kLine = "";
  static constexpr auto* kDelimiter = " ";
  EXPECT_TRUE(Split(kLine, kDelimiter).empty());
}

TEST(ObjLoaderTest, SplitStringWithOnlyTheDelimiterReturnsAnEmptyList) {
  static constexpr auto* kLine = "   ";
  static constexpr auto* kDelimiter = " ";
  EXPECT_TRUE(Split(kLine, kDelimiter).empty());
}

TEST(ObjLoaderTest, SplitStringWithoutDelimiterReturnsListWithTheOriginalString) {
  static constexpr auto* kLine = "Hello";
  static constexpr auto* kDelimiter = " ";
  EXPECT_EQ(Split(kLine, kDelimiter), (std::vector<std::string_view>{kLine}));
}

TEST(ObjLoaderTest, SplitStringWithDelimiterReturnsListWithSplitStringTokens) {
  static constexpr auto* kLine = " vt  0.707 0.395    0.684 ";
  static constexpr auto* kDelimiter = " ";
  EXPECT_EQ(Split(kLine, kDelimiter), (std::vector<std::string_view>{"vt", "0.707", "0.395", "0.684"}));
}

TEST(ObjLoaderTest, ParseEmptyStringThrowsAnException) { EXPECT_THROW(ParseToken<int>(""), std::invalid_argument); }

TEST(ObjLoaderTest, ParseInvalidTokenThrowsAnException) {
  EXPECT_THROW(ParseToken<float>("Definitely a float"), std::invalid_argument);
}

TEST(ObjLoaderTest, ParseIntTokenReturnsTheCorrectValue) { EXPECT_EQ(ParseToken<int>("42"), 42); }

TEST(ObjLoaderTest, ParseFloatTokenReturnsTheCorrectValue) { EXPECT_FLOAT_EQ(ParseToken<float>("3.14"), 3.14f); }

TEST(ObjLoaderTest, ParseEmptyLineThrowsAnException) { EXPECT_THROW((ParseLine<int, 3>("")), std::invalid_argument); }

TEST(ObjLoaderTest, ParseLineWithInvalidSizeArgumentThrowsAnException) {
  EXPECT_THROW((ParseLine<float, 2>("vt 0.707 0.395 0.684")), std::invalid_argument);
}

TEST(ObjLoaderTest, ParseLineReturnsVectorWithCorrectValues) {
  EXPECT_EQ((ParseLine<float, 3>("vt 0.707 0.395 0.684")), (glm::vec3{.707f, .395f, .684f}));
}

TEST(ObjLoaderTest, ParseIndexGroupWithOnlyPositionIndexReturnsCorrectIndexGroup) {
  EXPECT_EQ(ParseIndexGroup("1"), (glm::ivec3{0, kInvalidFaceIndex, kInvalidFaceIndex}));
}

TEST(ObjLoaderTest, ParseIndexGroupWithPositionAndTextureCoordinatesIndicesReturnsCorrectIndexGroup) {
  EXPECT_EQ(ParseIndexGroup("1/2"), (glm::ivec3{0, 1, kInvalidFaceIndex}));
}

TEST(ObjLoaderTest, ParseIndexGroupWithPositionAndNormalIndicesReturnsCorrectIndexGroup) {
  EXPECT_EQ(ParseIndexGroup("1//2"), (glm::ivec3{0, kInvalidFaceIndex, 1}));
}

TEST(ObjLoaderTest, ParseIndexGroupWithPositionTextureCoordinateAndNormalIndicesReturnsCorrectIndexGroup) {
  EXPECT_EQ(ParseIndexGroup("1/2/3"), (glm::ivec3{0, 1, 2}));
}

TEST(ObjLoaderTest, ParseInvalidIndexGroupThrowsAnException) {
  EXPECT_THROW(ParseIndexGroup(""), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("/"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("//"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("1/"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("/2"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("1//"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("/2/"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("//3"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("1/2/"), std::invalid_argument);
  EXPECT_THROW(ParseIndexGroup("/2/3"), std::invalid_argument);
}

TEST(ObjLoaderTest, ParseFaceWithInvalidNumberOfIndexGroupsThrowsAnException) {
  EXPECT_THROW(ParseFace("f 1/2/3 4/5/6"), std::invalid_argument);
  EXPECT_THROW(ParseFace("f 1/2/3 4/5/6 7/8/9 10/11/12"), std::invalid_argument);
}

TEST(ObjLoaderTest, ParseFaceReturnsCorrectIndexGroups) {
  EXPECT_EQ(ParseFace("f 1/2/3 4/5/6 7/8/9"),
            (std::array{glm::ivec3{0, 1, 2}, glm::ivec3{3, 4, 5}, glm::ivec3{6, 7, 8}}));
}

TEST(ObjLoaderTest, MeshLoadingGetsTheCorrectPositionsNormalsTextureCoordinates) {
  // clang-format off
  std::istringstream ss{R"(
    # positions
    v 0.0 0.1 0.2
    v 1.0 1.1 1.2
    v 2.0 2.1 2.2
    # texture coordinates
    vt 3.0 3.1
    vt 4.0 4.1
    vt 5.0 5.1
    # normals
    vn 6.0 6.1 6.2
    vn 7.0 7.1 7.2
    vn 8.0 8.1 8.2
  )"};
  // clang-format on

  const auto mesh = LoadMesh(ss);
  constexpr glm::vec3 kV0{0.f, 0.1f, 0.2f}, kV1{1.f, 1.1f, 1.2f}, kV2{2.f, 2.1f, 2.2f};
  constexpr glm::vec2 kVt0{3.f, 3.1f}, kVt1{4.f, 4.1f}, kVt2{5.f, 5.1f};
  constexpr glm::vec3 kVn0{6.f, 6.1f, 6.2f}, kVn1{7.f, 7.1f, 7.2f}, kVn2{8.f, 8.1f, 8.2f};

  EXPECT_EQ(mesh.positions(), (std::vector{kV0, kV1, kV2}));
  EXPECT_EQ(mesh.texture_coordinates(), (std::vector{kVt0, kVt1, kVt2}));
  EXPECT_EQ(mesh.normals(), (std::vector{kVn0, kVn1, kVn2}));
  EXPECT_TRUE(mesh.indices().empty());
}

TEST(ObjLoaderTest, IndexedMeshLoadingGetsTheCorrectPositionsNormalsTextureCoordinatesAndIndices) {
  // clang-format off
  std::istringstream ss{R"(
    # positions
    v 0.0 0.1 0.2
    v 1.0 1.1 1.2
    v 2.0 2.1 2.2
    v 3.0 3.1 3.2
    # texture coordinates
    vt 4.0 4.1
    vt 5.0 5.1
    vt 6.0 6.1
    vt 7.0 7.1
    # normals
    vn 8.0  8.1  8.2
    vn 9.0  9.1  9.2
    vn 10.0 10.1 10.2
    # faces
    f 1/4/2 2/1/3 3/2/1
    f 1/2/2 2/1/3 4/3/1
  )"};
  // clang-format on

  const auto mesh = LoadMesh(ss);
  constexpr glm::vec3 kV0{0.f, .1f, .2f}, kV1{1.f, 1.1f, 1.2f}, kV2{2.f, 2.1f, 2.2f}, kV3{3.f, 3.1f, 3.2f};
  constexpr glm::vec2 kVt0{4.f, 4.1f}, kVt1{5.f, 5.1f}, kVt2{6.f, 6.1f}, kVt3{7.f, 7.1f};
  constexpr glm::vec3 kVn0{8.f, 8.1f, 8.2f}, kVn1{9.f, 9.1f, 9.2f}, kVn2{10.f, 10.1f, 10.2f};

  EXPECT_EQ(mesh.positions(), (std::vector{kV0, kV1, kV2, kV0, kV3}));
  EXPECT_EQ(mesh.texture_coordinates(), (std::vector{kVt3, kVt0, kVt1, kVt1, kVt2}));
  EXPECT_EQ(mesh.normals(), (std::vector{kVn1, kVn2, kVn0, kVn1, kVn0}));
  EXPECT_EQ(mesh.indices(), (std::vector{0u, 1u, 2u, 3u, 1u, 4u}));
}
}  // namespace
