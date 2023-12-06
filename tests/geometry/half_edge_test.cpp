#include "geometry/half_edge.h"

#include <array>

#include <gtest/gtest.h>

namespace {

std::array<std::shared_ptr<qem::Vertex>, 3> CreateValidTriangle() {
  const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{-1.0f, -1.0f, 0.0f});
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{1.0f, -1.0f, 0.0f});
  const auto v2 = std::make_shared<qem::Vertex>(2, glm::vec3{0.0f, 0.5f, 0.0f});
  return std::array{v0, v1, v2};
}

TEST(HalfEdgeTest, InitializationSetsTheHalfEdgeVertex) {
  const auto vertex = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  const auto edge = std::make_shared<qem::HalfEdge>(vertex);
  EXPECT_EQ(edge->vertex(), vertex);
}

TEST(HalfEdgeTest, SetFlipUpdatesTheFlipHalfEdge) {
  const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  const auto edge10 = std::make_shared<qem::HalfEdge>(v0);
  edge01->set_flip(edge10);
  EXPECT_EQ(edge01->flip(), edge10);
}

TEST(HalfEdgeTest, SetNextUpdatesTheNextHalfEdge) {
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto v2 = std::make_shared<qem::Vertex>(2, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v2);
  const auto edge12 = std::make_shared<qem::HalfEdge>(v1);
  edge01->set_next(edge12);
  EXPECT_EQ(edge01->next(), edge12);
}

TEST(HalfEdgeTest, SetFaceUpdatesTheHalfEdgeFace) {
  const auto [v0, v1, v2] = CreateValidTriangle();
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  const auto face012 = std::make_shared<qem::Face>(v0, v1, v2);
  edge01->set_face(face012);
  EXPECT_EQ(edge01->face(), face012);
}

TEST(HalfEdgeTest, EqualHalfEdgesHaveTheSameHashValue) {
  const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  const auto edge10 = std::make_shared<qem::HalfEdge>(v0);
  edge01->set_flip(edge10);
  const auto edge01_copy = *edge01;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(*edge01, edge01_copy);
  EXPECT_EQ(hash_value(*edge01), hash_value(edge01_copy));
}

TEST(HalfEdgeTest, EqualHalfEdgeVerticesHaveTheSameHashValue) {
  const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  const auto edge10 = std::make_shared<qem::HalfEdge>(v0);
  edge01->set_flip(edge10);
  EXPECT_EQ(hash_value(*v0, *v1), hash_value(*edge01));
}

TEST(HalfEdgeTest, FlipHalfEdgesDoNotHaveTheSameHashValue) {
  const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  const auto edge10 = std::make_shared<qem::HalfEdge>(v0);
  edge01->set_flip(edge10);
  edge10->set_flip(edge01);
  EXPECT_NE(hash_value(*edge01), hash_value(*edge01->flip()));
}

#ifndef NDEBUG

TEST(HalfEdgeTest, GetExpiredVertexCausesProgramExit) {
  std::shared_ptr<qem::HalfEdge> edge10;
  {
    const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{});
    edge10 = std::make_shared<qem::HalfEdge>(v0);
  }
  EXPECT_DEATH({ std::ignore = edge10->vertex(); }, "");  // NOLINT(whitespace/newline) NOLINT(whitespace/comments)
}

TEST(HalfEdgeTest, GetExpiredFlipEdgeCausesProgramExit) {
  const auto v0 = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  {
    const auto edge10 = std::make_shared<qem::HalfEdge>(v0);
    edge10->set_flip(edge10);
  }
  EXPECT_DEATH({ std::ignore = edge01->flip(); }, "");  // NOLINT(whitespace/newline) NOLINT(whitespace/comments)
}

TEST(HalfEdgeTest, GetExpiredNextEdgeCausesProgramExit) {
  const auto v1 = std::make_shared<qem::Vertex>(1, glm::vec3{0.0f});
  const auto v2 = std::make_shared<qem::Vertex>(2, glm::vec3{0.0f});
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  {
    const auto edge12 = std::make_shared<qem::HalfEdge>(v2);
    edge01->set_next(edge12);
  }
  EXPECT_DEATH({ std::ignore = edge01->next(); }, "");  // NOLINT(whitespace/newline) NOLINT(whitespace/comments)
}

TEST(HalfEdgeTest, GetExpiredFaceCausesProgramExit) {
  const auto [v0, v1, v2] = CreateValidTriangle();
  const auto edge01 = std::make_shared<qem::HalfEdge>(v1);
  {
    const auto face012 = std::make_shared<qem::Face>(v0, v1, v2);
    edge01->set_face(face012);
  }
  EXPECT_DEATH({ std::ignore = edge01->face(); }, "");  // NOLINT(whitespace/newline) NOLINT(whitespace/comments)
}

#endif

}  // namespace
