#include "geometry/half_edge_mesh.cpp"  // NOLINT(build/include)

#include <vector>

#include <GL/gl3w.h>
#include <gtest/gtest.h>

namespace {

qem::Mesh CreateValidMesh() {
  const std::vector<glm::vec3> positions{
      {1.0f, 0.0f, 0.0f},   // v0
      {2.0f, 0.0f, 0.0f},   // v1
      {0.5f, -1.0f, 0.0f},  // v2
      {1.5f, -1.0f, 0.0f},  // v3
      {2.5f, -1.0f, 0.0f},  // v4
      {3.0f, 0.0f, 0.0f},   // v5
      {2.5f, 1.0f, 0.0f},   // v6
      {1.5f, 1.0f, 0.0f},   // v7
      {0.5f, 1.0f, 0.0f},   // v8
      {0.0f, 0.0f, 0.0f}    // v9
  };

  const std::vector<GLuint> indices{
      0, 2, 3,  // f0
      0, 3, 1,  // f1
      0, 1, 7,  // f2
      0, 7, 8,  // f3
      0, 8, 9,  // f4
      0, 9, 2,  // f5
      1, 3, 4,  // f6
      1, 4, 5,  // f7
      1, 5, 6,  // f8
      1, 6, 7   // f9
  };

  return qem::Mesh{positions, {}, std::vector(positions.size(), glm::vec3{0.0f, 0.0f, 1.0f}), indices};
}

qem::HalfEdgeMesh MakeHalfEdgeMesh() {
  const auto mesh = CreateValidMesh();
  return qem::HalfEdgeMesh{mesh};
}

void VerifyEdge(const std::shared_ptr<qem::Vertex>& v0,
                const std::shared_ptr<qem::Vertex>& v1,
                const std::unordered_map<std::size_t, std::shared_ptr<qem::HalfEdge>>& edges) {
  const auto edge01_iterator = edges.find(hash_value(*v0, *v1));
  const auto edge10_iterator = edges.find(hash_value(*v1, *v0));

  ASSERT_NE(edge01_iterator, edges.end());
  ASSERT_NE(edge10_iterator, edges.end());

  const auto& edge01 = edge01_iterator->second;
  const auto& edge10 = edge10_iterator->second;

  EXPECT_EQ(v0, edge10->vertex());
  EXPECT_EQ(v1, edge01->vertex());

  EXPECT_EQ(edge01, edge10->flip());
  EXPECT_EQ(edge10, edge01->flip());

  EXPECT_EQ(edge01, edge01->flip()->flip());
  EXPECT_EQ(edge10, edge10->flip()->flip());
}

void VerifyTriangles(const qem::HalfEdgeMesh& half_edge_mesh, const std::vector<GLuint>& indices) {
  const auto& vertices = half_edge_mesh.vertices();
  const auto& edges = half_edge_mesh.edges();
  const auto& faces = half_edge_mesh.faces();

  for (auto i = 0; std::cmp_less(i, indices.size()); i += 3) {
    const auto v0_iterator = vertices.find(static_cast<int>(indices[i]));
    const auto v1_iterator = vertices.find(static_cast<int>(indices[i + 1]));
    const auto v2_iterator = vertices.find(static_cast<int>(indices[i + 2]));

    ASSERT_NE(v0_iterator, vertices.end());
    ASSERT_NE(v1_iterator, vertices.end());
    ASSERT_NE(v2_iterator, vertices.end());

    const auto& v0 = v0_iterator->second;
    const auto& v1 = v1_iterator->second;
    const auto& v2 = v2_iterator->second;

    VerifyEdge(v0, v1, edges);
    VerifyEdge(v1, v2, edges);
    VerifyEdge(v2, v0, edges);

    const auto edge01 = edges.at(hash_value(*v0, *v1));
    const auto edge12 = edges.at(hash_value(*v1, *v2));
    const auto edge20 = edges.at(hash_value(*v2, *v0));

    EXPECT_EQ(edge01->next(), edge12);
    EXPECT_EQ(edge12->next(), edge20);
    EXPECT_EQ(edge20->next(), edge01);

    const auto face012_iterator = faces.find(hash_value(*v0, *v1, *v2));
    ASSERT_NE(face012_iterator, faces.end());

    const auto face012 = face012_iterator->second;
    EXPECT_EQ(edge01->face(), face012);
    EXPECT_EQ(edge12->face(), face012);
    EXPECT_EQ(edge20->face(), face012);
  }
}

TEST(HalfEdgeMeshTest, CreateHalfEdgeMeshHasCorrectVerticesEdgesFacesAndIndices) {
  const auto mesh = CreateValidMesh();
  const qem::HalfEdgeMesh half_edge_mesh{mesh};

  EXPECT_EQ(10, half_edge_mesh.vertices().size());
  EXPECT_EQ(38, half_edge_mesh.edges().size());
  EXPECT_EQ(10, half_edge_mesh.faces().size());

  VerifyTriangles(half_edge_mesh, mesh.indices());
}

TEST(HalfEdgeMeshTest, CollapseEdgeUpdatesIndicentEdgesToReferToNewVertex) {
  auto half_edge_mesh = MakeHalfEdgeMesh();
  const auto& vertices = half_edge_mesh.vertices();
  const auto& edges = half_edge_mesh.edges();
  const auto& v0 = vertices.at(0);
  const auto& v1 = vertices.at(1);
  const auto& edge01 = edges.at(hash_value(*v0, *v1));
  const qem::Vertex v_new{static_cast<int>(half_edge_mesh.vertices().size()), (v0->position() + v1->position()) / 2.0f};

  half_edge_mesh.Contract(*edge01, std::make_shared<qem::Vertex>(v_new));

  EXPECT_EQ(9, half_edge_mesh.vertices().size());
  EXPECT_EQ(32, half_edge_mesh.edges().size());
  EXPECT_EQ(8, half_edge_mesh.faces().size());

  VerifyTriangles(half_edge_mesh, {2, 3,  10,   // f0
                                   3, 4,  10,   // f1
                                   4, 5,  10,   // f2
                                   5, 6,  10,   // f3
                                   6, 7,  10,   // f4
                                   7, 8,  10,   // f5
                                   8, 9,  10,   // f6
                                   2, 10, 9});  // f7
}

#ifndef NDEBUG

TEST(HalfEdgeMeshTest, CollapseHalfEdgeWithExistingVertexCausesProgramExit) {
  auto half_edge_mesh = MakeHalfEdgeMesh();
  const auto& v0 = half_edge_mesh.vertices().at(0);
  const auto& v1 = half_edge_mesh.vertices().at(1);
  const auto& edge01 = half_edge_mesh.edges().at(hash_value(*v0, *v1));
  EXPECT_DEATH(half_edge_mesh.Contract(*edge01, v0), "");
}

TEST(HalfEdgeMeshTest, CollapseMissingHalfEdgeCausesProgramExit) {
  auto half_edge_mesh = MakeHalfEdgeMesh();
  const auto next_vertex_id = static_cast<int>(half_edge_mesh.vertices().size());
  const auto v_invalid0 = std::make_shared<qem::Vertex>(next_vertex_id, glm::vec3{0.0f});
  const auto v_invalid1 = std::make_shared<qem::Vertex>(next_vertex_id + 1, glm::vec3{0.0f});
  const auto edge_invalid01 = std::make_shared<qem::HalfEdge>(v_invalid1);
  const auto edge_invalid10 = std::make_shared<qem::HalfEdge>(v_invalid0);
  edge_invalid01->set_flip(edge_invalid10);
  EXPECT_DEATH(half_edge_mesh.Contract(*edge_invalid01, std::make_shared<qem::Vertex>(42, glm::vec3{0.0f})), "");
}

#endif

}  // namespace
