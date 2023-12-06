#ifndef SRC_GEOMETRY_VERTEX_H_
#define SRC_GEOMETRY_VERTEX_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

#include <glm/vec3.hpp>

namespace qem {
class HalfEdge;

/** \brief A half-edge mesh vertex. */
class Vertex {
public:
  /**
   * \brief Initializes a vertex.
   * \param position The vertex position.
   */
  explicit Vertex(const glm::vec3& position) noexcept : position_{position} {}

  /**
   * \brief Initializes a vertex.
   * \param id The vertex ID.
   * \param position The vertex position.
   */
  Vertex(const int id, const glm::vec3& position) noexcept : id_{id}, position_{position} {}

  /** \brief Gets the vertex ID. */
  [[nodiscard]] int id() const noexcept {
    assert(id_.has_value());
    return *id_;
  }

  /** \brief Sets the vertex ID. */
  void set_id(const int id) noexcept { id_ = id; }

  /** \brief Gets the vertex position. */
  [[nodiscard]] const glm::vec3& position() const noexcept { return position_; }

  /** \brief Gets the last created half-edge that points to this vertex. */
  [[nodiscard]] std::shared_ptr<const HalfEdge> edge() const noexcept {
    assert(!edge_.expired());
    return edge_.lock();
  }

  /** \brief Sets the vertex half-edge. */
  void set_edge(const std::shared_ptr<const HalfEdge>& edge) noexcept { edge_ = edge; }

  /** \brief Defines the vertex equality operator. */
  friend bool operator==(const Vertex& lhs, const Vertex& rhs) noexcept { return lhs.id() == rhs.id(); }

  /** \brief Gets the hash value for a vertex. */
  friend std::size_t hash_value(const Vertex& v0) noexcept { return v0.id(); }

  /** \brief Gets the hash value for a vertex pair. */
  friend std::size_t hash_value(const Vertex& v0, const Vertex& v1) noexcept {
    std::size_t seed = 0;
    hash_combine(seed, v0, v1);
    return seed;
  }

  /** \brief Gets the hash value for vertex triple. */
  friend std::size_t hash_value(const Vertex& v0, const Vertex& v1, const Vertex& v2) noexcept {
    std::size_t seed = 0;
    hash_combine(seed, v0, v1, v2);
    return seed;
  }

private:
  /**
   * \brief Combines the hash values of multiple vertices.
   * \param seed The starting seed to generate hash values from.
   * \param vertex The current vertex to process.
   * \param rest The remaining vertices to combine hash values for.
   * \note This hash algorithm is based on boost::hash_combine.
   * \see https://www.boost.org/doc/libs/1_83_0/libs/container_hash/doc/html/hash.html#notes_hash_combine
   */
  // NOLINTNEXTLINE(runtime/references)
  static void hash_combine(std::size_t& seed, const Vertex& vertex, const auto&... rest) {
    seed += 0x9e3779b9 + hash_value(vertex);
    if constexpr (sizeof(std::size_t) == sizeof(std::uint32_t)) {
      seed ^= seed >> 16u;
      seed *= 0x21f0aaad;
      seed ^= seed >> 15u;
      seed *= 0x735a2d97;
      seed ^= seed >> 15u;
    } else if constexpr (sizeof(std::size_t) == sizeof(std::uint64_t)) {
      seed ^= seed >> 32u;
      seed *= 0xe9846af9b1a615d;
      seed ^= seed >> 32u;
      seed *= 0xe9846af9b1a615d;
      seed ^= seed >> 28u;
    } else {
      // the following is a workaround to enable using static_assert(false) in if constexpr expressions
      ([]<bool False = false>() { static_assert(False, "Unsupported processor architecture"); })();
    }
    (hash_combine(seed, rest), ...);
  }

  std::optional<int> id_;
  glm::vec3 position_;
  std::weak_ptr<const HalfEdge> edge_;
};

}  // namespace qem

#endif  // SRC_GEOMETRY_VERTEX_H_
