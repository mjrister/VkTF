#include "graphics/obj_loader.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/gtx/hash.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "graphics/mesh.h"

namespace {

// sentinel value indicating an unspecified face index
constexpr auto kInvalidFaceIndex = -1;

/**
 * \brief Removes a set of characters from the beginning and end of the string.
 * \param line The string to evaluate.
 * \param delimiter A set of characters (in any order) to remove from the beginning and end of the string.
 * \return A view of the characters in \p delimiter removed from the beginning and end of \p line.
 */
constexpr std::string_view Trim(std::string_view line, const std::string_view delimiter = " \t") noexcept {
  line.remove_prefix(std::min<>(line.find_first_not_of(delimiter), line.size()));
  line.remove_suffix(line.size() - line.find_last_not_of(delimiter) - 1);
  return line;
}

/**
 * \brief Gets tokens delimited by a set of characters.
 * \param line The string to evaluate.
 * \param delimiter The set of characters (in any order) to split the string on.
 * \return A vector of tokens in \p line split on the characters in \p delimiter.
 */
std::vector<std::string_view> Split(const std::string_view line, const std::string_view delimiter = " \t") {
  std::vector<std::string_view> tokens;
  for (auto i = line.find_first_not_of(delimiter); i < line.size();) {
    const auto j = std::min<>(line.find_first_of(delimiter, i), line.size());
    tokens.push_back(line.substr(i, j - i));
    i = line.find_first_not_of(delimiter, j);
  }
  return tokens;
}

/**
 * \brief Parses a string token.
 * \tparam T The type to convert to.
 * \param token The token to parse.
 * \return The converted value of \p token to type \p T.
 */
template <typename T>
T ParseToken(const std::string_view token) {
  T value;
  if (const auto [_, error_code] = std::from_chars(token.data(), token.data() + token.size(), value);
      error_code == std::errc{}) {
    return value;
  }
  throw std::invalid_argument{std::format("Unable to convert {} to type {}", token, typeid(T).name())};
}

/**
 * \brief Parses a line in an .obj file.
 * \tparam T The type to convert to.
 * \tparam N The number of items to convert (does not include the first token identifying the line type).
 * \param line The line to parse.
 * \return A vector of size \p N containing each item in \p line converted to type \p T.
 */
template <typename T, glm::length_t N>
glm::vec<N, T> ParseLine(const std::string_view line) {
  if (const auto tokens = Split(line); tokens.size() == N + 1) {
    glm::vec<N, T> vec{};
    for (glm::length_t i = 0; i < N; ++i) {
      const auto j = static_cast<std::size_t>(i) + 1;
      vec[i] = ParseToken<T>(tokens[j]);
    }
    return vec;
  }
  throw std::invalid_argument{std::format("Unsupported format {}", line)};
}

/**
 * \brief Parses a token representing a face element index group.
 * \param token The token to parse. May optionally contain texture coordinate and normal indices.
 * \return A vector containing vertex position, texture coordinate, and normal indices. Unspecified texture
 *         coordinate and normal values are indicated by the value \c kInvalidFaceElementIndex.
 */
glm::ivec3 ParseIndexGroup(const std::string_view token) {
  static constexpr auto kDelimiter = "/";
  const auto delimiter_count = std::ranges::count(token, *kDelimiter);

  switch (const auto tokens = Split(token, kDelimiter); tokens.size()) {
    case 1: {
      // f v0 v1 v2
      if (delimiter_count == 0) {
        const auto v = ParseToken<int>(tokens[0]);
        return glm::ivec3{v - 1, kInvalidFaceIndex, kInvalidFaceIndex};
      }
      break;
    }
    case 2: {
      // f v0/vt0 v1/vt1 v2/vt2
      if (delimiter_count == 1) {
        const auto v = ParseToken<int>(tokens[0]);
        const auto vt = ParseToken<int>(tokens[1]);
        return glm::ivec3{v - 1, vt - 1, kInvalidFaceIndex};
      }
      // f v0//vn0 v1//vn1 v2//vn2
      if (delimiter_count == 2 && !token.starts_with(kDelimiter) && !token.ends_with(kDelimiter)) {
        const auto v = ParseToken<int>(tokens[0]);
        const auto vn = ParseToken<int>(tokens[1]);
        return glm::ivec3{v - 1, kInvalidFaceIndex, vn - 1};
      }
      break;
    }
    case 3: {
      // f v0/vt0/vn0 v1/vt1/vn1 v2/vt2/vn2
      if (delimiter_count == 2) {
        const auto v = ParseToken<int>(tokens[0]);
        const auto vt = ParseToken<int>(tokens[1]);
        const auto vn = ParseToken<int>(tokens[2]);
        return glm::ivec3{v - 1, vt - 1, vn - 1};
      }
      break;
    }
    default:
      break;
  }

  throw std::invalid_argument{std::format("Unsupported format {}", token)};
}

/**
 * \brief Parses a line representing a triangular face element.
 * \param line The line to parse.
 * \return An array containing three parsed index groups for the face.
 */
std::array<glm::ivec3, 3> ParseFace(const std::string_view line) {
  if (const auto tokens = Split(line, " \t"); tokens.size() == 4) {
    return std::array{ParseIndexGroup(tokens[1]), ParseIndexGroup(tokens[2]), ParseIndexGroup(tokens[3])};
  }
  throw std::invalid_argument{std::format("Unsupported format {}", line)};
}

/**
 * \brief Loads a triangle mesh from an input stream representing the contents of an .obj file.
 * \param is The input stream to parse.
 * \return A mesh defined by the position, texture coordinates, normals, and indices specified in the input stream.
 */
qem::Mesh LoadMesh(std::istream& is) {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec2> texture_coordinates;
  std::vector<glm::vec3> normals;
  std::vector<std::array<glm::ivec3, 3>> faces;

  for (std::string line; getline(is, line);) {
    if (const auto line_view = Trim(line); !line_view.empty() && !line_view.starts_with('#')) {
      if (line_view.starts_with("v ")) {
        positions.push_back(ParseLine<float, 3>(line_view));
      } else if (line_view.starts_with("vt ")) {
        texture_coordinates.push_back(ParseLine<float, 2>(line_view));
      } else if (line_view.starts_with("vn ")) {
        normals.push_back(ParseLine<float, 3>(line_view));
      } else if (line_view.starts_with("f ")) {
        faces.push_back(ParseFace(line_view));
      }
    }
  }

  if (faces.empty()) return qem::Mesh{positions, texture_coordinates, normals, {}};

  std::vector<glm::vec3> ordered_positions;
  std::vector<glm::vec2> ordered_texture_coordinates;
  std::vector<glm::vec3> ordered_normals;
  std::vector<GLuint> indices;
  indices.reserve(faces.size() * 3);

  // For each index group, store texture coordinate and normals at the same index as the vertex position so that data is
  // aligned when sent to the vertex shader. Occasionally, index groups may specify different texture coordinates or
  // normals for the same vertex position. To handle this situation, an unordered map is used to keep track of unique
  // index groups and appends new position, texture coordinate, normal triples to the end of each respective ordered
  // array as necessary.
  for (std::unordered_map<glm::ivec3, GLuint> index_groups; const auto& face : faces) {
    for (const auto& index_group : face) {
      auto iterator = index_groups.find(index_group);
      if (iterator == index_groups.end()) {
        if (const auto position_index = index_group[0]; position_index != kInvalidFaceIndex) {
          ordered_positions.push_back(positions.at(position_index));
        }
        if (const auto texture_coordinate_index = index_group[1]; texture_coordinate_index != kInvalidFaceIndex) {
          ordered_texture_coordinates.push_back(texture_coordinates.at(texture_coordinate_index));
        }
        if (const auto normal_index = index_group[2]; normal_index != kInvalidFaceIndex) {
          ordered_normals.push_back(normals.at(normal_index));
        }
        const auto index = static_cast<GLuint>(ordered_positions.size()) - 1;
        iterator = index_groups.emplace(index_group, index).first;
      }
      indices.push_back(iterator->second);
    }
  }

  return qem::Mesh{ordered_positions, ordered_texture_coordinates, ordered_normals, indices};
}

}  // namespace

qem::Mesh qem::obj_loader::LoadMesh(const std::filesystem::path& filepath) {
  if (std::ifstream ifs{filepath}) {
    return ::LoadMesh(ifs);
  }
  throw std::runtime_error{std::format("Unable to open {}", filepath.generic_string())};
}
