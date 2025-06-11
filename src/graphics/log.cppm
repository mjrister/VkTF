module;

#include <cassert>
#include <filesystem>
#include <format>
#include <ios>
#include <iostream>
#include <mutex>
#include <ostream>
#include <print>
#include <source_location>
#include <string>
#include <utility>

export module log;

namespace vktf {

export class [[nodiscard]] Log {
  class LineProxy;

public:
  enum class Severity : uint8_t { kInfo, kWarning, kError };

  [[nodiscard]] static Log& Default() {
    static Log default_log{std::clog, std::cerr, std::cerr};
    return default_log;
  }

  Log(std::ostream& info_ostream, std::ostream& warning_ostream, std::ostream& error_ostream);

  Log(const Log&) = delete;
  Log(Log&&) noexcept = delete;

  Log& operator=(const Log&) = delete;
  Log& operator=(Log&&) noexcept = delete;

  ~Log() noexcept;

  [[nodiscard]] LineProxy operator()(Severity severity,
                                     const std::source_location& source_location = std::source_location::current());

private:
  class [[nodiscard]] LineProxy {
  public:
    LineProxy(std::mutex& ostream_mutex, std::ostream& ostream, const std::source_location& source_location);

    LineProxy(const LineProxy&) = delete;
    LineProxy(LineProxy&&) noexcept = delete;

    LineProxy& operator=(const LineProxy&) = delete;
    LineProxy& operator=(LineProxy&&) noexcept = delete;

    ~LineProxy() noexcept;

    template <typename T>
    LineProxy& operator<<(T&& value) {
      ostream_ << std::forward<T>(value);
      return *this;
    }

    template <typename... Args>
    void Print(const std::format_string<Args...> format_string, Args&&... args) {
      std::print(ostream_, format_string, std::forward<Args>(args)...);
    }

  private:
    std::scoped_lock<std::mutex> ostream_lock_;
    std::ostream& ostream_;
  };

  std::mutex ostream_mutex_;
  std::ostream& info_ostream_;
  std::ostream& warning_ostream_;
  std::ostream& error_ostream_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

std::ostream& GetLogStream(const Log::Severity severity,
                           std::ostream& info_ostream,
                           std::ostream& warning_ostream,
                           std::ostream& error_ostream) {
  switch (severity) {
    using enum Log::Severity;
    case kInfo:
      return info_ostream;
    case kWarning:
      return warning_ostream;
    case kError:
      return error_ostream;
    default:
      std::unreachable();
  }
}

std::string GetPreamble(const std::source_location& source_location) {
  return std::format("[{}:{}] ",
                     std::filesystem::path{source_location.file_name()}.filename().string(),
                     source_location.line());
}

}  // namespace

Log::Log(std::ostream& info_ostream, std::ostream& warning_ostream, std::ostream& error_ostream)
    : info_ostream_{info_ostream}, warning_ostream_{warning_ostream}, error_ostream_{error_ostream} {}

Log::~Log() noexcept {
  try {
    if (info_ostream_) info_ostream_.flush();
    if (warning_ostream_) warning_ostream_.flush();
    if (error_ostream_) error_ostream_.flush();
  } catch (const std::ios_base::failure&) {
    assert(false);  // prevent exception propagation from noexcept destructor
  }
}

Log::LineProxy Log::operator()(const Severity severity, const std::source_location& source_location) {
  auto& ostream = GetLogStream(severity, info_ostream_, warning_ostream_, error_ostream_);
  return LineProxy{ostream_mutex_, ostream, source_location};
}

Log::LineProxy::LineProxy(std::mutex& ostream_mutex, std::ostream& ostream, const std::source_location& source_location)
    : ostream_lock_{ostream_mutex}, ostream_{ostream} {
  ostream << GetPreamble(source_location);
}

Log::LineProxy::~LineProxy() noexcept {
  try {
    if (ostream_) ostream_ << '\n';
  } catch (const std::ios_base::failure&) {
    assert(false);  // prevent exception propagation from noexcept destructor
  }
}

}  // namespace vktf
