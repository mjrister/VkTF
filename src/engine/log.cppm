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

/**
 * @brief A thread-safe utility for logging messages with varying severity levels based on standard output streams.
 * @code
 * using Log::Severity;
 * const auto& log = Log::Default();
 * log(Severity::kInfo) << "Hello, world";
 * log(Severity::kInfo).Print("The answer to life, the universe, and everything is {}", 42);
 * @endcode
 */
export class [[nodiscard]] Log {
  class LineProxy;

public:
  /** @brief An enumeration representing the severity of a log message. */
  enum class Severity : uint8_t { kInfo, kWarning, kError };

  /**
   * @brief Gets the default log implementation.
   * @details The default log implementation assigns messages with severity @ref Severity::kInfo to @c std::clog and
   *          messages with severity @ref Severity::kWarning or @ref Severity::kError to @c std::cerr.
   * @return A reference to the default log instance.
   */
  [[nodiscard]] static Log& Default() {
    static Log default_log{std::clog, std::cerr, std::cerr};
    return default_log;
  }

  /**
   * @brief Creates a @ref Log.
   * @param info_ostream The output stream for messages with severity @ref Severity::kInfo.
   * @param warning_ostream The output stream for messages with severity @ref Severity::kWarning.
   * @param error_ostream The output stream for messages with severity @ref Severity::kError.
   */
  Log(std::ostream& info_ostream, std::ostream& warning_ostream, std::ostream& error_ostream);

  Log(const Log&) = delete;
  Log(Log&&) noexcept = delete;

  Log& operator=(const Log&) = delete;
  Log& operator=(Log&&) noexcept = delete;

  /**
   * @brief Destroys a @ref Log.
   * @details Flushes output streams to ensure log messages are correctly written on destruction.
   */
  ~Log() noexcept;

  /**
   * @brief Begins a new single-line log message.
   * @param severity The log message severity.
   * @param source_location The source code location indicating where the log message originates from.
   * @return A thread-safe proxy for writing single-line log messages with the provided @p severity.
   */
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
