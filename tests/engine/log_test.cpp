#include <filesystem>
#include <source_location>
#include <sstream>
#include <string_view>

#include <gtest/gtest.h>

import log;

namespace {

class LogTest : public ::testing::Test {
protected:
  using Severity = vktf::Log::Severity;

  std::string GetLogFormat(const std::string_view message) const {
    return std::format("[{}:{}] {}\n",
                       std::filesystem::path{source_location_.file_name()}.filename().string(),
                       source_location_.line(),
                       message);
  }

  std::source_location source_location_ = std::source_location::current();
  std::ostringstream info_ostream_;
  std::ostringstream warning_ostream_;
  std::ostringstream error_ostream_;
  vktf::Log log_{info_ostream_, warning_ostream_, error_ostream_};
};

TEST_F(LogTest, StartsWithEmptyOutputStreams) {
  EXPECT_EQ(0, info_ostream_.tellp());
  EXPECT_EQ(0, warning_ostream_.tellp());
  EXPECT_EQ(0, error_ostream_.tellp());
}

TEST_F(LogTest, InsertionOperatorWritesToCorrectOutputStreamWithInfoSeverity) {
  static constexpr std::string_view kInfoMessage = "INFO";
  log_(Severity::kInfo, source_location_) << kInfoMessage;

  const auto expected_message = GetLogFormat(kInfoMessage);
  const auto actual_message = info_ostream_.str();

  EXPECT_EQ(expected_message, actual_message);
  EXPECT_EQ(0, warning_ostream_.tellp());
  EXPECT_EQ(0, error_ostream_.tellp());
}

TEST_F(LogTest, InsertionOperatorWritesToCorrectOutputStreamWithWarningSeverity) {
  static constexpr std::string_view kWarningMessage = "WARNING";
  log_(Severity::kWarning, source_location_) << kWarningMessage;

  const auto expected_message = GetLogFormat(kWarningMessage);
  const auto actual_message = warning_ostream_.str();

  EXPECT_EQ(0, info_ostream_.tellp());
  EXPECT_EQ(expected_message, actual_message);
  EXPECT_EQ(0, error_ostream_.tellp());
}

TEST_F(LogTest, InsertionOperatorWritesToCorrectOutputStreamWithErrorSeverity) {
  static constexpr std::string_view kErrorMessage = "ERROR";
  log_(Severity::kError, source_location_) << kErrorMessage;

  const auto expected_message = GetLogFormat(kErrorMessage);
  const auto actual_message = error_ostream_.str();

  EXPECT_EQ(0, info_ostream_.tellp());
  EXPECT_EQ(0, warning_ostream_.tellp());
  EXPECT_EQ(expected_message, actual_message);
}

TEST_F(LogTest, InsertionOperatorChainingWritesToOneLine) {
  static constexpr std::string_view kMessagePartA = "A";
  static constexpr std::string_view kMessagePartB = "B";
  static constexpr std::string_view kMessagePartC = "C";
  log_(Severity::kInfo, source_location_) << kMessagePartA << kMessagePartB << kMessagePartC;

  const auto expected_message = GetLogFormat(std::format("{}{}{}", kMessagePartA, kMessagePartB, kMessagePartC));
  const auto actual_message = info_ostream_.str();

  EXPECT_EQ(expected_message, actual_message);
}

TEST_F(LogTest, PrintsToCorrectOutputStreamWithInfoSeverity) {
  static constexpr std::string_view kInfoMessage = "INFO";
  log_(Severity::kInfo, source_location_).Print("{}", kInfoMessage);

  const auto expected_message = GetLogFormat(kInfoMessage);
  const auto actual_message = info_ostream_.str();

  EXPECT_EQ(expected_message, actual_message);
  EXPECT_EQ(0, warning_ostream_.tellp());
  EXPECT_EQ(0, error_ostream_.tellp());
}

TEST_F(LogTest, PrintsToCorrectOutputStreamWithWarningSeverity) {
  static constexpr std::string_view kWarningMessage = "WARNING";
  log_(Severity::kWarning, source_location_).Print("{}", kWarningMessage);

  const auto expected_message = GetLogFormat(kWarningMessage);
  const auto actual_message = warning_ostream_.str();

  EXPECT_EQ(0, info_ostream_.tellp());
  EXPECT_EQ(expected_message, actual_message);
  EXPECT_EQ(0, error_ostream_.tellp());
}

TEST_F(LogTest, PrintsToCorrectOutputStreamWithErrorSeverity) {
  static constexpr std::string_view kErrorMessage = "ERROR";
  log_(Severity::kError, source_location_).Print("{}", kErrorMessage);

  const auto expected_message = GetLogFormat(kErrorMessage);
  const auto actual_message = error_ostream_.str();

  EXPECT_EQ(0, info_ostream_.tellp());
  EXPECT_EQ(0, warning_ostream_.tellp());
  EXPECT_EQ(expected_message, actual_message);
}

}  // namespace
