// Reader for SOURCE_DEFAULTS, KLINE_SOURCE_SET, and CHANNEL_SET.
// One KLINE_SOURCE_SET expands to one source file per interval:
//   <root>/<instrument>/<interval>/<instrument>-<interval>-all-years.csv
// One CHANNEL_SET expands to one channel row per interval.
// SPDX-License-Identifier: MIT
#pragma once

#include <cctype>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "source_config.h"

namespace mityane::data {

namespace detail {

struct settings_field {
  std::string key;
  std::string value;
};

struct settings_block {
  std::string name;
  std::vector<settings_field> fields;
};

inline std::string settings_trim(std::string_view text) {
  std::size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin])) != 0)
    ++begin;
  std::size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1])) != 0)
    --end;
  return std::string(text.substr(begin, end - begin));
}

class settings_cursor {
public:
  explicit settings_cursor(std::string_view text) : text_(text) {}

  [[nodiscard]] bool done() {
    skip();
    return index_ >= text_.size();
  }

  [[nodiscard]] bool next_is(char token) {
    skip();
    return index_ < text_.size() && text_[index_] == token;
  }

  std::string identifier() {
    skip();
    if (index_ >= text_.size() || !is_start(text_[index_]))
      throw std::runtime_error("[settings] expected a name");
    const std::size_t begin = index_++;
    while (index_ < text_.size() && is_continue(text_[index_]))
      ++index_;
    return std::string(text_.substr(begin, index_ - begin));
  }

  void expect(char token) {
    skip();
    if (index_ >= text_.size() || text_[index_] != token)
      throw std::runtime_error(std::string("[settings] expected '") + token + "'");
    ++index_;
  }

  std::string value_until_semicolon() {
    skip();
    std::string value;
    while (index_ < text_.size() && text_[index_] != ';')
      value.push_back(text_[index_++]);
    if (index_ >= text_.size())
      throw std::runtime_error("[settings] assignment is missing ';'");
    ++index_;
    return settings_trim(value);
  }

private:
  std::string_view text_;
  std::size_t index_{0};

  static bool is_start(char token) {
    return std::isalpha(static_cast<unsigned char>(token)) != 0 || token == '_';
  }
  static bool is_continue(char token) {
    return std::isalnum(static_cast<unsigned char>(token)) != 0 || token == '_';
  }

  void skip() {
    for (;;) {
      while (index_ < text_.size() &&
             std::isspace(static_cast<unsigned char>(text_[index_])) != 0)
        ++index_;
      if (index_ + 1 < text_.size() && text_[index_] == '/' &&
          text_[index_ + 1] == '/') {
        index_ += 2;
        while (index_ < text_.size() && text_[index_] != '\n')
          ++index_;
        continue;
      }
      if (index_ + 1 < text_.size() && text_[index_] == '/' &&
          text_[index_ + 1] == '*') {
        index_ += 2;
        while (index_ + 1 < text_.size() &&
               !(text_[index_] == '*' && text_[index_ + 1] == '/'))
          ++index_;
        if (index_ + 1 >= text_.size())
          throw std::runtime_error("[settings] unterminated comment");
        index_ += 2;
        continue;
      }
      return;
    }
  }
};

inline std::vector<settings_block> scan_settings_blocks(std::string_view text) {
  settings_cursor cursor(text);
  std::vector<settings_block> blocks;
  while (!cursor.done()) {
    settings_block block;
    block.name = cursor.identifier();
    cursor.expect('{');
    while (!cursor.next_is('}')) {
      settings_field field;
      field.key = cursor.identifier();
      cursor.expect('=');
      field.value = cursor.value_until_semicolon();
      for (const auto &earlier : block.fields) {
        if (earlier.key == field.key)
          throw std::runtime_error("[settings] repeated " + block.name + "." + field.key);
      }
      block.fields.push_back(std::move(field));
    }
    cursor.expect('}');
    blocks.push_back(std::move(block));
  }
  return blocks;
}

inline const std::string *find_field(const settings_block &block, const char *key) {
  for (const auto &field : block.fields) {
    if (field.key == key)
      return &field.value;
  }
  return nullptr;
}

inline std::string required_field(const settings_block &block, const char *key) {
  const auto *value = find_field(block, key);
  if (value == nullptr || value->empty())
    throw std::runtime_error(std::string("[settings] ") + block.name + "." + key +
                             " is required");
  return *value;
}

inline void reject_unknown_fields(const settings_block &block,
                                  std::initializer_list<const char *> known) {
  for (const auto &field : block.fields) {
    bool found = false;
    for (const char *name : known) {
      if (field.key == name) {
        found = true;
        break;
      }
    }
    if (!found)
      throw std::runtime_error("[settings] unknown " + block.name + "." + field.key);
  }
}

inline std::vector<std::string> split_intervals(const std::string &text, const char *where) {
  std::vector<std::string> intervals;
  std::string current;
  for (char token : text) {
    if (token == ',') {
      intervals.push_back(settings_trim(current));
      current.clear();
    } else {
      current.push_back(token);
    }
  }
  intervals.push_back(settings_trim(current));
  if (intervals.empty())
    throw std::runtime_error(std::string("[settings] ") + where + " must not be empty");
  for (const auto &interval : intervals) {
    if (interval.empty())
      throw std::runtime_error(std::string("[settings] ") + where + " has an empty entry");
  }
  return intervals;
}

inline std::string kline_source_path(std::string root, const std::string &instrument,
                                     const std::string &interval) {
  while (root.size() > 1 && root.back() == '/')
    root.pop_back();
  return root + "/" + instrument + "/" + interval + "/" + instrument + "-" + interval +
         "-all-years.csv";
}

inline bool known_source_kind(const std::string &kind) {
  return kind == "real" || kind == "synthetic" || kind == "derived";
}

} // namespace detail

// Fills source_forms and channel_forms. SOURCE_DEFAULTS may appear after the
// sets it configures. A checked-in table is not part of this function.
[[nodiscard]] inline source_materialization_request_t
read_loader_settings(std::string_view text) {
  const auto blocks = detail::scan_settings_blocks(text);
  const detail::settings_block *defaults = nullptr;
  for (const auto &block : blocks) {
    if (block.name == "SOURCE_DEFAULTS") {
      if (defaults != nullptr)
        throw std::runtime_error("[settings] SOURCE_DEFAULTS must be declared at most once");
      defaults = &block;
    } else if (block.name != "KLINE_SOURCE_SET" && block.name != "CHANNEL_SET") {
      throw std::runtime_error("[settings] unsupported block " + block.name);
    }
  }

  std::string default_root;
  std::vector<std::string> default_intervals;
  if (defaults != nullptr) {
    detail::reject_unknown_fields(*defaults, {"SOURCE_ROOT", "KLINE_INTERVALS"});
    if (const auto *root = detail::find_field(*defaults, "SOURCE_ROOT"))
      default_root = *root;
    if (const auto *intervals = detail::find_field(*defaults, "KLINE_INTERVALS"))
      default_intervals =
          detail::split_intervals(*intervals, "SOURCE_DEFAULTS.KLINE_INTERVALS");
  }

  source_materialization_request_t request;
  for (const auto &block : blocks) {
    if (block.name == "KLINE_SOURCE_SET") {
      detail::reject_unknown_fields(
          block, {"INSTRUMENT", "MARKET_TYPE", "VENUE", "BASE_ASSET", "QUOTE_ASSET",
                  "SOURCE_KIND", "SOURCE_ROOT", "INTERVALS"});
      source_form_t base;
      base.instrument = detail::required_field(block, "INSTRUMENT");
      base.record_type = "kline";
      base.market_type = detail::required_field(block, "MARKET_TYPE");
      base.venue = detail::required_field(block, "VENUE");
      base.base_asset = detail::required_field(block, "BASE_ASSET");
      base.quote_asset = detail::required_field(block, "QUOTE_ASSET");
      base.source_kind = detail::required_field(block, "SOURCE_KIND");
      if (!detail::known_source_kind(base.source_kind))
        throw std::runtime_error("[settings] KLINE_SOURCE_SET.SOURCE_KIND must be "
                                 "real, synthetic, or derived");
      std::string error;
      if (!instrument_signature_validate(base.instrument_signature(), false,
                                         "KLINE_SOURCE_SET " + base.instrument, &error))
        throw std::runtime_error("[settings] " + error);
      std::string root = default_root;
      if (const auto *own = detail::find_field(block, "SOURCE_ROOT"))
        root = *own;
      if (root.empty())
        throw std::runtime_error("[settings] KLINE_SOURCE_SET.SOURCE_ROOT is required when "
                                 "SOURCE_DEFAULTS.SOURCE_ROOT is absent");
      auto intervals = default_intervals;
      if (const auto *own = detail::find_field(block, "INTERVALS"))
        intervals = detail::split_intervals(*own, "KLINE_SOURCE_SET.INTERVALS");
      if (intervals.empty())
        throw std::runtime_error("[settings] KLINE_SOURCE_SET.INTERVALS is required when "
                                 "SOURCE_DEFAULTS.KLINE_INTERVALS is absent");
      for (const auto &interval : intervals) {
        source_form_t row = base;
        try {
          row.interval = EnumTraits<interval_type_e>::fromString(interval);
        } catch (const std::invalid_argument &ex) {
          throw std::runtime_error(std::string("[settings] KLINE_SOURCE_SET.INTERVALS: ") +
                                   ex.what());
        }
        row.source = detail::kline_source_path(root, row.instrument, interval);
        request.source_forms.push_back(std::move(row));
      }
    } else if (block.name == "CHANNEL_SET") {
      detail::reject_unknown_fields(
          block, {"ACTIVE", "RECORD_TYPE", "INPUT_LENGTH", "FUTURE_LENGTH",
                  "CHANNEL_WEIGHT", "NORMALIZATION_POLICY", "INTERVALS"});
      channel_form_t base;
      base.active = detail::required_field(block, "ACTIVE");
      base.record_type = detail::required_field(block, "RECORD_TYPE");
      base.input_length = detail::required_field(block, "INPUT_LENGTH");
      base.future_length = detail::required_field(block, "FUTURE_LENGTH");
      base.channel_weight = detail::required_field(block, "CHANNEL_WEIGHT");
      base.normalization_policy = detail::required_field(block, "NORMALIZATION_POLICY");
      const auto intervals =
          detail::split_intervals(detail::required_field(block, "INTERVALS"),
                                  "CHANNEL_SET.INTERVALS");
      for (const auto &interval : intervals) {
        channel_form_t row = base;
        try {
          row.interval = EnumTraits<interval_type_e>::fromString(interval);
        } catch (const std::invalid_argument &ex) {
          throw std::runtime_error(std::string("[settings] CHANNEL_SET.INTERVALS: ") +
                                   ex.what());
        }
        request.channel_forms.push_back(std::move(row));
      }
    }
  }
  if (request.source_forms.empty() && request.channel_forms.empty())
    throw std::runtime_error("[settings] no KLINE_SOURCE_SET or CHANNEL_SET");
  return request;
}

} // namespace mityane::data
