// SPDX-License-Identifier: MIT
#pragma once

#include "piaabo/parse/simple_kv_block.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace cuwacunu::config {

// An owning, typed view of the legacy NAME { KEY = value; } configuration.
// Parsing deliberately preserves the imported parser's unquoted value syntax.
class Block {
public:
  explicit Block(piaabo::parse::simple_kv::block_t value)
      : value_(std::move(value)) {}

  [[nodiscard]] const std::string &name() const { return value_.name; }
  [[nodiscard]] const auto &values() const { return value_.values; }
  [[nodiscard]] bool contains(std::string_view key) const {
    return value_.values.contains(normalize(key));
  }

  [[nodiscard]] std::string string(std::string_view key) const {
    return piaabo::parse::simple_kv::required(value_, normalize(key));
  }
  [[nodiscard]] std::string string_or(std::string_view key,
                                    std::string fallback) const {
    return contains(key) ? string(key) : std::move(fallback);
  }
  [[nodiscard]] int64_t integer(std::string_view key) const {
    return piaabo::parse::simple_kv::parse_i64(string(key));
  }
  [[nodiscard]] int64_t integer_or(std::string_view key,
                                 int64_t fallback) const {
    return contains(key) ? integer(key) : fallback;
  }
  [[nodiscard]] double number(std::string_view key) const {
    const auto result = piaabo::parse::simple_kv::parse_double(string(key));
    if (!std::isfinite(result))
      fail("nonfinite number for " + normalize(key));
    return result;
  }
  [[nodiscard]] double number_or(std::string_view key, double fallback) const {
    if (!std::isfinite(fallback))
      fail("nonfinite default for " + normalize(key));
    return contains(key) ? number(key) : fallback;
  }
  [[nodiscard]] bool boolean(std::string_view key) const {
    return piaabo::parse::simple_kv::parse_bool(string(key));
  }
  [[nodiscard]] bool boolean_or(std::string_view key, bool fallback) const {
    return contains(key) ? boolean(key) : fallback;
  }
  [[nodiscard]] std::vector<int64_t> integers(std::string_view key) const {
    const auto raw = string(key);
    std::vector<int64_t> result;
    std::size_t begin = 0;
    while (begin <= raw.size()) {
      const auto comma = raw.find(',', begin);
      const auto item = piaabo::parse::simple_kv::trim(raw.substr(
          begin, comma == std::string::npos ? std::string::npos : comma - begin));
      if (item.empty())
        fail("empty integer list item for " + normalize(key));
      result.push_back(piaabo::parse::simple_kv::parse_i64(item));
      if (comma == std::string::npos)
        break;
      begin = comma + 1;
    }
    return result;
  }
  [[nodiscard]] std::vector<int64_t>
  integers_or(std::string_view key, std::vector<int64_t> fallback) const {
    return contains(key) ? integers(key) : std::move(fallback);
  }

  void reject_unknown_keys(
      std::initializer_list<std::string_view> allowed) const {
    std::unordered_set<std::string> names;
    for (const auto key : allowed)
      names.insert(normalize(key));
    for (const auto &[key, value] : value_.values) {
      (void)value;
      if (!names.contains(key))
        fail("unknown key " + key);
    }
  }

private:
  [[nodiscard]] static std::string normalize(std::string_view name) {
    return piaabo::parse::simple_kv::uppercase(
        piaabo::parse::simple_kv::trim(std::string(name)));
  }
  [[noreturn]] void fail(const std::string &message) const {
    throw std::runtime_error("[config] " + name() + ": " + message);
  }
  piaabo::parse::simple_kv::block_t value_;
};

class Document {
public:
  [[nodiscard]] static Document
  parse(const std::string &text, std::filesystem::path origin = {}) {
    validate_comment_termination(text);
    Document result;
    if (!origin.empty())
      result.source_path_ = std::filesystem::absolute(origin).lexically_normal();
    std::unordered_set<std::string> names;
    for (auto &value : piaabo::parse::simple_kv::parse_blocks(text)) {
      if (!names.insert(value.name).second)
        throw std::runtime_error("[config] duplicate block " + value.name);
      result.blocks_.emplace_back(std::move(value));
    }
    return result;
  }

  [[nodiscard]] static Document read(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("[config] cannot open " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    if (input.bad())
      throw std::runtime_error("[config] cannot read " + path.string());
    return parse(contents.str(), path);
  }

  [[nodiscard]] const Block &block(std::string_view name) const {
    const auto target = piaabo::parse::simple_kv::uppercase(
        piaabo::parse::simple_kv::trim(std::string(name)));
    for (const auto &value : blocks_)
      if (value.name() == target)
        return value;
    throw std::runtime_error("[config] missing block " + target);
  }
  [[nodiscard]] const std::vector<Block> &blocks() const { return blocks_; }
  [[nodiscard]] const std::filesystem::path &source_path() const {
    return source_path_;
  }
  [[nodiscard]] std::filesystem::path resolve_path(std::string_view raw) const {
    std::filesystem::path path(
        piaabo::parse::simple_kv::trim(std::string(raw)));
    if (path.empty())
      throw std::runtime_error("[config] empty path");
    if (path.is_absolute())
      return path.lexically_normal();
    if (source_path_.empty())
      throw std::runtime_error("[config] relative path requires document origin");
    return (source_path_.parent_path() / path).lexically_normal();
  }
  void reject_unknown_blocks(
      std::initializer_list<std::string_view> allowed) const {
    std::unordered_set<std::string> names;
    for (const auto name : allowed)
      names.insert(piaabo::parse::simple_kv::uppercase(
          piaabo::parse::simple_kv::trim(std::string(name))));
    for (const auto &value : blocks_)
      if (!names.contains(value.name()))
        throw std::runtime_error("[config] unknown block " + value.name());
  }

private:
  // Keep the imported unquoted comment rules, but do not accept a truncated
  // file as a complete document. Comment markers inside line comments are inert.
  static void validate_comment_termination(const std::string &text) {
    for (std::size_t i = 0; i < text.size();) {
      if (text[i] == '#' ||
          (i + 1 < text.size() && text[i] == '/' && text[i + 1] == '/')) {
        while (i < text.size() && text[i] != '\n')
          ++i;
      } else if (i + 1 < text.size() && text[i] == '/' && text[i + 1] == '*') {
        const auto end = text.find("*/", i + 2);
        if (end == std::string::npos)
          throw std::runtime_error("[config] unclosed block comment");
        i = end + 2;
      } else {
        ++i;
      }
    }
  }

  std::vector<Block> blocks_;
  std::filesystem::path source_path_;
};

} // namespace cuwacunu::config
