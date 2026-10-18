// SPDX-License-Identifier: MIT
#include "cuwacunu/config/document.h"

#include <chrono>
#include <iostream>

namespace {
using cuwacunu::config::Document;
void check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <typename Action> void rejects(Action action, const char *message) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  check(rejected, message);
}
} // namespace

int main() {
  try {
    const auto doc = Document::parse(R"(
      # Repository configuration
      Encoder {
        HISTORY_LENGTH = 128; // Positive history
        EMBEDDING_SIZE = 64; /* Native representation width */
        DEVICE = cuda;
        ENABLED = yes;
        RATE = 1e-3;
        CHANNEL_IDS = 0, 3, -2;
      }
      SOURCE { PATH = ../data/klines.csv; }
    )", "/tmp/cuwacunu-config/config/settings.dsl");
    const auto &encoder = doc.block("encoder");
    doc.reject_unknown_blocks({"encoder", "source"});
    encoder.reject_unknown_keys({"history_length", "embedding_size", "device",
                                 "enabled", "rate", "channel_ids"});
    check(encoder.integer("history_length") == 128, "integer parsing");
    check(encoder.contains(" history_length ") &&
              encoder.integer(" history_length ") == 128 &&
              encoder.integer_or(" history_length ", 8) == 128,
          "getters normalize key whitespace consistently");
    check(encoder.integer_or("missing", 8) == 8, "integer default");
    check(encoder.string("device") == "cuda", "string parsing");
    check(encoder.boolean("enabled"), "boolean parsing");
    check(encoder.number("rate") == 1e-3, "number parsing");
    check(encoder.integers("channel_ids") == std::vector<int64_t>({0, 3, -2}),
          "list parsing");
    check(doc.resolve_path(doc.block("source").string("path")) ==
              "/tmp/cuwacunu-config/data/klines.csv", "relative path");
    check(doc.resolve_path("/tmp/one/../two") == "/tmp/two", "absolute path");
    rejects([&] { encoder.string("missing"); }, "missing required key");
    rejects([&] { encoder.reject_unknown_keys({"device"}); }, "unknown key");
    rejects([&] { doc.reject_unknown_blocks({"encoder"}); }, "unknown block");
    rejects([&] { doc.block("missing"); }, "missing block");
    rejects([] { Document::parse("A { X = 1; x = 2; }"); }, "duplicate key");
    rejects([] { Document::parse("A {} a {}"); }, "duplicate block");
    for (const auto text : {"A X = 1;", "A { X 1; }", "A { X = 1;"})
      rejects([&] { Document::parse(text); }, "malformed syntax");
    for (const auto text : {"/*", "A {} /* unfinished",
                           "A { X = 1; } /*\nB { X = 2; }",
                           "A {} /* closed */ /* unfinished"})
      rejects([&] { Document::parse(text); },
              "unterminated block comment must reject the whole document");
    const auto line_comments = Document::parse(R"(
      A {
        X = 1; # Ignore a block opener /* in this line comment.
        Y = 2; // Ignore another /* through the newline.
        /* # and // inside a closed block comment stay comment text. */
      }
      // A final line comment can contain /* without a closing marker.
    )");
    check(line_comments.block("A").integer("X") == 1 &&
              line_comments.block("A").integer("Y") == 2,
          "comment preflight preserves imported line and block rules");
    for (const auto value : {"12tail", "1.2", "9223372036854775808"})
      rejects([&] { Document::parse(std::string("A { X = ") + value + "; }")
                        .block("A").integer("X"); }, "malformed integer");
    for (const auto value : {"nan", "inf", "1e9999", "1.2tail"})
      rejects([&] { Document::parse(std::string("A { X = ") + value + "; }")
                        .block("A").number("X"); }, "malformed number");
    for (const auto value : {"1,,2", "1,", "x,2"})
      rejects([&] { Document::parse(std::string("A { X = ") + value + "; }")
                        .block("A").integers("X"); }, "malformed list");
    rejects([] { Document::parse("A { X = maybe; }").block("A").boolean("X"); },
            "malformed bool");
    rejects([] { Document::parse("A { X = ; }").block("A").integer_or("X", 5); },
            "blank authored value must not use default");
    rejects([] { Document::parse("A {}").resolve_path("relative.csv"); },
            "relative path without origin");

    // Documents and blocks own data; parsing another document cannot mutate them.
    const auto second = Document::parse("A { X = 77; }");
    check(second.block("A").integer("X") == 77 &&
              encoder.integer("history_length") == 128, "owning storage");

    namespace fs = std::filesystem;
    const auto directory = fs::temp_directory_path() /
        ("cuwacunu-config-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    const auto file = directory / "settings.dsl";
    {
      std::ofstream output(file);
      output << "A { X = 42; }";
    }
    const auto loaded = Document::read(file);
    check(loaded.block("A").integer("X") == 42 &&
              loaded.resolve_path("data.csv") == directory / "data.csv",
          "file reading and origin");
    fs::remove(file);
    fs::remove(directory);
    rejects([&] { Document::read(file); }, "missing file");
    std::cout << "configuration document checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
