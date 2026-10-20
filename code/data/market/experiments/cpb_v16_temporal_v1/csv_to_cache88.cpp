// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cache88.h"
#include <cerrno>
#include <clocale>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
namespace cd = mityane::data::cpb_v15;
namespace fs = std::filesystem;
namespace {
void write_all(int fd, const char* bytes, size_t count) {
  while (count) {
    const auto written = ::write(fd, bytes, count);
    if (written < 0 && errno == EINTR) continue;
    cd::require(written > 0, "cache write failed; partial output retained");
    bytes += written; count -= static_cast<size_t>(written);
  }
}
std::string exact(const std::string& value, bool exists) {
  const fs::path file(value);
  cd::require(file.is_absolute() && file.lexically_normal() == file, "absolute canonical path required");
  cd::require(fs::canonical(file.parent_path()) == file.parent_path(), "redirected file parent refused");
  if (exists) cd::require(fs::canonical(file) == file && fs::is_regular_file(fs::symlink_status(file)), "canonical regular input required");
  return file.string();
}
}
int main(int argc, char** argv) {
  try {
    cd::require(fs::exists("/.dockerenv") && fs::canonical("/mityane") == "/mityane", "managed Linux /mityane container required");
    cd::require(std::setlocale(LC_NUMERIC, "C") != nullptr, "C numeric locale required");
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; i += 2) {
      cd::require(i + 1 < argc && (std::string(argv[i]) == "--input" || std::string(argv[i]) == "--output"
        || std::string(argv[i]) == "--first-open" || std::string(argv[i]) == "--rows")
        && options.emplace(argv[i], argv[i + 1]).second, "four unique explicit conversion arguments required");
    }
    cd::require(options.size() == 4, "usage: csv_to_cache88 --input CSV --output NEWCACHE --first-open MS --rows COUNT");
    const auto input = exact(options.at("--input"), true), output = exact(options.at("--output"), false);
    cd::require(input != output, "input and exclusive output must differ");
    cd::require(std::regex_match(input, std::regex("^/mityane/\\.temp/data/cpb-v16-temporal-001/attempt-[0-9]{3}/csv/(BTCUSDT|ETHUSDT|SOLUSDT)-1m-2024-03\\.csv$"))
      && fs::path(output).parent_path() == fs::path(input).parent_path()
      && fs::path(output).filename() == fs::path(input).stem().string() + ".cache.bin", "exact admitted March CSV/cache paths required");
    const auto first = cd::integer<int64_t>(options.at("--first-open")), rows = cd::integer<int64_t>(options.at("--rows"));
    cd::require(first == 1709251200000LL && rows == 44640, "only the declared March population is admitted by this executable");
    const int input_fd = ::open(input.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    cd::require(input_fd >= 0, "open canonical input failed");
    struct stat input_status{};
    const bool admitted = ::fstat(input_fd, &input_status) == 0 && S_ISREG(input_status.st_mode) && input_status.st_nlink == 1;
    ::close(input_fd);
    cd::require(admitted, "single-link regular input required");
    // Re-open through the admitted /proc descriptor, preventing a path switch
    // after open; the Node owner separately seals its input bytes before/after.
    const int source_fd = ::open(input.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    cd::require(source_fd >= 0, "open CSV failed");
    struct stat opened{};
    cd::require(::fstat(source_fd, &opened) == 0 && opened.st_dev == input_status.st_dev
      && opened.st_ino == input_status.st_ino && opened.st_size == input_status.st_size, "input redirected before conversion");
    std::ifstream source("/proc/self/fd/" + std::to_string(source_fd), std::ios::binary);
    cd::require(bool(source), "CSV descriptor read failed");
    const int destination = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    cd::require(destination >= 0, "exclusive cache output required; existing output preserved");
    cd::Conversion result{};
    try {
      result = cd::convert_csv(source, first, rows, [&](const cd::Cache88& record) {
        write_all(destination, reinterpret_cast<const char*>(&record), sizeof record);
      });
      cd::require(::fsync(destination) == 0, "cache sync failed");
      cd::require(::close(destination) == 0, "cache close failed");
    } catch (...) { ::close(destination); ::close(source_fd); throw; }
    ::close(source_fd);
    std::cout << "{\"kind\":\"cpb_v16_temporal_cache88_conversion_v1\",\"passed\":true,\"rows\":" << result.rows
      << ",\"first_open_ms\":" << result.first_open_ms << ",\"first_close_ms\":" << result.first_close_ms
      << ",\"last_close_ms\":" << result.last_close_ms << ",\"cache_bytes\":" << result.cache_bytes
      << ",\"normalizations\":0,\"gap_fills\":0,\"rows_filtered\":0}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CACHE88 conversion refused: " << error.what() << '\n'; return 1;
  }
}
