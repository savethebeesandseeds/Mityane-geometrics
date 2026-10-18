// SPDX-License-Identifier: MIT
#pragma once
#include <cstdarg>
#include <cstdio>

namespace cuwacunu::data {
inline void log_warn(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
}
inline void log_dbg(const char *, ...) {}
} // namespace cuwacunu::data
