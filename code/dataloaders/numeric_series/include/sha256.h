// SHA256 of bytes and files for the raw-cache door. Header only.
// SPDX-License-Identifier: MIT
#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mityane::data {

namespace detail {

inline std::uint32_t sha256_rotr(std::uint32_t value, std::uint32_t bits) {
  return (value >> bits) | (value << (32 - bits));
}

inline void sha256_block(std::uint32_t state[8], const unsigned char block[64]) {
  static constexpr std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::uint32_t words[64];
  for (int i = 0; i < 16; ++i) {
    words[i] = (std::uint32_t(block[i * 4]) << 24) |
               (std::uint32_t(block[i * 4 + 1]) << 16) |
               (std::uint32_t(block[i * 4 + 2]) << 8) |
               std::uint32_t(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = sha256_rotr(words[i - 15], 7) ^
                             sha256_rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
    const std::uint32_t s1 = sha256_rotr(words[i - 2], 17) ^
                             sha256_rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }
  std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
    const std::uint32_t ch = (e & f) ^ (~e & g);
    const std::uint32_t t1 = h + s1 + ch + k[i] + words[i];
    const std::uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

inline std::string sha256_finish(std::uint32_t state[8], std::uint64_t bits,
                                 unsigned char *tail, std::size_t tail_len) {
  tail[tail_len++] = 0x80;
  if (tail_len > 56) {
    while (tail_len < 64)
      tail[tail_len++] = 0;
    sha256_block(state, tail);
    tail_len = 0;
  }
  while (tail_len < 56)
    tail[tail_len++] = 0;
  for (int shift = 56; shift >= 0; shift -= 8)
    tail[tail_len++] = static_cast<unsigned char>((bits >> shift) & 0xff);
  sha256_block(state, tail);
  std::ostringstream hex;
  hex << std::hex;
  for (int i = 0; i < 8; ++i) {
    for (int shift = 28; shift >= 0; shift -= 4)
      hex << "0123456789abcdef"[(state[i] >> shift) & 0xf];
  }
  return hex.str();
}

} // namespace detail

[[nodiscard]] inline std::string sha256_bytes_hex(std::string_view bytes) {
  std::uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char block[64];
  std::size_t filled = 0;
  for (unsigned char byte : bytes) {
    block[filled++] = byte;
    if (filled == 64) {
      detail::sha256_block(state, block);
      filled = 0;
    }
  }
  return detail::sha256_finish(state, std::uint64_t(bytes.size()) * 8, block, filled);
}

[[nodiscard]] inline std::string sha256_file_hex(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read file for SHA256: " + path);
  std::uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char block[64];
  std::size_t filled = 0;
  std::uint64_t total = 0;
  char buffer[4096];
  while (in) {
    in.read(buffer, sizeof buffer);
    const auto count = in.gcount();
    for (std::streamsize n = 0; n < count; ++n) {
      block[filled++] = static_cast<unsigned char>(buffer[n]);
      ++total;
      if (filled == 64) {
        detail::sha256_block(state, block);
        filled = 0;
      }
    }
  }
  if (in.bad())
    throw std::runtime_error("failed while reading file for SHA256: " + path);
  return detail::sha256_finish(state, total * 8, block, filled);
}

[[nodiscard]] inline bool sha256_hex_equal(std::string_view actual, std::string_view expected) {
  if (actual.size() != 64 || expected.size() != 64)
    return false;
  for (std::size_t i = 0; i < 64; ++i) {
    const auto left = std::tolower(static_cast<unsigned char>(actual[i]));
    const auto right = std::tolower(static_cast<unsigned char>(expected[i]));
    if (!std::isxdigit(left) || !std::isxdigit(right) || left != right)
      return false;
  }
  return true;
}

} // namespace mityane::data
