/* enums.h */
#pragma once
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace cuwacunu::data {
/* Generalities */
template <typename T> struct EnumTraits;

template <typename T> struct is_optional_enum : std::false_type {};

template <typename E>
struct is_optional_enum<std::optional<E>> : std::is_enum<E> {};

template <typename T>
[[nodiscard]] inline std::string enum_to_string(T enumValue) {
  static_assert(std::is_enum_v<T>,
                "enum_to_string<T> requires T to be an enum type");
  return EnumTraits<T>::toString(enumValue);
}

template <typename T>
[[nodiscard]] inline T string_to_enum(const std::string &str) {
  static_assert(std::is_enum_v<T>,
                "string_to_enum<T> requires T to be an enum type");
  return EnumTraits<T>::fromString(str);
}

// Convenience overload for callers with std::string_view; preserves your
// EnumTraits policy.
template <typename T>
[[nodiscard]] inline T string_to_enum(std::string_view sv) {
  static_assert(std::is_enum_v<T>,
                "string_to_enum<T> requires T to be an enum type");
  return EnumTraits<T>::fromString(std::string(sv));
}
/* interval_type_e */
enum class interval_type_e {
  /* utilities */ utility_constant,
  utility_sine,
  utility_triangular,
  /* seconds */ interval_1s,
  /* minutes */ interval_1m,
  interval_3m,
  interval_5m,
  interval_15m,
  interval_30m,
  /* hours */ interval_1h,
  interval_2h,
  interval_4h,
  interval_6h,
  interval_8h,
  interval_12h,
  /* days */ interval_1d,
  interval_3d,
  /* weeks */ interval_1w,
  /* month */ interval_1M
};
template <> struct EnumTraits<interval_type_e> {
  static std::string toString(interval_type_e value) {
    switch (value) {
    case interval_type_e::utility_constant:
      return "constant";
    case interval_type_e::utility_sine:
      return "sine";
    case interval_type_e::utility_triangular:
      return "triangular";
    case interval_type_e::interval_1s:
      return "1s";
    case interval_type_e::interval_1m:
      return "1m";
    case interval_type_e::interval_3m:
      return "3m";
    case interval_type_e::interval_5m:
      return "5m";
    case interval_type_e::interval_15m:
      return "15m";
    case interval_type_e::interval_30m:
      return "30m";
    case interval_type_e::interval_1h:
      return "1h";
    case interval_type_e::interval_2h:
      return "2h";
    case interval_type_e::interval_4h:
      return "4h";
    case interval_type_e::interval_6h:
      return "6h";
    case interval_type_e::interval_8h:
      return "8h";
    case interval_type_e::interval_12h:
      return "12h";
    case interval_type_e::interval_1d:
      return "1d";
    case interval_type_e::interval_3d:
      return "3d";
    case interval_type_e::interval_1w:
      return "1w";
    case interval_type_e::interval_1M:
      return "1M";
    default:
      throw std::invalid_argument(
          "Unknown interval_type_e requested to convert into str");
    }
  }
  static interval_type_e fromString(const std::string &str) {
    if (str == "constant")
      return interval_type_e::utility_constant;
    if (str == "sine")
      return interval_type_e::utility_sine;
    if (str == "triangular")
      return interval_type_e::utility_triangular;
    if (str == "1s")
      return interval_type_e::interval_1s;
    if (str == "1m")
      return interval_type_e::interval_1m;
    if (str == "3m")
      return interval_type_e::interval_3m;
    if (str == "5m")
      return interval_type_e::interval_5m;
    if (str == "15m")
      return interval_type_e::interval_15m;
    if (str == "30m")
      return interval_type_e::interval_30m;
    if (str == "1h")
      return interval_type_e::interval_1h;
    if (str == "2h")
      return interval_type_e::interval_2h;
    if (str == "4h")
      return interval_type_e::interval_4h;
    if (str == "6h")
      return interval_type_e::interval_6h;
    if (str == "8h")
      return interval_type_e::interval_8h;
    if (str == "12h")
      return interval_type_e::interval_12h;
    if (str == "1d")
      return interval_type_e::interval_1d;
    if (str == "3d")
      return interval_type_e::interval_3d;
    if (str == "1w")
      return interval_type_e::interval_1w;
    if (str == "1M")
      return interval_type_e::interval_1M;
    throw std::invalid_argument(
        "Unknown string requested to convert into interval_type_e: " + str);
  }
};


} // namespace cuwacunu::data
