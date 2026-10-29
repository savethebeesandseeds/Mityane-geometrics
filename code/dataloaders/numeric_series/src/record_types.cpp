// Extracted from cuwacunu_torch; see code/data/README.md for provenance.
/* registry_data.cpp */
#include "record_types.h"
#include "kline_feature_registry.h"
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mityane::data {

namespace {

constexpr double kPositiveFloor = 1e-12;
constexpr double kLog1pInputFloor = -1.0 + kPositiveFloor;
constexpr std::size_t kMaxDetailedNormalizationWarningsPerKind = 3;
constexpr i64 kUnixTimeMicrosecondThreshold = 100000000000000LL;

void require_complete_token(const std::string &raw, std::size_t consumed) {
  while (consumed < raw.size() && std::isspace(static_cast<unsigned char>(raw[consumed]))) ++consumed;
  if (consumed != raw.size()) throw std::invalid_argument("trailing characters in numeric CSV token");
}

template <typename Integer> Integer parse_integer_token(const std::string &raw) {
  std::size_t consumed = 0;
  const auto value = std::stoll(raw, &consumed);
  require_complete_token(raw, consumed);
  if (value < std::numeric_limits<Integer>::min() || value > std::numeric_limits<Integer>::max())
    throw std::out_of_range("integer CSV token exceeds field range");
  return static_cast<Integer>(value);
}

double parse_double_token(const std::string &raw) {
  std::size_t consumed = 0;
  const auto value = std::stod(raw, &consumed);
  require_complete_token(raw, consumed);
  return value;
}

enum class normalization_warning_kind_e : std::size_t {
  LogReturnValueNonPositive = 0,
  LogReturnValueNonFinite,
  LogReturnReferenceNonPositive,
  LogReturnReferenceNonFinite,
  Log1pBelowFloor,
  Log1pNonFinite,
  Count
};

struct normalization_warning_scope_state_t {
  bool active{false};
  std::string filename{};
  std::string policy{};
  normalization_warning_summary_t summary{};
  std::array<std::size_t,
             static_cast<std::size_t>(normalization_warning_kind_e::Count)>
      detail_counts{};
};

thread_local normalization_warning_scope_state_t
    g_normalization_warning_scope{};

void reset_normalization_warning_scope_state() {
  g_normalization_warning_scope = normalization_warning_scope_state_t{};
}

[[nodiscard]] ms_t canonical_exchange_time_ms(i64 raw_time) {
  while (raw_time >= kUnixTimeMicrosecondThreshold ||
         raw_time <= -kUnixTimeMicrosecondThreshold) {
    raw_time /= 1000;
  }
  return static_cast<ms_t>(raw_time);
}

[[nodiscard]] ms_t parse_exchange_time_ms(const std::string &raw) {
  return canonical_exchange_time_ms(parse_integer_token<i64>(raw));
}

std::size_t &warning_summary_slot(normalization_warning_summary_t &summary,
                                  normalization_warning_kind_e kind) {
  switch (kind) {
  case normalization_warning_kind_e::LogReturnValueNonPositive:
    return summary.log_return_value_nonpositive;
  case normalization_warning_kind_e::LogReturnValueNonFinite:
    return summary.log_return_value_nonfinite;
  case normalization_warning_kind_e::LogReturnReferenceNonPositive:
    return summary.log_return_reference_nonpositive;
  case normalization_warning_kind_e::LogReturnReferenceNonFinite:
    return summary.log_return_reference_nonfinite;
  case normalization_warning_kind_e::Log1pBelowFloor:
    return summary.log1p_below_floor;
  case normalization_warning_kind_e::Log1pNonFinite:
    return summary.log1p_nonfinite;
  case normalization_warning_kind_e::Count:
  default:
    return summary.log1p_nonfinite;
  }
}

[[nodiscard]] const char *
warning_kind_label(normalization_warning_kind_e kind) {
  switch (kind) {
  case normalization_warning_kind_e::LogReturnValueNonPositive:
    return "log_return value <= 0";
  case normalization_warning_kind_e::LogReturnValueNonFinite:
    return "log_return value non-finite";
  case normalization_warning_kind_e::LogReturnReferenceNonPositive:
    return "log_return reference <= 0";
  case normalization_warning_kind_e::LogReturnReferenceNonFinite:
    return "log_return reference non-finite";
  case normalization_warning_kind_e::Log1pBelowFloor:
    return "log1p input <= -1";
  case normalization_warning_kind_e::Log1pNonFinite:
    return "log1p input non-finite";
  case normalization_warning_kind_e::Count:
  default:
    return "unknown normalization warning";
  }
}

void note_normalization_warning(normalization_warning_kind_e kind,
                                double original, double clamped) {
  if (!g_normalization_warning_scope.active)
    return;

  ++warning_summary_slot(g_normalization_warning_scope.summary, kind);

  const std::size_t index = static_cast<std::size_t>(kind);
  std::size_t &detailed_count =
      g_normalization_warning_scope.detail_counts[index];
  if (detailed_count < kMaxDetailedNormalizationWarningsPerKind) {
    log_warn("[normalize_log_returns] policy=%s file=%s%s%s %s: original=%.17g "
             "clamped=%.17g\n",
             g_normalization_warning_scope.policy.c_str(), "",
             g_normalization_warning_scope.filename.c_str(), "",
             warning_kind_label(kind), original, clamped);
  }
  ++detailed_count;
}

[[nodiscard]] inline double
clamp_positive_for_log_return(double value,
                              normalization_warning_kind_e nonpositive_kind,
                              normalization_warning_kind_e nonfinite_kind) {
  if (std::isfinite(value)) {
    if (value > 0.0)
      return value;
    note_normalization_warning(nonpositive_kind, value, kPositiveFloor);
    return kPositiveFloor;
  }

  note_normalization_warning(nonfinite_kind, value, kPositiveFloor);
  return kPositiveFloor;
}

[[nodiscard]] inline double safe_log_return(double value, double reference) {
  const double safe_value = clamp_positive_for_log_return(
      value, normalization_warning_kind_e::LogReturnValueNonPositive,
      normalization_warning_kind_e::LogReturnValueNonFinite);
  const double safe_reference = clamp_positive_for_log_return(
      reference, normalization_warning_kind_e::LogReturnReferenceNonPositive,
      normalization_warning_kind_e::LogReturnReferenceNonFinite);
  return std::log(safe_value / safe_reference);
}

[[nodiscard]] inline double safe_log1p_value(double value) {
  if (std::isfinite(value)) {
    if (value > -1.0) {
      return value == 0.0 ? 0.0 : std::log1p(value);
    }

    note_normalization_warning(normalization_warning_kind_e::Log1pBelowFloor,
                               value, kLog1pInputFloor);
    return std::log1p(kLog1pInputFloor);
  }

  note_normalization_warning(normalization_warning_kind_e::Log1pNonFinite,
                             value, kLog1pInputFloor);
  return std::log1p(kLog1pInputFloor);
}

template <typename KlineRecord>
[[nodiscard]] double kline_feature_value(const KlineRecord &record,
                                         kline_feature_e feature) {
  switch (feature) {
  case kline_feature_e::open:
    return record.open_price;
  case kline_feature_e::high:
    return record.high_price;
  case kline_feature_e::low:
    return record.low_price;
  case kline_feature_e::close:
    return record.close_price;
  case kline_feature_e::volume:
    return record.volume;
  case kline_feature_e::quote_volume:
    return record.quote_asset_volume;
  case kline_feature_e::trades:
    return static_cast<double>(record.number_of_trades);
  case kline_feature_e::taker_buy_base:
    return record.taker_buy_base_volume;
  case kline_feature_e::taker_buy_quote:
    return record.taker_buy_quote_volume;
  }
  return 0.0;
}

void set_kline_cache_feature(kline_cache_t &record, kline_feature_e feature,
                             double value) {
  switch (feature) {
  case kline_feature_e::open:
    record.open_price = value;
    return;
  case kline_feature_e::high:
    record.high_price = value;
    return;
  case kline_feature_e::low:
    record.low_price = value;
    return;
  case kline_feature_e::close:
    record.close_price = value;
    return;
  case kline_feature_e::volume:
    record.volume = value;
    return;
  case kline_feature_e::quote_volume:
    record.quote_asset_volume = value;
    return;
  case kline_feature_e::trades:
    record.number_of_trades = value;
    return;
  case kline_feature_e::taker_buy_base:
    record.taker_buy_base_volume = value;
    return;
  case kline_feature_e::taker_buy_quote:
    record.taker_buy_quote_volume = value;
    return;
  }
}

template <typename KlineRecord>
[[nodiscard]] std::vector<double>
kline_tensor_features_from_registry(const KlineRecord &record) {
  std::vector<double> out(static_cast<std::size_t>(kKlineFeatureWidth), 0.0);
  for (const auto &entry : kKlineFeatureRegistry) {
    out[static_cast<std::size_t>(entry.coord)] =
        kline_feature_value(record, entry.feature);
  }
  return out;
}

[[nodiscard]] double
normalize_kline_feature_value(const kline_feature_descriptor_t &entry,
                              const kline_cache_t &current,
                              const kline_cache_t &previous_valid) {
  const double value = kline_feature_value(current, entry.feature);
  switch (entry.normalization_rule) {
  case kline_normalization_rule_e::log_return_to_previous_close:
    return safe_log_return(value, previous_valid.close_price);
  case kline_normalization_rule_e::log1p:
    return safe_log1p_value(value);
  }
  return value;
}

} // namespace

void begin_normalization_warning_scope(std::string_view filename,
                                       std::string_view policy_name) {
  reset_normalization_warning_scope_state();
  g_normalization_warning_scope.active = true;
  g_normalization_warning_scope.filename = std::string(filename);
  g_normalization_warning_scope.policy = std::string(policy_name);
}

normalization_warning_summary_t end_normalization_warning_scope() {
  normalization_warning_summary_t summary =
      g_normalization_warning_scope.summary;
  reset_normalization_warning_scope_state();
  return summary;
}

trade_t trade_t::from_csv(const std::string &line, char delimiter,
                          size_t line_number) {
  const size_t expected_fields = 7;
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  if (tokens.size() != expected_fields) {
    throw std::runtime_error(
        "[from_csv](trade_t) Incorrect number of fields in line " +
        std::to_string(line_number) + ": expected " +
        std::to_string(expected_fields) + ", got " +
        std::to_string(tokens.size()) + ". Line content: " + line);
  }

  trade_t trade;
  size_t idx = 0;

  try {
    trade.id = parse_integer_token<i64>(tokens[idx++]);
    trade.price = parse_double_token(tokens[idx++]);
    trade.qty = parse_double_token(tokens[idx++]);
    trade.quoteQty = parse_double_token(tokens[idx++]);
    trade.time = parse_exchange_time_ms(tokens[idx++]);

    // isBuyerMaker
    {
      std::string &bool_str = tokens[idx++];
      if (bool_str == "true" || bool_str == "1") {
        trade.isBuyerMaker = true;
      } else if (bool_str == "false" || bool_str == "0") {
        trade.isBuyerMaker = false;
      } else {
        throw std::runtime_error(
            "[from_csv](trade_t) Invalid boolean value for isBuyerMaker: " +
            bool_str);
      }
    }

    // isBestMatch
    {
      std::string &bool_str = tokens[idx++];
      if (bool_str == "true" || bool_str == "1") {
        trade.isBestMatch = true;
      } else if (bool_str == "false" || bool_str == "0") {
        trade.isBestMatch = false;
      } else {
        throw std::runtime_error(
            "[from_csv](trade_t) Invalid boolean value for isBestMatch: " +
            bool_str);
      }
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("[from_csv](trade_t) Error parsing tokens in line ") +
        std::to_string(line_number) + ": " + e.what());
  }

  return trade;
}

kline_t kline_t::from_csv(const std::string &line, char delimiter,
                          size_t line_number) {
  const size_t expected_fields =
      11 + 1; /* unused additional field (per Binance doc) */
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  if (tokens.size() != expected_fields) {
    throw std::runtime_error(
        "[from_csv](kline_t) Incorrect number of fields in line " +
        std::to_string(line_number) + ": expected " +
        std::to_string(expected_fields) + ", got " +
        std::to_string(tokens.size()) + ". Line content: " + line);
  }

  kline_t kline;
  size_t idx = 0;

  try {
    kline.open_time = parse_exchange_time_ms(tokens[idx++]);
    kline.open_price = parse_double_token(tokens[idx++]);
    kline.high_price = parse_double_token(tokens[idx++]);
    kline.low_price = parse_double_token(tokens[idx++]);
    kline.close_price = parse_double_token(tokens[idx++]);
    kline.volume = parse_double_token(tokens[idx++]);
    kline.close_time = parse_exchange_time_ms(tokens[idx++]);
    kline.quote_asset_volume = parse_double_token(tokens[idx++]);
    kline.number_of_trades = parse_integer_token<int32_t>(tokens[idx++]);
    kline.taker_buy_base_volume = parse_double_token(tokens[idx++]);
    kline.taker_buy_quote_volume = parse_double_token(tokens[idx++]);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("[from_csv](kline_t) Error parsing tokens in line ") +
        std::to_string(line_number) + ": " + e.what());
  }

  return kline;
}

basic_t basic_t::from_csv(const std::string &line, char delimiter,
                          size_t line_number) {
  const size_t expected_fields = 2;
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  if (tokens.size() != expected_fields) {
    throw std::runtime_error(
        "[from_csv](basic_t) Incorrect number of fields in line " +
        std::to_string(line_number) + ": expected " +
        std::to_string(expected_fields) + ", got " +
        std::to_string(tokens.size()) + ". Line content: " + line);
  }

  basic_t out;
  size_t idx = 0;

  try {
    out.time = parse_double_token(tokens[idx++]);
    out.value = parse_double_token(tokens[idx++]);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("[from_csv](basic_t) Error parsing tokens in line ") +
        std::to_string(line_number) + ": " + e.what());
  }

  return out;
}

/* --- --- --- --- --- --- --- --- --- --- --- */
/*                key_value                    */
/* --- --- --- --- --- --- --- --- --- --- --- */
kline_t::key_type_t kline_t::key_value() { return close_time; }
trade_t::key_type_t trade_t::key_value() { return time; }
basic_t::key_type_t basic_t::key_value() { return time; }

/* --- --- --- --- --- --- --- --- --- --- --- */
/*               from_binary                   */
/* --- --- --- --- --- --- --- --- --- --- --- */
kline_t kline_t::from_binary(const char *data) {
  kline_t obj;
  std::memcpy(&obj, data, sizeof(kline_t));
  return obj;
}

trade_t trade_t::from_binary(const char *data) {
  trade_t obj;
  std::memcpy(&obj, data, sizeof(trade_t));
  return obj;
}

basic_t basic_t::from_binary(const char *data) {
  basic_t obj;
  std::memcpy(&obj, data, sizeof(basic_t));
  return obj;
}

trade_cache_t trade_cache_t::from_binary(const char *data) {
  trade_cache_t obj;
  std::memcpy(&obj, data, sizeof(trade_cache_t));
  return obj;
}

kline_cache_t kline_cache_t::from_binary(const char *data) {
  kline_cache_t obj;
  std::memcpy(&obj, data, sizeof(kline_cache_t));
  return obj;
}

basic_cache_t basic_cache_t::from_binary(const char *data) {
  basic_cache_t obj;
  std::memcpy(&obj, data, sizeof(basic_cache_t));
  return obj;
}

trade_cache_t trade_cache_t::from_raw(const raw_type_t &raw) {
  trade_cache_t out{};
  out.id = raw.id;
  out.price = raw.price;
  out.qty = raw.qty;
  out.quoteQty = raw.quoteQty;
  out.time = raw.time;
  out.isBuyerMaker = raw.isBuyerMaker;
  out.isBestMatch = raw.isBestMatch;
  return out;
}

kline_cache_t kline_cache_t::from_raw(const raw_type_t &raw) {
  kline_cache_t out{};
  out.open_time = raw.open_time;
  out.open_price = raw.open_price;
  out.high_price = raw.high_price;
  out.low_price = raw.low_price;
  out.close_price = raw.close_price;
  out.volume = raw.volume;
  out.close_time = raw.close_time;
  out.quote_asset_volume = raw.quote_asset_volume;
  out.number_of_trades = static_cast<double>(raw.number_of_trades);
  out.taker_buy_base_volume = raw.taker_buy_base_volume;
  out.taker_buy_quote_volume = raw.taker_buy_quote_volume;
  return out;
}

basic_cache_t basic_cache_t::from_raw(const raw_type_t &raw) {
  basic_cache_t out{};
  out.time = raw.time;
  out.value = raw.value;
  return out;
}

/* --- --- --- --- --- --- --- --- --- --- --- */
/*             tensor_features                 */
/* --- --- --- --- --- --- --- --- --- --- --- */
std::vector<double> kline_t::tensor_features() const {
  return kline_tensor_features_from_registry(*this);
}

std::vector<double> trade_t::tensor_features() const {
  return {// static_cast<double>(id),
          price, qty, quoteQty,
          // static_cast<double>(time),
          static_cast<double>(isBuyerMaker), static_cast<double>(isBestMatch)};
}

std::vector<double> basic_t::tensor_features() const {
  return {// time,
          value};
}

std::vector<double> trade_cache_t::tensor_features() const {
  return {price, qty, quoteQty, static_cast<double>(isBuyerMaker),
          static_cast<double>(isBestMatch)};
}

std::vector<double> kline_cache_t::tensor_features() const {
  return kline_tensor_features_from_registry(*this);
}

std::vector<double> basic_cache_t::tensor_features() const { return {value}; }

trade_cache_t
trade_cache_t::normalize_log_returns(const trade_cache_t &current,
                                     const trade_cache_t *previous_valid) {
  if (previous_valid == nullptr) {
    return trade_cache_t::null_instance(current.time);
  }
  trade_cache_t normalized = current;
  const double reference_price = previous_valid->price;

  normalized.price = safe_log_return(current.price, reference_price);
  normalized.qty = safe_log1p_value(current.qty);
  normalized.quoteQty = safe_log1p_value(current.quoteQty);
  return normalized;
}

kline_cache_t
kline_cache_t::normalize_log_returns(const kline_cache_t &current,
                                     const kline_cache_t *previous_valid) {
  if (previous_valid == nullptr) {
    return kline_cache_t::null_instance(current.close_time);
  }
  kline_cache_t normalized = current;
  for (const auto &entry : kKlineFeatureRegistry) {
    set_kline_cache_feature(
        normalized, entry.feature,
        normalize_kline_feature_value(entry, current, *previous_valid));
  }
  return normalized;
}

basic_cache_t
basic_cache_t::normalize_log_returns(const basic_cache_t &current,
                                     const basic_cache_t *previous_valid) {
  if (previous_valid == nullptr) {
    return basic_cache_t::null_instance(current.time);
  }
  basic_cache_t normalized = current;
  const double reference_value = previous_valid->value;
  normalized.value = safe_log_return(current.value, reference_value);
  return normalized;
}

/* --- --- --- --- --- --- --- --- --- --- --- */
/*               null_instance                 */
/* --- --- --- --- --- --- --- --- --- --- --- */
kline_t kline_t::null_instance(key_type_t key_value) {
  kline_t dnew;

  dnew.open_time = INT64_MIN;
  dnew.open_price = 0.0;
  dnew.high_price = 0.0;
  dnew.low_price = 0.0;
  dnew.close_price = 0.0;
  dnew.volume = 0.0;
  dnew.close_time = key_value;
  dnew.quote_asset_volume = 0.0;
  dnew.number_of_trades = -1;
  dnew.taker_buy_base_volume = 0.0;
  dnew.taker_buy_quote_volume = 0.0;

  return dnew;
}

trade_t trade_t::null_instance(key_type_t key_value) {
  trade_t dnew;

  dnew.id = -1;
  dnew.price = 0.0;
  dnew.qty = 0.0;
  dnew.quoteQty = 0.0;
  dnew.time = key_value;
  dnew.isBuyerMaker = false;
  dnew.isBestMatch = false;

  return dnew;
}

basic_t basic_t::null_instance(key_type_t key_value) {
  basic_t dnew;

  dnew.time = key_value;
  dnew.value = std::numeric_limits<double>::min();

  return dnew;
}

trade_cache_t trade_cache_t::null_instance(key_type_t key_value) {
  trade_cache_t dnew{};
  dnew.id = -1;
  dnew.price = 0.0;
  dnew.qty = 0.0;
  dnew.quoteQty = 0.0;
  dnew.time = key_value;
  dnew.isBuyerMaker = false;
  dnew.isBestMatch = false;
  return dnew;
}

kline_cache_t kline_cache_t::null_instance(key_type_t key_value) {
  kline_cache_t dnew{};
  dnew.open_time = INT64_MIN;
  dnew.open_price = 0.0;
  dnew.high_price = 0.0;
  dnew.low_price = 0.0;
  dnew.close_price = 0.0;
  dnew.volume = 0.0;
  dnew.close_time = key_value;
  dnew.quote_asset_volume = 0.0;
  dnew.number_of_trades = -1.0;
  dnew.taker_buy_base_volume = 0.0;
  dnew.taker_buy_quote_volume = 0.0;
  return dnew;
}

basic_cache_t basic_cache_t::null_instance(key_type_t key_value) {
  basic_cache_t dnew{};
  dnew.time = key_value;
  dnew.value = std::numeric_limits<double>::min();
  return dnew;
}

/* --- --- --- --- --- --- --- --- --- --- --- */
/*                  is_valid                   */
/* --- --- --- --- --- --- --- --- --- --- --- */
bool kline_t::is_valid() const { return open_time != INT64_MIN; }
bool trade_t::is_valid() const { return id >= 0; }
bool basic_t::is_valid() const {
  return value != std::numeric_limits<double>::min();
}
bool trade_cache_t::is_valid() const { return id >= 0; }
bool kline_cache_t::is_valid() const { return open_time != INT64_MIN; }
bool basic_cache_t::is_valid() const {
  return value != std::numeric_limits<double>::min();
}

trade_cache_t::key_type_t trade_cache_t::key_value() { return time; }
kline_cache_t::key_type_t kline_cache_t::key_value() { return close_time; }
basic_cache_t::key_type_t basic_cache_t::key_value() { return time; }

/* --- --- --- --- --- --- --- --- --- --- --- */
/*                   to_csv                    */
/* --- --- --- --- --- --- --- --- --- --- --- */
void kline_t::to_csv(std::ostream &os, char delimiter) const {
  os << std::fixed << std::setprecision(8) << open_time << delimiter
     << open_price << delimiter << high_price << delimiter << low_price
     << delimiter << close_price << delimiter << volume << delimiter
     << close_time << delimiter << quote_asset_volume << delimiter
     << number_of_trades << delimiter << taker_buy_base_volume << delimiter
     << taker_buy_quote_volume << delimiter << 0;
}

void trade_t::to_csv(std::ostream &os, char delimiter) const {
  os << std::fixed << std::setprecision(8) << id << delimiter << price
     << delimiter << qty << delimiter << quoteQty << delimiter << time
     << delimiter << std::boolalpha << isBuyerMaker << delimiter << isBestMatch;
}

void basic_t::to_csv(std::ostream &os, char delimiter) const {
  os << std::fixed << std::setprecision(8) << time << delimiter << value;
}


} // namespace mityane::data
