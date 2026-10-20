// SPDX-License-Identifier: MIT
// Private bounded netstring bridge. This is deliberately not a JSON parser or
// model executor. Request field order mirrors request.mjs and paired_response.h.
#include "mityane/operations/counterfactuals/paired_response.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cf = mityane::operations::counterfactuals::v1;
namespace {
constexpr std::size_t maximum_request = 64U * 1024U * 1024U;
constexpr std::size_t maximum_token = 65536U;
constexpr std::int64_t safe_integer = 9007199254740991LL;

[[noreturn]] void malformed(const char* detail) { throw std::runtime_error(detail); }

bool valid_utf8(std::string_view value) {
  for (std::size_t index = 0; index < value.size();) {
    const auto lead = static_cast<unsigned char>(value[index++]);
    if (lead < 0x80) { if (lead == 0) return false; continue; }
    std::uint32_t code = 0, minimum = 0;
    std::size_t remaining = 0;
    if (lead >= 0xc2 && lead <= 0xdf) { code = lead & 0x1fU; minimum = 0x80; remaining = 1; }
    else if (lead >= 0xe0 && lead <= 0xef) { code = lead & 0x0fU; minimum = 0x800; remaining = 2; }
    else if (lead >= 0xf0 && lead <= 0xf4) { code = lead & 0x07U; minimum = 0x10000; remaining = 3; }
    else return false;
    if (remaining > value.size() - index) return false;
    while (remaining-- != 0) {
      const auto next = static_cast<unsigned char>(value[index++]);
      if ((next & 0xc0U) != 0x80U) return false;
      code = (code << 6U) | (next & 0x3fU);
    }
    if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
  }
  return true;
}

class Reader {
 public:
  explicit Reader(std::string bytes) : bytes_(std::move(bytes)) {}

  std::string_view token() {
    const auto colon = bytes_.find(':', position_);
    if (colon == std::string::npos || colon == position_ || colon - position_ > 6)
      malformed("Invalid private netstring length");
    const std::string_view digits(bytes_.data() + position_, colon - position_);
    if (digits.size() > 1 && digits.front() == '0') malformed("Noncanonical netstring length");
    std::size_t count = 0;
    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || count > maximum_token)
      malformed("Invalid or oversized private token");
    const auto start = colon + 1;
    if (count >= bytes_.size() - start || bytes_[start + count] != ',') malformed("Truncated private token");
    position_ = start + count + 1;
    return std::string_view(bytes_.data() + start, count);
  }

  std::string text() {
    const auto value = token();
    if (!valid_utf8(value)) malformed("Private string must be valid UTF-8 without NUL");
    return std::string(value);
  }
  std::int64_t integer() {
    const auto value = token();
    std::int64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        result < -safe_integer || result > safe_integer) malformed("Invalid safe integer");
    return result;
  }
  std::size_t size() {
    const auto value = integer();
    if (value < 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max())
      malformed("Invalid size value");
    return static_cast<std::size_t>(value);
  }
  double number() {
    const auto value = token();
    double result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result,
                                        std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !std::isfinite(result))
      malformed("Invalid finite binary64 number");
    return result;
  }
  bool boolean() {
    const auto value = token();
    if (value == "true") return true;
    if (value == "false") return false;
    malformed("Invalid boolean");
  }
  template <typename T> T enumeration(std::initializer_list<std::pair<std::string_view, T>> choices) {
    const auto value = token();
    for (const auto& [name, result] : choices) if (name == value) return result;
    malformed("Invalid enum value");
  }
  template <typename Function> auto optional(Function function) -> std::optional<decltype(function())> {
    const auto present = token();
    if (present == "0") return std::nullopt;
    if (present != "1") malformed("Invalid optional marker");
    return function();
  }
  template <typename Function> auto array(Function function, std::size_t maximum = 10000)
      -> std::vector<decltype(function())> {
    const auto count = size();
    if (count > maximum) malformed("Private array exceeds its declared limit");
    std::vector<decltype(function())> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) result.push_back(function());
    return result;
  }
  void finish() const { if (position_ != bytes_.size()) malformed("Trailing private request bytes"); }

 private:
  std::string bytes_;
  std::size_t position_ = 0;
};

cf::ArtifactRef artifact(Reader& reader) { return {reader.text(), reader.text()}; }
cf::Dependency dependency(Reader& reader) {
  cf::Dependency value;
  value.role = reader.text();
  value.artifact = artifact(reader);
  value.fit_population = reader.text();
  value.selection_population = reader.text();
  value.pretraining_exposure = reader.text();
  value.available_at = reader.optional([&] { return reader.integer(); });
  return value;
}
cf::SpaceRef space(Reader& reader) {
  cf::SpaceRef value;
  value.artifact = artifact(reader);
  value.provider = artifact(reader);
  value.preprocessing = artifact(reader);
  value.fitted_transform = artifact(reader);
  value.output_selector = reader.text();
  value.native_shape = reader.array([&] { return reader.size(); }, 64);
  value.layout = reader.text();
  value.normalization = reader.text();
  value.geometry = reader.text();
  value.input_role = reader.text();
  return value;
}
cf::Axis axis(Reader& reader) { return {reader.text(), reader.text(), reader.text()}; }
cf::PipelineRef pipeline(Reader& reader) {
  cf::PipelineRef value;
  value.artifact = artifact(reader);
  value.response_model = artifact(reader);
  value.configuration_sha256 = reader.text();
  value.stochastic_policy = reader.text();
  value.provider_revision_known = reader.boolean();
  value.dependencies = reader.array([&] { return dependency(reader); }, 1000);
  return value;
}
cf::AttributeRule rule(Reader& reader) {
  cf::AttributeRule value;
  value.name = reader.text();
  value.units = reader.text();
  value.role = reader.enumeration<cf::AttributeRole>({
    {"unspecified", cf::AttributeRole::unspecified}, {"changed", cf::AttributeRole::changed},
    {"held_fixed", cf::AttributeRole::held_fixed}, {"coupled", cf::AttributeRole::coupled}});
  value.minimum = reader.number();
  value.maximum = reader.number();
  return value;
}
cf::ScalarValue scalar(Reader& reader) { return {reader.text(), reader.text(), reader.number()}; }
cf::RecordedCheck check(Reader& reader) {
  cf::RecordedCheck value;
  value.state = reader.enumeration<cf::CheckState>({{"unspecified", cf::CheckState::unspecified},
    {"passed", cf::CheckState::passed}, {"rejected", cf::CheckState::rejected}});
  value.evidence = artifact(reader);
  value.detail = reader.text();
  return value;
}
cf::TemporalSupport timing(Reader& reader) {
  cf::TemporalSupport value;
  value.support_start = reader.optional([&] { return reader.integer(); });
  value.support_end = reader.optional([&] { return reader.integer(); });
  value.available_at = reader.optional([&] { return reader.integer(); });
  value.vintage = reader.text();
  return value;
}
cf::Branch branch(Reader& reader) {
  cf::Branch value;
  value.input = artifact(reader);
  value.source_observation = artifact(reader);
  value.parent_input_id = reader.text();
  value.origin = reader.enumeration<cf::Origin>({{"unspecified", cf::Origin::unspecified},
    {"observed", cf::Origin::observed}, {"edited_input", cf::Origin::edited_input},
    {"simulated_input", cf::Origin::simulated_input}, {"latent_candidate", cf::Origin::latent_candidate}});
  value.representation = artifact(reader);
  value.representation_kind = reader.enumeration<cf::RepresentationKind>({
    {"unspecified", cf::RepresentationKind::unspecified},
    {"observation_embedding", cf::RepresentationKind::observation_embedding},
    {"task_state", cf::RepresentationKind::task_state},
    {"predicted_state", cf::RepresentationKind::predicted_state},
    {"transformed_candidate", cf::RepresentationKind::transformed_candidate},
    {"generated_content", cf::RepresentationKind::generated_content}});
  value.parent_representation_id = reader.text();
  value.space = space(reader);
  value.pipeline = artifact(reader);
  value.response = artifact(reader);
  value.response_execution_id = reader.text();
  value.output_position = reader.integer();
  value.mask = artifact(reader);
  value.output_valid = reader.boolean();
  value.upstream_succeeded = reader.boolean();
  value.timing = timing(reader);
  value.attributes = reader.array([&] { return scalar(reader); });
  value.axes = reader.array([&] { return axis(reader); });
  value.response_values = reader.array([&] { return reader.number(); });
  return value;
}
cf::Pair pair(Reader& reader) {
  cf::Pair value;
  value.id = reader.text();
  value.unit_id = reader.text();
  value.entity_id = reader.text();
  value.group_id = reader.text();
  value.baseline = reader.optional([&] { return branch(reader); });
  value.alternative = reader.optional([&] { return branch(reader); });
  value.constructor = artifact(reader);
  value.change_definition = reader.text();
  value.interpretation = reader.text();
  value.control_evidence = artifact(reader);
  value.alternative_validity = check(reader);
  value.support = check(reader);
  value.interpretability = check(reader);
  return value;
}
cf::Request request(Reader& reader) {
  if (reader.token() != "mityane-paired-response-request-v1") malformed("Unsupported private wire version");
  cf::Request value;
  value.request_id = reader.text();
  value.operation_run_id = reader.text();
  value.contract_id = reader.text();
  value.contract_revision = reader.text();
  value.rer_id = reader.text();
  value.rer_revision = reader.text();
  value.claim_level = reader.enumeration<cf::ClaimLevel>({{"unspecified", cf::ClaimLevel::unspecified},
    {"controlled_model_response", cf::ClaimLevel::controlled_model_response},
    {"observational_contrast", cf::ClaimLevel::observational_contrast},
    {"identified_intervention_effect", cf::ClaimLevel::identified_intervention_effect}});
  value.purpose = reader.text();
  value.population_id = reader.text();
  value.sampling_rule = reader.text();
  value.response_horizon = reader.text();
  value.permitted_interpretation = reader.text();
  value.permissions_ref = reader.text();
  value.resource_budget = reader.text();
  value.evidence_destination = reader.text();
  value.code_sha256 = reader.text();
  value.configuration_sha256 = reader.text();
  value.temporal_mode = reader.enumeration<cf::TemporalMode>({{"unspecified", cf::TemporalMode::unspecified},
    {"non_temporal", cf::TemporalMode::non_temporal}, {"retrospective", cf::TemporalMode::retrospective},
    {"as_of", cf::TemporalMode::as_of}});
  value.as_of = reader.optional([&] { return reader.integer(); });
  value.non_temporal_justification = reader.text();
  value.expected_space = space(reader);
  value.expected_pipeline = pipeline(reader);
  value.response_axes = reader.array([&] { return axis(reader); });
  value.selected_output_position = reader.integer();
  value.attribute_rules = reader.array([&] { return rule(reader); });
  value.control_tolerance = reader.number();
  value.pairs = reader.array([&] { return pair(reader); }, 10000);
  reader.finish();
  return value;
}

void quoted(std::ostream& output, std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  output << '"';
  for (const unsigned char code : value) {
    if (code == '"' || code == '\\') output << '\\' << static_cast<char>(code);
    else if (code < 0x20) output << "\\u00" << hex[code >> 4U] << hex[code & 15U];
    else output << static_cast<char>(code);
  }
  output << '"';
}
void number(std::ostream& output, double value) {
  if (!std::isfinite(value)) malformed("Unexpected nonfinite operation result");
  std::array<char, 64> bytes{};
  const auto converted = std::to_chars(bytes.data(), bytes.data() + bytes.size(), value,
                                      std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (converted.ec != std::errc{}) malformed("Could not encode operation result");
  output.write(bytes.data(), converted.ptr - bytes.data());
}
void optional_integer(std::ostream& output, const std::optional<std::int64_t>& value) {
  if (value) output << *value; else output << "null";
}
void values(std::ostream& output, const std::vector<double>& value) {
  output << '[';
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index != 0) output << ',';
    number(output, value[index]);
  }
  output << ']';
}
void diagnostics(std::ostream& output, const std::vector<cf::Diagnostic>& value) {
  output << '[';
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index != 0) output << ',';
    output << "{\"code\":"; quoted(output, cf::to_string(value[index].code));
    output << ",\"record_id\":"; quoted(output, value[index].record_id);
    output << ",\"message\":"; quoted(output, value[index].message);
    output << '}';
  }
  output << ']';
}
void result(std::ostream& output, const cf::RunResult& value) {
  output << "{\"admitted\":" << (value.admitted ? "true" : "false") << ",\"admission_diagnostics\":";
  diagnostics(output, value.admission_diagnostics);
  output << ",\"pairs\":[";
  for (std::size_t index = 0; index < value.pairs.size(); ++index) {
    if (index != 0) output << ',';
    const auto& entry = value.pairs[index];
    output << "{\"pair_index\":" << entry.pair_index << ",\"pair_id\":"; quoted(output, entry.pair_id);
    output << ",\"state\":"; quoted(output, cf::to_string(entry.state));
    output << ",\"diagnostics\":"; diagnostics(output, entry.diagnostics);
    output << ",\"contrast\":";
    if (entry.contrast) {
      output << "{\"baseline\":"; values(output, entry.contrast->baseline);
      output << ",\"alternative\":"; values(output, entry.contrast->alternative);
      output << ",\"delta\":"; values(output, entry.contrast->delta); output << '}';
    } else output << "null";
    output << ",\"available_at\":"; optional_integer(output, entry.available_at); output << '}';
  }
  const auto& summary = value.summary;
  output << "],\"summary\":{\"requested\":" << summary.requested << ",\"succeeded\":" << summary.succeeded
         << ",\"abstained\":" << summary.abstained << ",\"failed\":" << summary.failed
         << ",\"unresolved\":" << summary.unresolved << ",\"success_rate\":";
  if (summary.success_rate) number(output, *summary.success_rate); else output << "null";
  output << "}}\n";
}
std::string stdin_bytes() {
  std::string bytes;
  std::array<char, 65536> buffer{};
  while (std::cin) {
    std::cin.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = std::cin.gcount();
    if (static_cast<std::size_t>(count) > maximum_request - bytes.size()) malformed("Private request exceeds 64 MiB");
    bytes.append(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!std::cin.eof()) malformed("Could not read private request");
  return bytes;
}
} // namespace

int main(int argc, char**) {
  try {
    if (argc != 1) malformed("Private bridge accepts request bytes on stdin only");
    Reader reader(stdin_bytes());
    const auto input = request(reader);
    const auto compared = cf::compare(input);
    result(std::cout, compared);
    if (!std::cout) malformed("Could not write private result");
    return 0; // Valid operation refusals are data; the outer tool sets its exit code.
  } catch (const std::exception& error) {
    std::cerr << "paired-response private bridge: " << error.what() << '\n';
    return 2;
  }
}
