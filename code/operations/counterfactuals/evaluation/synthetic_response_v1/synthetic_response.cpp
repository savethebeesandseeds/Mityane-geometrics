#include "mityane/operations/counterfactuals/paired_response.h"
#include "cuwacunu/artifacts/v1/sidecar.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cf = mityane::operations::counterfactuals::v1;
namespace ar = cuwacunu::artifacts::v1;
namespace fs = std::filesystem;

namespace {

constexpr const char* study = "OR06-SYNTH-RESP-001";
constexpr const char* version = "1.0.0";
constexpr const char* fit_method =
    "ols_normal_equations_partial_pivot_1e-12_train_population_sd";
constexpr std::array<const char*, 4> arms{
    "mixed", "raw_polynomial", "raw_linear", "lossy"};
constexpr std::array<int, 6> changes{-1024, -512, 0, 512, 1024, 8192};
constexpr std::array<const char*, 4> feature_definitions{
    "a+b,a-b,ab+a^2,ab-a^2", "a,b,ab,a^2", "a,b", "b,b^2,b^3,b^4"};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::string number(double value) {
  require(std::isfinite(value), "nonfinite numeric output refused");
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17) << value;
  return out.str();
}

std::string hex(std::string_view bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const unsigned char byte : bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot read " + path.string());
  std::ostringstream bytes;
  bytes << input.rdbuf();
  require(!input.bad(), "read failure for " + path.string());
  return bytes.str();
}

// glibc implements the ISO C exclusive-create 'x' mode. No existing output is
// truncated, including in a concurrent invocation or after an interrupted run.
class NewFile {
  std::FILE* file_ = nullptr;
  std::string path_;
public:
  explicit NewFile(const fs::path& path) : path_(path.string()) {
    file_ = std::fopen(path_.c_str(), "wbx");
    require(file_ != nullptr, "exclusive creation failed: " + path_);
  }
  NewFile(const NewFile&) = delete;
  NewFile& operator=(const NewFile&) = delete;
  ~NewFile() { if (file_) std::fclose(file_); }
  void write(std::string_view bytes) {
    require(file_ != nullptr, "write after close: " + path_);
    require(std::fwrite(bytes.data(), 1, bytes.size(), file_) == bytes.size(),
            "write failure: " + path_);
  }
  void close() {
    require(file_ != nullptr, "duplicate close: " + path_);
    auto* file = std::exchange(file_, nullptr);
    require(std::fclose(file) == 0, "close failure: " + path_);
  }
};

std::vector<std::string> fields(std::string_view line, char delimiter = '\t') {
  std::vector<std::string> result;
  std::size_t begin = 0;
  for (;;) {
    const auto end = line.find(delimiter, begin);
    result.emplace_back(line.substr(begin, end == line.npos ? end : end - begin));
    if (end == line.npos) return result;
    begin = end + 1;
  }
}

struct TextLine { std::string parsed; std::string actual; };
std::vector<TextLine> lines(const std::string& bytes) {
  require(!bytes.empty() && bytes.back() == '\n', "TSV must end with LF");
  require(bytes.find('\0') == bytes.npos, "TSV contains a NUL byte");
  std::vector<TextLine> result;
  for (std::size_t begin = 0; begin < bytes.size();) {
    const auto end = bytes.find('\n', begin);
    auto parsed = bytes.substr(begin, end - begin);
    if (!parsed.empty() && parsed.back() == '\r') parsed.pop_back();
    require(!parsed.empty(), "TSV contains an empty row");
    result.push_back({std::move(parsed), bytes.substr(begin, end + 1 - begin)});
    begin = end + 1;
  }
  return result;
}

template <typename Integer>
Integer integer(const std::string& value, const std::string& context) {
  Integer result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
              !value.empty() && std::to_string(result) == value,
          "invalid canonical integer for " + context);
  return result;
}

double real(const std::string& value, const std::string& context) {
  require(!value.empty() && value.find_first_not_of("0123456789+-.eE") == value.npos,
          "invalid decimal for " + context);
  char* end = nullptr;
  errno = 0;
  const double result = std::strtod(value.c_str(), &end);
  require(end == value.c_str() + value.size() && errno != ERANGE && std::isfinite(result),
          "invalid finite decimal for " + context);
  return result;
}

struct Unit {
  std::string id;
  int a_num;
  int b_num;
  int noise_sign;
  std::string actual_row;
};
struct Population {
  std::string bytes;
  std::string split;
  std::vector<Unit> units;
};

Population load_population(const fs::path& path, bool fitting) {
  Population population;
  population.bytes = read_file(path);
  const auto input = lines(population.bytes);
  require(input.front().parsed == "unit\ta_num\tb_num\tnoise_sign", "wrong input TSV header");
  require(input.size() > 1 && input.size() <= 8194, "invalid input population size");
  std::set<int> b_values;
  for (std::size_t row = 1; row < input.size(); ++row) {
    const auto item = fields(input[row].parsed);
    require(item.size() == 4, "input row must have four fields");
    const auto separator = item[0].rfind('-');
    require(separator != item[0].npos, "unit ID lacks split/index");
    const auto split = item[0].substr(0, separator);
    const auto index = integer<std::size_t>(item[0].substr(separator + 1), "unit index");
    require(index == row - 1, "unit IDs must be ordered contiguous zero-based indices");
    require(fitting ? split == "training" : split == "development" || split == "assessment",
            "wrong input split for command");
    if (population.split.empty()) population.split = split;
    require(population.split == split, "mixed input splits refused");
    Unit unit{item[0], integer<int>(item[1], "a_num"), integer<int>(item[2], "b_num"),
              integer<int>(item[3], "noise_sign"), input[row].actual};
    require(unit.a_num >= -3072 && unit.a_num <= 3072 &&
                unit.b_num >= -4096 && unit.b_num <= 4096,
            "raw coordinates outside generator bounds");
    require(unit.noise_sign == -1 || unit.noise_sign == 1, "noise_sign must be +/-1");
    require(b_values.insert(unit.b_num).second, "duplicate b_num/source unit refused");
    population.units.push_back(std::move(unit));
  }
  return population;
}

std::vector<double> features(std::size_t arm, int a_num, int b_num) {
  const double a = static_cast<double>(a_num) / 4096;
  const double b = static_cast<double>(b_num) / 4096;
  switch (arm) {
    case 0: return {a + b, a - b, a * b + a * a, a * b - a * a};
    case 1: return {a, b, a * b, a * a};
    case 2: return {a, b};
    case 3: return {b, b * b, b * b * b, b * b * b * b};
    default: throw std::runtime_error("unknown feature arm");
  }
}

std::string coordinates(const std::vector<double>& values) {
  std::string result;
  for (double value : values) {
    if (!result.empty()) result += ',';
    result += number(value);
  }
  return result;
}

double reference(const Unit& unit) {
  const double a = static_cast<double>(unit.a_num) / 4096;
  const double b = static_cast<double>(unit.b_num) / 4096;
  return 1 + 2 * a - b + 1.5 * a * b + 0.5 * a * a;
}

struct Head {
  double intercept = 0;
  std::vector<double> mean;
  std::vector<double> scale;
  std::vector<double> coefficient;
};
using Model = std::array<Head, 4>;

std::vector<double> solve(std::vector<std::vector<double>> matrix,
                          std::vector<double> target) {
  const auto size = matrix.size();
  for (std::size_t column = 0; column < size; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < size; ++row)
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
    require(std::isfinite(matrix[pivot][column]) &&
                std::abs(matrix[pivot][column]) > 1e-12,
            "OLS pivot refusal at column " + std::to_string(column));
    std::swap(matrix[pivot], matrix[column]);
    std::swap(target[pivot], target[column]);
    for (std::size_t row = column + 1; row < size; ++row) {
      const double multiplier = matrix[row][column] / matrix[column][column];
      for (std::size_t j = column; j < size; ++j)
        matrix[row][j] -= multiplier * matrix[column][j];
      target[row] -= multiplier * target[column];
    }
  }
  std::vector<double> coefficients(size, 0);
  for (std::size_t remaining = size; remaining > 0; --remaining) {
    const auto row = remaining - 1;
    double value = target[row];
    for (std::size_t column = row + 1; column < size; ++column)
      value -= matrix[row][column] * coefficients[column];
    coefficients[row] = value / matrix[row][row];
    require(std::isfinite(coefficients[row]), "nonfinite OLS coefficient refused");
  }
  return coefficients;
}

Model fit(const Population& training) {
  require(training.units.size() >= 5, "OLS requires at least five training units");
  Model result;
  for (std::size_t arm = 0; arm < arms.size(); ++arm) {
    auto& head = result[arm];
    const auto count = features(arm, 0, 0).size();
    head.mean.assign(count, 0);
    head.scale.assign(count, 0);
    std::vector<std::vector<double>> raw;
    for (const auto& unit : training.units) {
      raw.push_back(features(arm, unit.a_num, unit.b_num));
      for (std::size_t j = 0; j < count; ++j) head.mean[j] += raw.back()[j];
    }
    const auto population = static_cast<double>(training.units.size());
    for (auto& mean : head.mean) mean /= population;
    for (const auto& row : raw)
      for (std::size_t j = 0; j < count; ++j) {
        const double centered = row[j] - head.mean[j];
        head.scale[j] += centered * centered;
      }
    for (auto& scale : head.scale) {
      scale = std::sqrt(scale / population);
      require(std::isfinite(scale) && scale > 0, "degenerate training feature refused");
    }
    std::vector<std::vector<double>> normal(count + 1,
                                            std::vector<double>(count + 1, 0));
    std::vector<double> target(count + 1, 0);
    for (std::size_t i = 0; i < raw.size(); ++i) {
      std::vector<double> row{1};
      for (std::size_t j = 0; j < count; ++j)
        row.push_back((raw[i][j] - head.mean[j]) / head.scale[j]);
      const double label = reference(training.units[i]) +
                           static_cast<double>(training.units[i].noise_sign) / 32;
      for (std::size_t j = 0; j <= count; ++j) {
        target[j] += row[j] * label;
        for (std::size_t k = 0; k <= count; ++k) normal[j][k] += row[j] * row[k];
      }
    }
    auto coefficients = solve(std::move(normal), std::move(target));
    head.intercept = coefficients.front();
    head.coefficient.assign(coefficients.begin() + 1, coefficients.end());
  }
  return result;
}

std::string model_text(const Model& model) {
  std::string result = "arm\tfeature\tmean\tscale\tcoefficient\n";
  for (std::size_t arm = 0; arm < arms.size(); ++arm) {
    const auto prefix = std::string(arms[arm]) + '\t';
    const auto& head = model[arm];
    result += prefix + "intercept\t0\t1\t" + number(head.intercept) + '\n';
    for (std::size_t feature = 0; feature < head.mean.size(); ++feature)
      result += prefix + std::to_string(feature) + '\t' + number(head.mean[feature]) +
                '\t' + number(head.scale[feature]) + '\t' +
                number(head.coefficient[feature]) + '\n';
  }
  return result;
}

Model load_model(const std::string& bytes) {
  const auto input = lines(bytes);
  require(input.front().parsed == "arm\tfeature\tmean\tscale\tcoefficient",
          "wrong model TSV header");
  std::size_t row = 1;
  Model result;
  for (std::size_t arm = 0; arm < arms.size(); ++arm) {
    auto& head = result[arm];
    const auto count = features(arm, 0, 0).size();
    for (std::size_t item = 0; item <= count; ++item) {
      require(row < input.size(), "truncated model rows");
      const auto cells = fields(input[row++].parsed);
      require(cells.size() == 5 && cells[0] == arms[arm] &&
                  cells[1] == (item == 0 ? "intercept" : std::to_string(item - 1)),
              "wrong/duplicate/out-of-order model arm or feature");
      const double mean = real(cells[2], "mean");
      const double scale = real(cells[3], "scale");
      const double coefficient = real(cells[4], "coefficient");
      require(scale > 0, "model scale must be positive");
      if (item == 0) {
        require(mean == 0 && scale == 1, "intercept mean/scale must be 0/1");
        head.intercept = coefficient;
      } else {
        head.mean.push_back(mean);
        head.scale.push_back(scale);
        head.coefficient.push_back(coefficient);
      }
    }
  }
  require(row == input.size(), "unexpected model rows");
  return result;
}

std::map<std::string, std::string> load_metadata(const std::string& bytes,
                                              const std::string& model_sha,
                                              std::uint32_t seed) {
  const auto input = lines(bytes);
  require(input.front().parsed == "key\tvalue", "wrong model metadata header");
  const std::array<std::string, 7> expected{
      "study", "version", "seed", "training_rows", "training_sha256",
      "model_sha256", "fit_method"};
  require(input.size() == expected.size() + 1, "wrong model metadata row count");
  std::map<std::string, std::string> result;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto cells = fields(input[i + 1].parsed);
    require(cells.size() == 2 && cells[0] == expected[i], "wrong model metadata key/order");
    result.emplace(cells[0], cells[1]);
  }
  require(result.at("study") == study && result.at("version") == version &&
              result.at("fit_method") == fit_method, "unsupported model provenance");
  require(integer<std::uint32_t>(result.at("seed"), "metadata seed") == seed,
          "model seed mismatch");
  const auto count = integer<std::size_t>(result.at("training_rows"), "training_rows");
  require(count >= 5 && count <= 8193, "invalid metadata training population size");
  require(ar::is_sha256(result.at("training_sha256")) &&
              result.at("model_sha256") == model_sha,
          "model/training digest provenance mismatch");
  return result;
}

double predict(const Head& head, const std::vector<double>& raw) {
  require(raw.size() == head.mean.size(), "response input dimension mismatch");
  double response = head.intercept;
  for (std::size_t i = 0; i < raw.size(); ++i)
    response += head.coefficient[i] * ((raw[i] - head.mean[i]) / head.scale[i]);
  require(std::isfinite(response), "nonfinite fitted response refused");
  return response;
}

class Store {
  std::map<std::string, std::string> payloads_;
  std::map<std::string, cf::ArtifactRef> ids_;
  std::map<std::string, cf::ArtifactRef> descriptors_;
public:
  cf::ArtifactRef add(const std::string& id, const std::string& payload,
                      bool descriptor = false) {
    cf::ArtifactRef ref{id, ar::sha256_bytes(payload)};
    const auto [payload_it, inserted] = payloads_.emplace(ref.sha256, payload);
    require(inserted || payload_it->second == payload, "SHA256 payload collision");
    const auto [id_it, new_id] = ids_.emplace(id, ref);
    require(new_id || id_it->second == ref, "conflicting artifact ID: " + id);
    if (descriptor) descriptors_.emplace(id, ref);
    return ref;
  }
  void save(NewFile& artifacts, NewFile& descriptors) const {
    artifacts.write("sha256\tpayload_hex\n");
    for (const auto& [sha, payload] : payloads_)
      artifacts.write(sha + '\t' + hex(payload) + '\n');
    descriptors.write("id\tsha256\tpayload_hex\n");
    for (const auto& [id, ref] : descriptors_)
      descriptors.write(id + '\t' + ref.sha256 + '\t' + hex(payloads_.at(ref.sha256)) + '\n');
  }
};

std::string input_payload(int a_num, int b_num) {
  return "a_num\t" + std::to_string(a_num) + "\nb_num\t" + std::to_string(b_num) + '\n';
}

void record(NewFile& out, const std::string& arm, const std::string& pair,
            const std::string& scope, const std::string& field, const std::string& value) {
  out.write(arm + '\t' + pair + '\t' + scope + '\t' + field + '\t' + hex(value) + '\n');
}

void record_ref(NewFile& out, const std::string& arm, const std::string& pair,
                const std::string& scope, const std::string& name, const cf::ArtifactRef& ref) {
  record(out, arm, pair, scope, name + ".id", ref.id);
  record(out, arm, pair, scope, name + ".sha256", ref.sha256);
}

void record_space(NewFile& out, const std::string& arm, const std::string& pair,
                  const std::string& scope, const cf::SpaceRef& space) {
  record_ref(out, arm, pair, scope, "artifact", space.artifact);
  record_ref(out, arm, pair, scope, "provider", space.provider);
  record_ref(out, arm, pair, scope, "preprocessing", space.preprocessing);
  record_ref(out, arm, pair, scope, "fitted_transform", space.fitted_transform);
  record(out, arm, pair, scope, "output_selector", space.output_selector);
  std::string shape;
  for (auto dimension : space.native_shape) {
    if (!shape.empty()) shape += ',';
    shape += std::to_string(dimension);
  }
  record(out, arm, pair, scope, "native_shape", shape);
  record(out, arm, pair, scope, "layout", space.layout);
  record(out, arm, pair, scope, "normalization", space.normalization);
  record(out, arm, pair, scope, "geometry", space.geometry);
  record(out, arm, pair, scope, "input_role", space.input_role);
}

std::string optional_integer(const std::optional<std::int64_t>& value) {
  return value ? std::to_string(*value) : "NA";
}

void record_branch(NewFile& out, const std::string& arm, const std::string& pair,
                   const std::string& scope, const cf::Branch& branch) {
  record_ref(out, arm, pair, scope, "input", branch.input);
  record_ref(out, arm, pair, scope, "source_observation", branch.source_observation);
  record_ref(out, arm, pair, scope, "representation", branch.representation);
  record_ref(out, arm, pair, scope, "pipeline", branch.pipeline);
  record_ref(out, arm, pair, scope, "response", branch.response);
  record_ref(out, arm, pair, scope, "mask", branch.mask);
  record(out, arm, pair, scope, "parent_input_id", branch.parent_input_id);
  record(out, arm, pair, scope, "origin", branch.origin == cf::Origin::observed ? "observed" : "edited_input");
  record(out, arm, pair, scope, "representation_kind",
         branch.representation_kind == cf::RepresentationKind::observation_embedding ?
             "observation_embedding" : "transformed_candidate");
  record(out, arm, pair, scope, "parent_representation_id", branch.parent_representation_id);
  record_space(out, arm, pair, scope + ".space", branch.space);
  record(out, arm, pair, scope, "response_execution_id", branch.response_execution_id);
  record(out, arm, pair, scope, "output_position", std::to_string(branch.output_position));
  record(out, arm, pair, scope, "output_valid", branch.output_valid ? "true" : "false");
  record(out, arm, pair, scope, "upstream_succeeded", branch.upstream_succeeded ? "true" : "false");
  record(out, arm, pair, scope, "timing.support_start", optional_integer(branch.timing.support_start));
  record(out, arm, pair, scope, "timing.support_end", optional_integer(branch.timing.support_end));
  record(out, arm, pair, scope, "timing.available_at", optional_integer(branch.timing.available_at));
  record(out, arm, pair, scope, "timing.vintage", branch.timing.vintage);
  for (std::size_t i = 0; i < branch.attributes.size(); ++i) {
    const auto prefix = "attributes." + std::to_string(i) + '.';
    const auto& value = branch.attributes[i];
    record(out, arm, pair, scope, prefix + "name", value.name);
    record(out, arm, pair, scope, prefix + "units", value.units);
    record(out, arm, pair, scope, prefix + "value", number(value.value));
  }
  for (std::size_t i = 0; i < branch.axes.size(); ++i) {
    const auto prefix = "axes." + std::to_string(i) + '.';
    record(out, arm, pair, scope, prefix + "id", branch.axes[i].id);
    record(out, arm, pair, scope, prefix + "definition_revision", branch.axes[i].definition_revision);
    record(out, arm, pair, scope, prefix + "units", branch.axes[i].units);
  }
  record(out, arm, pair, scope, "response_values", coordinates(branch.response_values));
}

void record_request(NewFile& out, const std::string& arm, const cf::Request& request) {
  const auto common = [&](const std::string& key, const std::string& value) {
    record(out, arm, "-", "request", key, value);
  };
  common("request_id", request.request_id);
  common("operation_run_id", request.operation_run_id);
  common("contract_id", request.contract_id);
  common("contract_revision", request.contract_revision);
  common("rer_id", request.rer_id);
  common("rer_revision", request.rer_revision);
  common("claim_level", "controlled_model_response");
  common("purpose", request.purpose);
  common("population_id", request.population_id);
  common("sampling_rule", request.sampling_rule);
  common("response_horizon", request.response_horizon);
  common("permitted_interpretation", request.permitted_interpretation);
  common("permissions_ref", request.permissions_ref);
  common("resource_budget", request.resource_budget);
  common("evidence_destination", request.evidence_destination);
  common("code_sha256", request.code_sha256);
  common("configuration_sha256", request.configuration_sha256);
  common("temporal_mode", "non_temporal");
  common("as_of", optional_integer(request.as_of));
  common("non_temporal_justification", request.non_temporal_justification);
  common("selected_output_position", std::to_string(request.selected_output_position));
  common("control_tolerance", number(request.control_tolerance));
  record_space(out, arm, "-", "expected_space", request.expected_space);
  const auto& pipeline = request.expected_pipeline;
  record_ref(out, arm, "-", "pipeline", "artifact", pipeline.artifact);
  record_ref(out, arm, "-", "pipeline", "response_model", pipeline.response_model);
  record(out, arm, "-", "pipeline", "configuration_sha256", pipeline.configuration_sha256);
  record(out, arm, "-", "pipeline", "stochastic_policy", pipeline.stochastic_policy);
  record(out, arm, "-", "pipeline", "provider_revision_known", pipeline.provider_revision_known ? "true" : "false");
  for (const auto& dependency : pipeline.dependencies) {
    const auto scope = "dependency." + dependency.role;
    record_ref(out, arm, "-", scope, "artifact", dependency.artifact);
    record(out, arm, "-", scope, "role", dependency.role);
    record(out, arm, "-", scope, "fit_population", dependency.fit_population);
    record(out, arm, "-", scope, "selection_population", dependency.selection_population);
    record(out, arm, "-", scope, "pretraining_exposure", dependency.pretraining_exposure);
    record(out, arm, "-", scope, "available_at", optional_integer(dependency.available_at));
  }
  for (std::size_t i = 0; i < request.response_axes.size(); ++i) {
    const auto scope = "response_axis." + std::to_string(i);
    const auto& axis = request.response_axes[i];
    record(out, arm, "-", scope, "id", axis.id);
    record(out, arm, "-", scope, "definition_revision", axis.definition_revision);
    record(out, arm, "-", scope, "units", axis.units);
  }
  for (std::size_t i = 0; i < request.attribute_rules.size(); ++i) {
    const auto scope = "attribute_rule." + std::to_string(i);
    const auto& rule = request.attribute_rules[i];
    record(out, arm, "-", scope, "name", rule.name);
    record(out, arm, "-", scope, "units", rule.units);
    record(out, arm, "-", scope, "role", rule.role == cf::AttributeRole::changed ? "changed" : "held_fixed");
    record(out, arm, "-", scope, "minimum", number(rule.minimum));
    record(out, arm, "-", scope, "maximum", number(rule.maximum));
  }
  for (const auto& pair : request.pairs) {
    record(out, arm, pair.id, "pair", "id", pair.id);
    record(out, arm, pair.id, "pair", "unit_id", pair.unit_id);
    record(out, arm, pair.id, "pair", "entity_id", pair.entity_id);
    record(out, arm, pair.id, "pair", "group_id", pair.group_id);
    record_ref(out, arm, pair.id, "pair", "constructor", pair.constructor);
    record_ref(out, arm, pair.id, "pair", "control_evidence", pair.control_evidence);
    record(out, arm, pair.id, "pair", "change_definition", pair.change_definition);
    record(out, arm, pair.id, "pair", "interpretation", pair.interpretation);
    const std::array<std::pair<const char*, const cf::RecordedCheck*>, 3> checks{{
        {"alternative_validity", &pair.alternative_validity},
        {"support", &pair.support}, {"interpretability", &pair.interpretability}}};
    for (const auto& [scope, check] : checks) {
      record_ref(out, arm, pair.id, scope, "evidence", check->evidence);
      record(out, arm, pair.id, scope, "state", check->state == cf::CheckState::passed ? "passed" : "rejected");
      record(out, arm, pair.id, scope, "detail", check->detail);
    }
    record_branch(out, arm, pair.id, "baseline", *pair.baseline);
    record_branch(out, arm, pair.id, "alternative", *pair.alternative);
  }
}

cf::Request request_for(Store& store, std::size_t arm, const Population& population,
                        const Model& model, const std::string& model_bytes,
                        const std::string& metadata_bytes,
                        const std::map<std::string, std::string>& metadata,
                        std::uint32_t seed, const std::string& source_sha,
                        const std::string& config_sha) {
  const std::string arm_name = arms[arm];
  const auto prefix = std::string(study) + "/" + std::to_string(seed) + "/" + arm_name;
  const auto model_sha = ar::sha256_bytes(model_bytes);
  const auto binding = "study\t" + std::string(study) + "\nversion\t" + version +
                       "\nsource_sha256\t" + source_sha + "\nconfiguration_sha256\t" +
                       config_sha + "\nseed\t" + std::to_string(seed) + '\n';
  const auto model_ref = store.add(prefix + "/model", model_bytes, true);
  const auto metadata_ref = store.add(prefix + "/fit_metadata", metadata_bytes, true);
  const auto config = store.add(prefix + "/configuration_binding", binding, true);
  const auto provider = store.add(prefix + "/provider", binding + "arm\t" + arm_name +
      "\nfeatures\t" + feature_definitions[arm] + "\nprovider\tfixed_exact_arithmetic_v1\n", true);
  const auto preprocessing = store.add(prefix + "/preprocessing", binding +
      "method\ttraining_population_mean_and_sd\nmodel_sha256\t" + model_sha +
      "\nfit_metadata_sha256\t" + metadata_ref.sha256 + '\n', true);
  const auto transform = store.add(prefix + "/fitted_transform", model_bytes, true);
  const auto constructor = store.add(prefix + "/constructor", binding +
      "method\tadd_change_num_to_a_num_hold_b_num_exact\nchanges\t-1024,-512,0,512,1024,8192\n", true);
  const auto support = store.add(prefix + "/support_policy", binding +
      "a_min\t-1\na_max\t1\nb_min\t-1\nb_max\t1\n", true);
  const auto mask = store.add(prefix + "/mask", "output_position\t0\nvalid\ttrue\n", true);
  const auto space_ref = store.add(prefix + "/space", binding + "arm\t" + arm_name +
      "\nprovider_sha256\t" + provider.sha256 + "\npreprocessing_sha256\t" + preprocessing.sha256 +
      "\nfitted_transform_sha256\t" + transform.sha256 + "\nnative_shape\t" +
      std::to_string(model[arm].mean.size()) +
      "\noutput_selector\tfeatures\nlayout\tdense\nnormalization\ttrain_population_zscore_in_response_head\ngeometry\teuclidean\ninput_role\tresponse_input\n", true);
  const auto pipeline_ref = store.add(prefix + "/pipeline", binding +
      "provider_sha256\t" + provider.sha256 + "\npreprocessing_sha256\t" + preprocessing.sha256 +
      "\nfitted_transform_sha256\t" + transform.sha256 + "\nresponse_model_sha256\t" + model_ref.sha256 +
      "\nconstructor_sha256\t" + constructor.sha256 + "\nsupport_policy_sha256\t" + support.sha256 +
      "\nconfiguration_binding_sha256\t" + config.sha256 +
      "\nfit_metadata_sha256\t" + metadata_ref.sha256 +
      "\nstochastic_policy\tdeterministic\nprovider_revision_known\ttrue\n", true);

  cf::Request request;
  request.request_id = prefix + "/" + population.split + "/request";
  request.operation_run_id = prefix + "/" + population.split + "/run";
  request.contract_id = "OR06-PCR";
  request.contract_revision = "0.1.0";
  request.rer_id = "OR06-PCR-RER";
  request.rer_revision = "0.1.0";
  request.claim_level = cf::ClaimLevel::controlled_model_response;
  request.purpose = "Compare frozen synthetic response predictions under controlled raw input changes";
  request.population_id = std::string(study) + "/" + std::to_string(seed) + "/" +
                          population.split + "/sha256/" + ar::sha256_bytes(population.bytes);
  request.sampling_rule = "Every source unit, all six predeclared changes, no exclusions";
  request.response_horizon = "static_synthetic_no_temporal_horizon";
  request.permitted_interpretation = "model_input_response";
  request.permissions_ref = std::string(study) + "/synthetic_generated_inputs_no_external_data";
  request.resource_budget = std::to_string(population.units.size()) + " units x 6 changes x one arm; CPU only; no refit";
  request.evidence_destination = "retained_run_artifacts";
  request.code_sha256 = source_sha;
  request.configuration_sha256 = config_sha;
  request.temporal_mode = cf::TemporalMode::non_temporal;
  request.non_temporal_justification = "Synthetic independent scalar inputs and static deterministic responses; no temporal or causal claim";
  request.expected_space = {space_ref, provider, preprocessing, transform, "features",
      {model[arm].mean.size()}, "dense", "train_population_zscore_in_response_head",
      "euclidean", "response_input"};
  request.expected_pipeline.artifact = pipeline_ref;
  request.expected_pipeline.response_model = model_ref;
  request.expected_pipeline.configuration_sha256 = config_sha;
  request.expected_pipeline.stochastic_policy = "deterministic";
  request.expected_pipeline.provider_revision_known = true;
  const auto train_population = std::string(study) + "/" + std::to_string(seed) +
                                 "/training/sha256/" + metadata.at("training_sha256");
  const std::array<std::pair<const char*, cf::ArtifactRef>, 6> dependencies{{
      {"provider", provider}, {"preprocessing", preprocessing},
      {"fitted_transform", transform}, {"response_model", model_ref},
      {"support_policy", support}, {"constructor", constructor}}};
  for (const auto& [role, artifact] : dependencies) {
    const std::string role_name = role;
    const bool fitted = role_name == "preprocessing" || role_name == "fitted_transform" ||
                        role_name == "response_model";
    request.expected_pipeline.dependencies.push_back({role, artifact,
        fitted ? train_population : "none_fixed_by_design", "none_no_selection",
        "excluded_by_design", std::nullopt});
  }
  request.response_axes = {{"synthetic_response", "synthetic_response_units_v1", "synthetic_unit"}};
  request.selected_output_position = 0;
  request.attribute_rules = {{"a", "dimensionless", cf::AttributeRole::changed, -1, 1},
                             {"b", "dimensionless", cf::AttributeRole::held_fixed, -1, 1}};
  request.control_tolerance = 0;

  for (const auto& unit : population.units) {
    const auto source = store.add(std::string(study) + "/" + std::to_string(seed) +
                                    "/source/" + unit.id, unit.actual_row);
    const auto unit_prefix = prefix + "/" + unit.id;
    const auto base_z = features(arm, unit.a_num, unit.b_num);
    const auto base_value = predict(model[arm], base_z);
    const auto make_branch = [&](const std::string& branch_prefix, int a_num,
                                  const std::vector<double>& z, double response) {
      cf::Branch branch;
      branch.input = store.add(branch_prefix + "/input", input_payload(a_num, unit.b_num));
      branch.source_observation = source;
      branch.representation = store.add(branch_prefix + "/representation", "arm\t" +
          arm_name + "\nz\t" + coordinates(z) + '\n');
      branch.space = request.expected_space;
      branch.pipeline = pipeline_ref;
      branch.response = store.add(branch_prefix + "/response", "arm\t" + arm_name +
          "\nmodel_sha256\t" + model_sha + "\nvalue\t" + number(response) + '\n');
      branch.response_execution_id = branch_prefix + "/execution";
      branch.output_position = 0;
      branch.mask = mask;
      branch.output_valid = true;
      branch.upstream_succeeded = true;
      branch.timing.vintage = std::string(study) + "/static_generated_v1";
      branch.attributes = {{"a", "dimensionless", static_cast<double>(a_num) / 4096},
                           {"b", "dimensionless", static_cast<double>(unit.b_num) / 4096}};
      branch.axes = request.response_axes;
      branch.response_values = {response};
      return branch;
    };
    auto baseline = make_branch(unit_prefix + "/baseline", unit.a_num, base_z, base_value);
    baseline.origin = cf::Origin::observed;
    baseline.representation_kind = cf::RepresentationKind::observation_embedding;
    for (int change : changes) {
      const int alternative_a = unit.a_num + change;
      const auto pair_prefix = unit_prefix + "/change/" + std::to_string(change);
      const auto alternative_z = features(arm, alternative_a, unit.b_num);
      // Every alternative, including a no-change control, executes the readout.
      const double alternative_value = predict(model[arm], alternative_z);
      auto alternative = make_branch(pair_prefix + "/alternative", alternative_a,
                                      alternative_z, alternative_value);
      alternative.parent_input_id = baseline.input.id;
      alternative.origin = cf::Origin::edited_input;
      alternative.representation_kind = cf::RepresentationKind::transformed_candidate;
      alternative.parent_representation_id = baseline.representation.id;
      cf::Pair pair;
      pair.id = pair_prefix;
      pair.unit_id = unit.id;
      pair.entity_id = "synthetic_entity/" + unit.id;
      pair.group_id = "synthetic_seed/" + std::to_string(seed);
      pair.baseline = baseline;
      pair.alternative = std::move(alternative);
      pair.constructor = constructor;
      pair.change_definition = "a_num += " + std::to_string(change) + "; b_num held exactly fixed; denominator=4096";
      pair.interpretation = "model_input_response";
      pair.control_evidence = store.add(pair_prefix + "/control", "policy_sha256\t" +
          constructor.sha256 + "\nbase_b_num\t" + std::to_string(unit.b_num) +
          "\nalt_b_num\t" + std::to_string(unit.b_num) + "\noutput_position\t0\n");
      pair.alternative_validity = {cf::CheckState::passed,
          store.add(pair_prefix + "/validity", "constructor_sha256\t" + constructor.sha256 +
              "\nbase_a_num\t" + std::to_string(unit.a_num) + "\nchange_num\t" +
              std::to_string(change) + "\nalt_a_num\t" + std::to_string(alternative_a) +
              "\nresult\tpassed\n"), "Exact integer a edit and held-fixed b verified"};
      const bool supported = alternative_a >= -4096 && alternative_a <= 4096;
      pair.support = {supported ? cf::CheckState::passed : cf::CheckState::rejected,
          store.add(pair_prefix + "/support", "support_policy_sha256\t" + support.sha256 +
              "\nbase_a_num\t" + std::to_string(unit.a_num) + "\nalt_a_num\t" +
              std::to_string(alternative_a) + "\nb_num\t" + std::to_string(unit.b_num) +
              "\nresult\t" + (supported ? "passed" : "rejected") + '\n'),
          supported ? "Both branches within declared [-1,1] scalar support" :
                      "Stress alternative a exceeds declared [-1,1] support"};
      pair.interpretability = {cf::CheckState::passed,
          store.add(pair_prefix + "/interpretability", "interpretation\tmodel_input_response\nclaim\tcontrolled_model_response\npipeline_sha256\t" +
              pipeline_ref.sha256 + "\nresult\tpassed\n"),
          "Controlled response of the same frozen deterministic model; no causal interpretation"};
      request.pairs.push_back(std::move(pair));
    }
  }
  return request;
}

void fit_command(const fs::path& input, const fs::path& output, std::uint32_t seed) {
  const auto training = load_population(input, true);
  const auto model = fit(training);
  const auto bytes = model_text(model);
  const auto model_sha = ar::sha256_bytes(bytes);
  const auto metadata = "key\tvalue\nstudy\t" + std::string(study) + "\nversion\t" + version +
      "\nseed\t" + std::to_string(seed) + "\ntraining_rows\t" +
      std::to_string(training.units.size()) + "\ntraining_sha256\t" +
      ar::sha256_bytes(training.bytes) + "\nmodel_sha256\t" + model_sha +
      "\nfit_method\t" + fit_method + '\n';
  const auto sidecar = fs::path(output.string() + ".metadata.tsv");
  require(!fs::exists(output) && !fs::exists(sidecar), "model or model metadata already exists");
  NewFile model_file(output);
  NewFile metadata_file(sidecar);
  model_file.write(bytes);
  metadata_file.write(metadata);
  model_file.close();
  metadata_file.close();
  std::cout << "fit seed=" << seed << " training_rows=" << training.units.size()
            << " model_sha256=" << model_sha << '\n';
}

void evaluate_command(const fs::path& input, const fs::path& model_path,
                       const fs::path& output, std::uint32_t seed,
                       const std::string& source_sha, const std::string& config_sha) {
  require(ar::is_sha256(source_sha) && ar::is_sha256(config_sha), "invalid source/configuration SHA256");
  const auto population = load_population(input, false);
  const auto model_bytes = read_file(model_path);
  const auto metadata_bytes = read_file(fs::path(model_path.string() + ".metadata.tsv"));
  const auto model = load_model(model_bytes);
  const auto metadata = load_metadata(metadata_bytes, ar::sha256_bytes(model_bytes), seed);
  require(!fs::exists(output), "evaluation output directory already exists");
  require(fs::create_directory(output), "cannot exclusively create evaluation output directory");
  NewFile pairs(output / "pairs.tsv");
  NewFile artifacts(output / "artifacts.tsv");
  NewFile descriptors(output / "descriptors.tsv");
  NewFile records(output / "request_records.tsv");
  NewFile roster(output / "roster.tsv");
  NewFile summary(output / "summary.tsv");
  pairs.write("arm\tunit\tchange_num\ta_num\tb_num\talt_a_num\tstate\treason\trecorded_base\trecorded_alt\tdelta\tz_base\tz_alt\n");
  records.write("arm\tpair_id\tscope\tfield\tvalue_hex\n");
  roster.write("arm\tpair_id\tunit\tchange_num\n");
  summary.write("arm\trequested\tsucceeded\tabstained\tfailed\tunresolved\n");
  Store store;
  for (std::size_t arm = 0; arm < arms.size(); ++arm) {
    const auto request = request_for(store, arm, population, model, model_bytes,
                                      metadata_bytes, metadata, seed, source_sha, config_sha);
    const auto result = cf::compare(request);
    if (!result.admitted) {
      std::string diagnostic;
      for (const auto& issue : result.admission_diagnostics)
        diagnostic += cf::to_string(issue.code) + ":" + issue.message + "; ";
      throw std::runtime_error("study operation request rejected: " + diagnostic);
    }
    require(result.pairs.size() == request.pairs.size(), "operation truncated admitted roster");
    record_request(records, arms[arm], result.input);
    for (std::size_t index = 0; index < result.pairs.size(); ++index) {
      const auto& unit = population.units[index / changes.size()];
      const int change = changes[index % changes.size()];
      const auto& pair = result.input.pairs[index];
      const auto& item = result.pairs[index];
      require(item.pair_index == index && item.pair_id == pair.id, "operation reordered pair roster");
      require(item.diagnostics.size() <= 1, "unexpected multiple diagnostics in synthetic pair");
      const auto reason = item.diagnostics.empty() ? "NONE" : cf::to_string(item.diagnostics.front().code);
      const auto& base = *pair.baseline;
      const auto& alt = *pair.alternative;
      const auto delta = item.contrast ? number(item.contrast->delta.at(0)) : "NA";
      pairs.write(std::string(arms[arm]) + '\t' + unit.id + '\t' + std::to_string(change) +
          '\t' + std::to_string(unit.a_num) + '\t' + std::to_string(unit.b_num) + '\t' +
          std::to_string(unit.a_num + change) + '\t' + cf::to_string(item.state) + '\t' +
          reason + '\t' + number(base.response_values.at(0)) + '\t' +
          number(alt.response_values.at(0)) + '\t' + delta + '\t' +
          coordinates(features(arm, unit.a_num, unit.b_num)) + '\t' +
          coordinates(features(arm, unit.a_num + change, unit.b_num)) + '\n');
      roster.write(std::string(arms[arm]) + '\t' + pair.id + '\t' + unit.id + '\t' +
                   std::to_string(change) + '\n');
    }
    const auto& count = result.summary;
    require(count.requested == request.pairs.size() &&
                count.succeeded + count.abstained + count.failed + count.unresolved == count.requested,
            "operation summary does not reconcile roster");
    summary.write(std::string(arms[arm]) + '\t' + std::to_string(count.requested) + '\t' +
        std::to_string(count.succeeded) + '\t' + std::to_string(count.abstained) + '\t' +
        std::to_string(count.failed) + '\t' + std::to_string(count.unresolved) + '\n');
  }
  store.save(artifacts, descriptors);
  pairs.close();
  artifacts.close();
  descriptors.close();
  records.close();
  roster.close();
  summary.close();
  std::cout << "evaluate seed=" << seed << " split=" << population.split
            << " units=" << population.units.size() << " pairs="
            << population.units.size() * arms.size() * changes.size() << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::locale::global(std::locale::classic());
    if (argc == 5 && std::string(argv[1]) == "fit") {
      fit_command(argv[2], argv[3], integer<std::uint32_t>(argv[4], "seed"));
    } else if (argc == 8 && std::string(argv[1]) == "evaluate") {
      evaluate_command(argv[2], argv[3], argv[4], integer<std::uint32_t>(argv[5], "seed"), argv[6], argv[7]);
    } else {
      throw std::runtime_error("usage: synthetic_response fit TRAIN.tsv MODEL.tsv SEED | evaluate INPUT.tsv MODEL.tsv OUT_DIR SEED SOURCE_SHA CONFIG_SHA");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "synthetic_response: " << error.what() << '\n';
    return 1;
  }
}
