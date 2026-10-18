// SPDX-License-Identifier: MIT
#include "embedding/shared/projection_diagnostic.h"
#include <ATen/Context.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[projection diagnostic] " + message);
}
bool safe_name(const std::string &name) {
  return !name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
std::string read_bytes(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot read declared input: " + path.string());
  std::ostringstream out;
  out << input.rdbuf();
  require(!input.bad(), "cannot finish reading declared input: " + path.string());
  return out.str();
}
std::string checksum(const std::string &bytes) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char c : bytes) { hash ^= c; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
void write_text(const fs::path &path, const std::string &text) {
  require(!fs::exists(path), "artifact already exists: " + path.string());
  std::ofstream output(path, std::ios::binary);
  require(bool(output), "cannot create artifact: " + path.string());
  output << text;
  output.close();
  require(bool(output), "cannot save artifact: " + path.string());
}
void save_archive(const fs::path &path, torch::serialize::OutputArchive &output) {
  require(!fs::exists(path), "archive already exists: " + path.string());
  archive::save_archive(path.string(), output);
}
// Explicit input declarations are the only archive discovery mechanism.
// Filenames marked test/testing are rejected as an additional role safeguard.
bool testing_filename(const fs::path &path) {
  std::string token;
  auto is_test = [&]() { return token == "test" || token == "testing"; };
  for (unsigned char c : path.filename().string()) {
    if (std::isalnum(c)) token += char(std::tolower(c));
    else { if (is_test()) return true; token.clear(); }
  }
  return is_test();
}
struct InputGuard {
  std::map<std::string, std::string> originals;
  void add(const std::string &path) {
    require(!path.empty() && fs::is_regular_file(path) && !testing_filename(path),
            "declared training/validation path missing, non-file, or marked testing: " + path);
    const auto resolved = fs::canonical(path).string();
    require(!testing_filename(resolved), "resolved archive is marked testing: " + resolved);
    if (!originals.count(resolved)) originals.emplace(resolved, read_bytes(resolved));
  }
  void verify() const {
    for (const auto &[path, original] : originals)
      require(read_bytes(path) == original, "input archive changed: " + path);
  }
  std::string manifest() const {
    std::ostringstream out;
    out << "{\"checksum_algorithm\":\"fnv1a64-file-bytes; non-cryptographic\","
        << "\"preservation_check\":\"exact file-byte equality\",\"files\":[";
    bool first = true;
    for (const auto &[path, bytes] : originals) {
      if (!first) out << ',';
      first = false;
      out << "{\"path\":" << quote(path) << ",\"bytes\":" << bytes.size()
          << ",\"checksum\":" << quote(checksum(bytes)) << '}';
    }
    return out.str() + "]}";
  }
};
// All measurement, including TinyProbe's manual_seed, preserves ambient RNG.
struct RngIsolation {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RngIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  ~RngIsolation() noexcept {
    try {
      for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
    } catch (...) { std::terminate(); }
  }
};
// Minimal strict JSON string-array reader for persisted source IDs, including
// escaped Unicode. No report/parser dependency and no archive interpretation.
class SourceParser {
  const std::string &text;
  size_t position{0};
  void space() { while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position; }
  char take() { require(position < text.size(), "truncated source-ID JSON"); return text[position++]; }
  uint32_t hex4() {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = take();
      value *= 16;
      if (c >= '0' && c <= '9') value += c - '0';
      else if (c >= 'a' && c <= 'f') value += c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') value += c - 'A' + 10;
      else require(false, "invalid source-ID Unicode escape");
    }
    return value;
  }
  static void utf8(std::string &out, uint32_t c) {
    if (c <= 0x7f) out += char(c);
    else if (c <= 0x7ff) { out += char(0xc0 | (c >> 6)); out += char(0x80 | (c & 63)); }
    else if (c <= 0xffff) { out += char(0xe0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 63)); out += char(0x80 | (c & 63)); }
    else { out += char(0xf0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 63)); out += char(0x80 | ((c >> 6) & 63)); out += char(0x80 | (c & 63)); }
  }
  std::string string() {
    require(take() == '"', "source IDs must be JSON strings");
    std::string out;
    while (true) {
      const unsigned char c = take();
      if (c == '"') return out;
      require(c >= 32, "unescaped source-ID control character");
      if (c != '\\') { out += char(c); continue; }
      const char escape = take();
      if (escape == '"' || escape == '\\' || escape == '/') out += escape;
      else if (escape == 'b') out += '\b';
      else if (escape == 'f') out += '\f';
      else if (escape == 'n') out += '\n';
      else if (escape == 'r') out += '\r';
      else if (escape == 't') out += '\t';
      else if (escape == 'u') {
        uint32_t code = hex4();
        if (code >= 0xd800 && code <= 0xdbff) {
          require(take() == '\\' && take() == 'u', "missing source-ID low surrogate");
          const auto low = hex4();
          require(low >= 0xdc00 && low <= 0xdfff, "invalid source-ID low surrogate");
          code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
        } else require(code < 0xdc00 || code > 0xdfff, "unpaired source-ID surrogate");
        utf8(out, code);
      } else require(false, "invalid source-ID JSON escape");
    }
  }
public:
  explicit SourceParser(const std::string &value) : text(value) {}
  std::vector<std::string> parse() {
    space(); require(take() == '[', "source IDs must be a JSON array"); space();
    std::vector<std::string> out;
    if (position < text.size() && text[position] == ']') ++position;
    else while (true) {
      out.push_back(string()); space();
      const char delimiter = take();
      if (delimiter == ']') break;
      require(delimiter == ',', "invalid source-ID JSON delimiter"); space();
    }
    space(); require(position == text.size(), "trailing source-ID JSON");
    return out;
  }
};
struct Split {
  FeatureSurface features;
  torch::Tensor labels;
  std::vector<std::string> source_ids;
};
Split load_split(const ProjectionArchiveInput &input, bool training) {
  torch::serialize::InputArchive features, labels;
  features.load_from(training ? input.training_features : input.validation_features, torch::kCPU);
  labels.load_from(training ? input.training_labels : input.validation_labels, torch::kCPU);
  Split out;
  torch::Tensor provenance, sources;
  features.read(input.feature_values_key, out.features.values, true);
  features.read(input.feature_valid_key, out.features.valid, true);
  features.read(input.feature_provenance_key, provenance, true);
  out.features.provenance = archive::tensor_text(provenance);
  validate_features(out.features);
  require(!out.features.provenance.empty(), "frozen feature provenance is empty");
  labels.read(input.labels_key, out.labels, true);
  labels.read(input.source_ids_key, sources, true);
  const auto source_text = archive::tensor_text(sources);
  out.source_ids = SourceParser(source_text).parse();
  require(out.labels.defined() && out.labels.device().is_cpu() && out.labels.scalar_type() == torch::kInt64 &&
          out.labels.sizes() == torch::IntArrayRef({out.features.values.size(0)}) &&
          out.labels.ge(0).logical_and(out.labels.le(1)).all().item<bool>() &&
          out.source_ids.size() == size_t(out.labels.size(0)), "binary labels/source rows do not match feature rows");
  std::map<std::string, std::vector<int64_t>> groups;
  for (int64_t row = 0; row < out.labels.size(0); ++row) groups[out.source_ids.at(row)].push_back(row);
  for (const auto &[id, rows] : groups)
    require(!id.empty() && rows.size() == 2 && out.labels[rows[0]].item<int64_t>() != out.labels[rows[1]].item<int64_t>(),
            "controlled archive requires two opposite-label variants per source");
  out.features.values = out.features.values.detach().clone();
  out.features.valid = out.features.valid.clone(); out.labels = out.labels.clone();
  return out;
}
std::string population(const Split &split, const torch::Tensor &valid) {
  std::map<std::string, std::pair<int64_t, int64_t>> groups;
  std::set<std::string> selected;
  int64_t classes[2]{0, 0}, complete = 0;
  for (int64_t row = 0; row < valid.size(0); ++row) {
    auto &group = groups[split.source_ids.at(row)]; ++group.first;
    if (valid[row].item<bool>()) { ++group.second; selected.insert(split.source_ids.at(row)); ++classes[split.labels[row].item<int64_t>()]; }
  }
  for (const auto &[id, count] : groups) { (void)id; if (count.first == count.second) ++complete; }
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total_rows\":" << valid.size(0) << ",\"valid_rows\":" << classes[0] + classes[1]
      << ",\"class_valid_rows\":[" << classes[0] << ',' << classes[1] << "],\"total_source_groups\":" << groups.size()
      << ",\"valid_source_groups\":" << selected.size() << ",\"complete_source_pairs\":" << complete
      << ",\"coverage\":" << double(classes[0] + classes[1]) / valid.size(0) << '}';
  return out.str();
}
std::string score_json(const Score &value) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total\":" << value.total << ",\"valid\":" << value.valid
      << ",\"correct\":" << value.correct << ",\"coverage\":" << value.coverage << ",\"accuracy\":";
  if (value.supported) out << value.accuracy; else out << "null";
  return out.str() + '}';
}
std::string interval_json(const GroupedInterval &value, int64_t replicates) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"source_groups\":" << value.source_groups
      << ",\"replicates\":" << replicates << ",\"confidence\":0.95,\"estimate\":";
  if (value.source_groups) out << value.estimate; else out << "null";
  out << ",\"lower\":";
  if (value.supported) out << value.lower; else out << "null";
  out << ",\"upper\":";
  if (value.supported) out << value.upper; else out << "null";
  return out.str() + '}';
}
std::string input_json(const ProjectionArchiveInput &input) {
  return "{\"id\":" + quote(input.id) + ",\"architecture\":" + quote(input.architecture) +
      ",\"master_seed\":" + quote(std::to_string(input.master_seed)) + ",\"checkpoint_steps\":" + std::to_string(input.checkpoint_steps) +
      ",\"training_features\":" + quote(input.training_features) + ",\"validation_features\":" + quote(input.validation_features) +
      ",\"training_labels\":" + quote(input.training_labels) + ",\"validation_labels\":" + quote(input.validation_labels) +
      ",\"feature_values_key\":" + quote(input.feature_values_key) + ",\"feature_valid_key\":" + quote(input.feature_valid_key) +
      ",\"feature_provenance_key\":" + quote(input.feature_provenance_key) + ",\"labels_key\":" + quote(input.labels_key) +
      ",\"source_ids_key\":" + quote(input.source_ids_key) + '}';
}
std::string card_json(const ProjectionDiagnosticRun &run) {
  std::ostringstream out;
  out << "{\"version\":1,\"protocol\":\"frozen-projection-diagnostic-v1\",\"stage\":\"development\","
      << "\"validation_only\":true,\"test_access\":false,\"checkpoint_training\":false,"
      << "\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty) << ",\"widths\":[";
  for (size_t i = 0; i < run.widths.size(); ++i) { if (i) out << ','; out << run.widths[i]; }
  out << "],\"native_reference\":true,\"threads\":" << run.threads << ",\"repetitions\":[";
  for (size_t i = 0; i < run.repetitions.size(); ++i) {
    if (i) out << ',';
    const auto &rep = run.repetitions[i];
    out << "{\"id\":" << quote(rep.id) << ",\"projection_seed\":" << quote(std::to_string(rep.projection_seed))
        << ",\"probe_seed\":" << quote(std::to_string(rep.probe_seed)) << '}';
  }
  out << "],\"inputs\":[";
  for (size_t i = 0; i < run.inputs.size(); ++i) { if (i) out << ','; out << input_json(run.inputs[i]); }
  out << "],\"recipe\":{"
      << "\"fit_population\":\"valid TRAINING rows only; both supervised classes required\","
      << "\"pipeline\":\"outer training population normalizer -> PCA or fixed orthonormal random map -> probe own training normalizer\","
      << "\"native_pipeline\":\"outer training normalizer -> probe own training normalizer; no compression\","
      << "\"compression_rank_policy\":\"both methods unsupported when width exceeds valid centered training-row bound or numerical rank\","
      << "\"random_map\":\"local std::mt19937_64 Gaussian; reduced QR; positive R diagonal; CPU float64\","
      << "\"actual_probe_seed\":\"stream_seed(repetition.probe_seed, probe_dimensions), shared across methods/architectures/checkpoints\","
      << "\"projection_pairing\":\"same declared projection seed and native/projected width across checkpoints/architectures\","
      << "\"ridge\":{\"penalty\":1,\"classes\":2,\"parameters\":\"2*input_dimensions+2\"},"
      << "\"tiny_secondary\":{\"activation\":\"tanh\",\"hidden\":16,\"updates\":100,\"optimizer\":\"Adam\",\"learning_rate\":0.01,"
      << "\"parameters\":\"16*input_dimensions+50\"},"
      << "\"primary_comparison\":\"random minus PCA at the same width and repetition; ridge\","
      << "\"secondary_comparison\":\"same paired population, fixed tanh16 readout\","
      << "\"validation_selection\":\"none; every declared repetition and width retained; no best-seed selection\","
      << "\"uncertainty\":\"paired source-group percentile bootstrap, within-checkpoint/readout validation diagnostic; no across-seed CI\","
      << "\"confidence\":0.95,\"bootstrap_replicates\":" << run.bootstrap_replicates << ","
      << "\"row_order\":\"feature rows must retain declared label/source archive order\","
      << "\"scope\":\"development compression diagnostic; no acceptance/performance claim\"}}";
  return out.str();
}
void save_outer(const fs::path &path, const FeatureNormalizer &outer) {
  torch::serialize::OutputArchive output;
  output.write("feature_mean", outer.mean, true); output.write("feature_scale", outer.scale, true);
  output.write("fitted_rows", torch::tensor(outer.fitted_rows), true);
  save_archive(path, output);
}
void save_pca(const fs::path &path, const TrainPca &pca) {
  torch::serialize::OutputArchive output;
  output.write("pca_mean", pca.mean, true); output.write("pca_components", pca.components, true);
  output.write("pca_singular_values", pca.singular_values, true);
  output.write("pca_numerical_rank", torch::tensor(pca.numerical_rank), true);
  output.write("fitted_rows", torch::tensor(pca.fitted_rows), true);
  save_archive(path, output);
}
void save_random(const fs::path &path, const FrozenOrthonormalProjection &projection) {
  torch::serialize::OutputArchive output;
  output.write("random_components", projection.components, true);
  output.write("projection_seed_decimal", archive::text_tensor(std::to_string(projection.seed)), true);
  output.write("fitted_rows", torch::tensor(int64_t(0)), true);
  save_archive(path, output);
}
struct Readout {
  RidgeProbe ridge;
  TinyProbe tiny;
  torch::Tensor training_ridge, training_tiny, validation_ridge, validation_tiny;
  uint64_t seed;
  Readout(const FeatureSurface &training, const FeatureSurface &validation,
          const torch::Tensor &labels, uint64_t actual_seed)
      : ridge(training, labels), tiny(training, labels, actual_seed), seed(actual_seed) {
    training_ridge = ridge.predict(training); training_tiny = tiny.predict(training);
    validation_ridge = ridge.predict(validation); validation_tiny = tiny.predict(validation);
  }
};
void save_readout(const fs::path &path, const Readout &readout, const FeatureNormalizer &outer) {
  torch::serialize::OutputArchive output;
  output.write("feature_mean", outer.mean, true); output.write("feature_scale", outer.scale, true);
  output.write("fitted_rows", torch::tensor(outer.fitted_rows), true);
  output.write("ridge_mean", readout.ridge.normalizer.mean, true); output.write("ridge_scale", readout.ridge.normalizer.scale, true);
  output.write("ridge_weights", readout.ridge.weights, true); output.write("ridge_intercept", readout.ridge.intercept, true);
  output.write("tiny_mean", readout.tiny.normalizer.mean, true); output.write("tiny_scale", readout.tiny.normalizer.scale, true);
  output.write("tiny_w1", readout.tiny.w1, true); output.write("tiny_b1", readout.tiny.b1, true);
  output.write("tiny_w2", readout.tiny.w2, true); output.write("tiny_b2", readout.tiny.b2, true);
  output.write("actual_probe_seed_decimal", archive::text_tensor(std::to_string(readout.seed)), true);
  save_archive(path, output);
}
void save_predictions(const fs::path &path, const torch::Tensor &ridge, const torch::Tensor &tiny,
                      const FeatureSurface &features, const Split &split) {
  torch::serialize::OutputArchive output;
  output.write("ridge", ridge, true); output.write("tiny_secondary", tiny, true);
  output.write("valid", features.valid, true); output.write("probe_features", features.values, true);
  output.write("labels_scoring_only", split.labels, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  save_archive(path, output);
}
std::string measured_json(const std::string &method, const fs::path &directory,
    const FeatureSurface &training, const FeatureSurface &validation, const Split &train_split,
    const Split &validation_split, const FeatureNormalizer &outer, uint64_t probe_seed,
    int64_t replicates, Readout &readout) {
  const auto width = training.values.size(1);
  save_readout(directory / (method + "-fit.pt"), readout, outer);
  save_predictions(directory / (method + "-training-predictions.pt"), readout.training_ridge, readout.training_tiny, training, train_split);
  save_predictions(directory / (method + "-validation-predictions.pt"), readout.validation_ridge, readout.validation_tiny, validation, validation_split);
  auto method_score = [&](const torch::Tensor &train, const torch::Tensor &valid, uint64_t stream) {
    return "{\"training\":" + score_json(score(train, train_split.labels, training.valid)) +
        ",\"validation\":" + score_json(score(valid, validation_split.labels, validation.valid)) +
        ",\"validation_interval\":" + interval_json(grouped_accuracy_interval(valid, validation_split.labels,
            validation.valid, validation_split.source_ids, stream_seed(probe_seed, stream), replicates), replicates) + '}';
  };
  return "{\"method\":" + quote(method) + ",\"status\":\"measured\",\"probe_dimensions\":" + std::to_string(width) +
      ",\"actual_probe_seed\":" + quote(std::to_string(probe_seed)) +
      ",\"ridge_parameters\":" + std::to_string(2 * width + 2) + ",\"tiny_parameters\":" + std::to_string(16 * width + 50) +
      ",\"fit_file\":" + quote(method + "-fit.pt") + ",\"training_predictions_file\":" + quote(method + "-training-predictions.pt") +
      ",\"validation_predictions_file\":" + quote(method + "-validation-predictions.pt") +
      ",\"ridge\":" + method_score(readout.training_ridge, readout.validation_ridge, 0x7269646765ULL) +
      ",\"tiny_secondary\":" + method_score(readout.training_tiny, readout.validation_tiny, 0x74696e79ULL) + '}';
}
std::string unsupported(const std::string &method, int64_t width, const std::string &status, const std::string &reason) {
  return "{\"method\":" + quote(method) + ",\"status\":" + quote(status) + ",\"reason\":" + quote(reason) +
      ",\"probe_dimensions\":" + std::to_string(width) + ",\"ridge\":null,\"tiny_secondary\":null}";
}
bool both_classes(const Split &split) {
  const auto selected = split.labels.masked_select(split.features.valid);
  return selected.eq(0).any().item<bool>() && selected.eq(1).any().item<bool>();
}
void validate_run(const ProjectionDiagnosticRun &run) {
  require(!run.output_directory.empty() && !fs::exists(run.output_directory), "new nonempty output directory required");
  require(!run.inputs.empty() && !run.widths.empty() && !run.repetitions.empty() &&
          run.threads > 0 && run.bootstrap_replicates >= 100, "empty/invalid diagnostic recipe");
  std::set<std::string> names;
  for (const auto &input : run.inputs) {
    require(safe_name(input.id) && safe_name(input.architecture) && names.insert(input.id).second &&
            input.checkpoint_steps >= 0, "input requires unique safe ID/architecture and nonnegative checkpoint budget");
    require(!input.feature_values_key.empty() && !input.feature_valid_key.empty() && !input.feature_provenance_key.empty() &&
            !input.labels_key.empty() && !input.source_ids_key.empty(), "empty declared archive key");
  }
  std::set<int64_t> widths;
  for (const auto width : run.widths) require(width > 0 && widths.insert(width).second, "compression widths must be unique positive values");
  names.clear();
  for (const auto &rep : run.repetitions) require(safe_name(rep.id) && names.insert(rep.id).second, "repetition requires unique safe ID");
}
} // namespace

FrozenOrthonormalProjection::FrozenOrthonormalProjection(int64_t native_dimensions, int64_t projected_dimensions, uint64_t value)
    : seed(value) {
  require(native_dimensions > 0 && projected_dimensions > 0 && projected_dimensions <= native_dimensions,
          "random projection width must be positive and at most native width");
  auto gaussian = torch::empty({native_dimensions, projected_dimensions}, torch::kFloat64);
  auto *data = gaussian.data_ptr<double>();
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int64_t i = 0; i < gaussian.numel(); ++i) data[i] = normal(rng);
  const auto qr = at::linalg_qr(gaussian, "reduced");
  const auto diagonal = std::get<1>(qr).diagonal();
  require(torch::isfinite(diagonal).all().item<bool>() && diagonal.abs().gt(0).all().item<bool>(),
          "random projection QR is singular/nonfinite");
  components = (std::get<0>(qr) * torch::where(diagonal.ge(0), torch::ones_like(diagonal), -torch::ones_like(diagonal)).unsqueeze(0)).contiguous();
  require(torch::isfinite(components).all().item<bool>() &&
          torch::allclose(components.transpose(0, 1).matmul(components), torch::eye(projected_dimensions, torch::kFloat64), 1e-10, 1e-10),
          "random projection columns are not finite orthonormal");
}
FeatureSurface FrozenOrthonormalProjection::transform(const FeatureSurface &surface) const {
  validate_features(surface);
  require(surface.values.size(1) == components.size(0), "random projection native width mismatch");
  const auto selected = torch::nonzero(surface.valid).reshape({-1});
  auto out = torch::zeros({surface.values.size(0), components.size(1)}, torch::kFloat64);
  out.index_copy_(0, selected, surface.values.detach().to(torch::kFloat64).index_select(0, selected).matmul(components));
  require(torch::isfinite(out).all().item<bool>(), "random projection transform overflow");
  return {out, surface.valid.clone(), surface.provenance + "; fixed orthonormal random projection seed=" + std::to_string(seed)};
}

void run_projection_diagnostic(const ProjectionDiagnosticRun &run) {
  validate_run(run);
  InputGuard guard;
  for (const auto &input : run.inputs) {
    guard.add(input.training_features); guard.add(input.validation_features);
    guard.add(input.training_labels); guard.add(input.validation_labels);
    for (const auto &training_path : {input.training_features, input.training_labels})
      for (const auto &validation_path : {input.validation_features, input.validation_labels})
        require(fs::canonical(training_path) != fs::canonical(validation_path), "same archive declared in training and validation roles");
  }
  const fs::path root = fs::absolute(run.output_directory);
  fs::create_directories(root);
  // Freeze all roles, widths, seeds, parameter counts and comparisons before
  // archive deserialization/fitting/scoring, including unsupported inputs.
  write_text(root / "projection-card.json", card_json(run));
  write_text(root / "input-manifest.json", guard.manifest());
  RngIsolation isolation;
  torch::set_num_threads(run.threads);
  try {
    std::ostringstream report;
    report << "{\"version\":1,\"protocol\":\"frozen-projection-diagnostic-v1\",\"stage\":\"development\","
        << "\"card_file\":\"projection-card.json\",\"input_manifest_file\":\"input-manifest.json\","
        << "\"uncertainty_scope\":\"validation source groups conditional on frozen checkpoint and fitted readout; no across-seed interval\","
        << "\"source_fingerprint\":" << quote(run.source_fingerprint) << ",\"inputs\":[";
    bool first_input = true;
    for (const auto &input : run.inputs) {
      const auto training = load_split(input, true), validation = load_split(input, false);
      require(training.features.values.size(1) == validation.features.values.size(1) &&
              training.features.values.scalar_type() == validation.features.values.scalar_type() &&
              training.features.provenance == validation.features.provenance, "feature dimension/dtype/provenance changed between training and validation");
      const std::set<std::string> train_sources(training.source_ids.begin(), training.source_ids.end());
      for (const auto &id : validation.source_ids) require(!train_sources.count(id), "training and validation source groups overlap");
      const fs::path directory = root / input.id;
      fs::create_directory(directory);
      const int64_t native_width = training.features.values.size(1);
      const bool fit_supported = training.features.valid.sum().item<int64_t>() >= 2 && both_classes(training);
      const std::string fit_reason = fit_supported ? "" : "fewer than two valid training rows or a training class is absent";
      std::unique_ptr<FeatureNormalizer> outer;
      FeatureSurface normalized_train, normalized_validation;
      int64_t rank = 0, rank_bound = std::min(native_width, training.features.valid.sum().item<int64_t>() - 1);
      std::map<int64_t, std::unique_ptr<TrainPca>> pcas;
      if (fit_supported) {
        outer = std::make_unique<FeatureNormalizer>(training.features);
        normalized_train = outer->transform(training.features); normalized_validation = outer->transform(validation.features);
        save_outer(directory / "outer-normalizer.pt", *outer);
        const auto x = normalized_train.values.index_select(0, torch::nonzero(normalized_train.valid).reshape({-1}));
        const auto spectrum = std::get<1>(at::linalg_svd(x - x.mean(0), false));
        require(torch::isfinite(spectrum).all().item<bool>(), "training rank spectrum is nonfinite");
        const double largest = spectrum.numel() ? spectrum[0].item<double>() : 0;
        rank = spectrum.gt(std::max(x.size(0), x.size(1)) * std::numeric_limits<double>::epsilon() * largest).sum().item<int64_t>();
        for (const auto width : run.widths)
          if (width <= rank_bound && width <= rank) pcas.emplace(width, std::make_unique<TrainPca>(normalized_train, width));
      }
      if (!first_input) report << ',';
      first_input = false;
      report << "{\"input\":" << input_json(input) << ",\"native_dimensions\":" << native_width
          << ",\"feature_provenance\":" << quote(training.features.provenance)
          << ",\"training_population\":" << population(training, training.features.valid)
          << ",\"validation_population\":" << population(validation, validation.features.valid)
          << ",\"valid_centered_training_row_bound\":" << std::max<int64_t>(0, rank_bound)
          << ",\"numerical_training_rank\":";
      if (fit_supported) report << rank; else report << "null";
      report << ",\"repetitions\":[";
      bool first_rep = true;
      for (const auto &rep : run.repetitions) {
        const fs::path rep_directory = directory / rep.id;
        fs::create_directory(rep_directory);
        if (!first_rep) report << ',';
        first_rep = false;
        report << "{\"id\":" << quote(rep.id) << ",\"projection_seed\":" << quote(std::to_string(rep.projection_seed))
            << ",\"artifact_directory\":" << quote(input.id + "/" + rep.id)
            << ",\"native\":";
        if (!fit_supported) report << unsupported("native", native_width, "unsupported_fit", fit_reason);
        else {
          const auto native_seed = stream_seed(rep.probe_seed, uint64_t(native_width));
          Readout native(normalized_train, normalized_validation, training.labels, native_seed);
          report << measured_json("native", rep_directory, normalized_train, normalized_validation, training, validation,
              *outer, native_seed, run.bootstrap_replicates, native);
        }
        report << ",\"widths\":[";
        bool first_width = true;
        for (const auto width : run.widths) {
          if (!first_width) report << ',';
          first_width = false;
          const fs::path width_directory = rep_directory / ("width-" + std::to_string(width));
          fs::create_directory(width_directory);
          report << "{\"width\":" << width << ",\"artifact_directory\":" << quote(input.id + "/" + rep.id + "/width-" + std::to_string(width));
          if (!fit_supported || !pcas.count(width)) {
            const std::string status = fit_supported ? "unsupported_compression" : "unsupported_fit";
            const std::string reason = fit_supported ? "width exceeds valid centered training-row bound or numerical training rank; both methods share this rank policy" : fit_reason;
            report << ",\"pca\":" << unsupported("pca", width, status, reason) << ",\"random\":" << unsupported("random", width, status, reason)
                << ",\"paired_random_minus_pca\":null}";
            continue;
          }
          const auto &pca = *pcas.at(width);
          FrozenOrthonormalProjection projection(native_width, width, rep.projection_seed);
          save_pca(width_directory / "pca-map.pt", pca);
          save_random(width_directory / "random-map.pt", projection);
          const auto pca_train = pca.transform(normalized_train), pca_validation = pca.transform(normalized_validation);
          const auto random_train = projection.transform(normalized_train), random_validation = projection.transform(normalized_validation);
          validate_features(pca_train); validate_features(pca_validation);
          validate_features(random_train); validate_features(random_validation);
          require(torch::equal(pca_validation.valid, random_validation.valid), "paired methods have unequal validation support");
          const auto actual_seed = stream_seed(rep.probe_seed, uint64_t(width));
          Readout pca_readout(pca_train, pca_validation, training.labels, actual_seed);
          Readout random_readout(random_train, random_validation, training.labels, actual_seed);
          report << ",\"pca_map_file\":\"pca-map.pt\",\"random_map_file\":\"random-map.pt\","
              << "\"pca\":" << measured_json("pca", width_directory, pca_train, pca_validation, training, validation,
                  *outer, actual_seed, run.bootstrap_replicates, pca_readout)
              << ",\"random\":" << measured_json("random", width_directory, random_train, random_validation, training, validation,
                  *outer, actual_seed, run.bootstrap_replicates, random_readout);
          const auto common = pca_validation.valid.logical_and(random_validation.valid);
          auto comparison = [&](const torch::Tensor &candidate, const torch::Tensor &comparator, uint64_t stream) {
            return "{\"candidate\":" + score_json(score(candidate, validation.labels, common)) +
                ",\"comparator\":" + score_json(score(comparator, validation.labels, common)) +
                ",\"difference_interval\":" + interval_json(grouped_accuracy_interval(candidate, validation.labels, common,
                    validation.source_ids, stream_seed(actual_seed, stream), run.bootstrap_replicates, comparator), run.bootstrap_replicates) + '}';
          };
          const auto paired = "{\"candidate\":\"random\",\"comparator\":\"pca\",\"common_validation_population\":" + population(validation, common) +
              ",\"ridge\":" + comparison(random_readout.validation_ridge, pca_readout.validation_ridge, 0x706169727269ULL) +
              ",\"tiny_secondary\":" + comparison(random_readout.validation_tiny, pca_readout.validation_tiny, 0x706169727469ULL) + '}';
          write_text(width_directory / "comparison.json", paired);
          report << ",\"comparison_file\":\"comparison.json\",\"paired_random_minus_pca\":" << paired << '}';
        }
        report << "]}";
      }
      report << "]}";
      std::cout << "projection diagnostic input=" << input.id << " native=" << native_width
                << " valid_training=" << training.features.valid.sum().item<int64_t>() << " rank=" << rank << '\n' << std::flush;
    }
    report << "],\"input_archives_preserved\":true}";
    guard.verify();
    write_text(root / "report.json", report.str());
  } catch (...) {
    guard.verify();
    throw;
  }
}
} // namespace embedding::evaluation
