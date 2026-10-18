// SPDX-License-Identifier: MIT
#include "embedding/shared/saved_feature_reliability.h"
#include <ATen/Context.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
constexpr double near_zero = 1e-12;
constexpr std::array<double, 7> probabilities{0, .05, .25, .5, .75, .95, 1};
constexpr std::array<uint64_t, 3> repetitions{2701, 2802, 2903};
void require(bool ok, const std::string &why) {
  if (!ok) throw std::runtime_error("[saved feature reliability] " + why);
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
bool safe(const std::string &value) {
  return !value.empty() && value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
std::string bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary); require(bool(in), "cannot read declared metadata: " + path);
  std::ostringstream out; out << in.rdbuf(); require(!in.bad(), "metadata read failed"); return out.str();
}
void write_new(const fs::path &path, const std::string &value) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(fd >= 0, "refuse existing/unwritable output: " + path.string());
  size_t offset = 0;
  while (offset < value.size()) {
    const auto n = ::write(fd, value.data() + offset, value.size() - offset);
    if (n <= 0) { ::close(fd); throw std::runtime_error("output write failed"); }
    offset += static_cast<size_t>(n);
  }
  const int synced = ::fsync(fd), closed = ::close(fd);
  require(synced == 0 && closed == 0, "output fsync/close failed");
}
void save_new(const fs::path &path, torch::serialize::OutputArchive &value) {
  require(!fs::exists(path), "refuse existing archive"); archive::save_archive(path.string(), value);
}
void tensor(const torch::Tensor &x, torch::Dtype dtype, const std::vector<int64_t> &shape, const std::string &name) {
  require(x.defined() && x.device().is_cpu() && x.scalar_type() == dtype && x.sizes().vec() == shape,
          "CPU tensor dtype/shape: " + name);
}
void finite(const torch::Tensor &x, const std::string &name) {
  require(torch::isfinite(x).all().item<bool>(), "nonfinite/overflow: " + name);
}
void near(const torch::Tensor &actual, const torch::Tensor &expected, double atol, double rtol, const std::string &name) {
  require(actual.sizes() == expected.sizes(), "replay shape mismatch: " + name);
  finite(actual, name); finite(expected, name);
  require((actual - expected).abs().le(atol + rtol * expected.abs()).all().item<bool>(), "saved arithmetic mismatch: " + name);
}
torch::Tensor row_l2_norm(const torch::Tensor &x) {
  auto value = x.pow(2).sum(1).sqrt(); finite(value, "row/pair norm"); return value;
}
std::string distribution(const torch::Tensor &input) {
  const auto x = input.detach().to(torch::kFloat64).flatten().contiguous(); finite(x, "distribution");
  std::ostringstream out; out << std::setprecision(17) << "{\"count\":" << x.numel();
  if (!x.numel()) return out.str() + ",\"mean\":null,\"population_sd\":null,\"minimum\":null,\"maximum\":null,\"quantiles\":null}";
  std::vector<double> values(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
  std::sort(values.begin(), values.end());
  // Long-double accumulation avoids a false infinite summary of finite values.
  long double total = 0; for (const double value : values) total += value;
  const double mean = static_cast<double>(total / values.size()); require(std::isfinite(mean), "summary mean overflow");
  long double variance = 0;
  for (const double value : values) { const long double difference = static_cast<long double>(value) - mean; variance += difference * difference; }
  const double sd = static_cast<double>(std::sqrt(variance / values.size()));
  require(std::isfinite(sd), "summary population SD overflow");
  out << ",\"mean\":" << mean << ",\"population_sd\":" << sd << ",\"minimum\":" << values.front()
      << ",\"maximum\":" << values.back() << ",\"quantiles\":[";
  for (size_t i = 0; i < probabilities.size(); ++i) {
    if (i) out << ',';
    const double position = (values.size() - 1) * probabilities[i];
    const size_t lower = static_cast<size_t>(position), upper = std::min(lower + 1, values.size() - 1);
    const long double fraction = position - lower;
    const double q = static_cast<double>((1 - fraction) * values[lower] + fraction * values[upper]);
    require(std::isfinite(q), "quantile overflow"); out << q;
  }
  return out.str() + "]}";
}
SavedFeatureGeometry geometry_values(const torch::Tensor &values, const torch::Tensor &valid, const torch::Tensor &labels) {
  const auto rows = valid.nonzero().flatten(), selected = values.index_select(0, rows);
  SavedFeatureGeometry out;
  out.row_norm = torch::zeros({values.size(0)}, torch::kFloat64);
  out.centered_row_norm = torch::zeros_like(out.row_norm);
  if (!selected.size(0)) {
    out.coordinate_mean = torch::zeros({values.size(1)}, torch::kFloat64);
    out.population_sd = torch::zeros_like(out.coordinate_mean); return out;
  }
  out.coordinate_mean = selected.mean(0);
  out.population_sd = (selected - out.coordinate_mean).pow(2).mean(0).sqrt();
  finite(out.coordinate_mean, "coordinate mean"); finite(out.population_sd, "population SD");
  out.row_norm.index_copy_(0, rows, row_l2_norm(selected));
  out.centered_row_norm.index_copy_(0, rows, row_l2_norm(selected - out.coordinate_mean));
  const auto left = values.index_select(0, valid.logical_and(labels.eq(0)).nonzero().flatten());
  const auto right = values.index_select(0, valid.logical_and(labels.eq(1)).nonzero().flatten());
  if (left.size(0) && right.size(0)) out.pooled_label_mean_distance = row_l2_norm((right.mean(0) - left.mean(0)).unsqueeze(0)).item<double>();
  return out;
}
std::string geometry(const SavedFeatureGeometry &value, const torch::Tensor &valid, const torch::Tensor &labels) {
  const auto rows = valid.nonzero().flatten();
  const auto n = rows.numel();
  std::ostringstream out; out << "{\"valid_rows\":" << n << ",\"dimensions\":" << value.coordinate_mean.numel();
  if (!n) return out.str() + ",\"coordinate_mean\":null,\"population_sd\":null,\"row_norm\":null,\"centered_row_norm\":null,\"pooled_label_mean_distance\":null,\"finite_coordinates\":0,\"constant_coordinates\":null,\"near_zero_rows\":0,\"near_zero_centered_rows\":0}";
  const auto norms = value.row_norm.index_select(0, rows), centered = value.centered_row_norm.index_select(0, rows);
  out << std::setprecision(17) << ",\"coordinate_mean\":" << distribution(value.coordinate_mean) << ",\"population_sd\":" << distribution(value.population_sd)
      << ",\"row_norm\":" << distribution(norms) << ",\"centered_row_norm\":" << distribution(centered)
      << ",\"finite_coordinates\":" << value.coordinate_mean.numel() << ",\"constant_coordinates\":" << value.population_sd.le(near_zero).sum().item<int64_t>()
      << ",\"near_zero_rows\":" << norms.le(near_zero).sum().item<int64_t>()
      << ",\"near_zero_centered_rows\":" << centered.le(near_zero).sum().item<int64_t>() << ",\"pooled_label_mean_distance\":";
  if (labels.index_select(0, rows).eq(0).any().item<bool>() && labels.index_select(0, rows).eq(1).any().item<bool>()) out << value.pooled_label_mean_distance;
  else out << "null";
  return out.str() + '}';
}
std::string accuracy(const torch::Tensor &prediction, const torch::Tensor &truth, const torch::Tensor &valid) {
  const auto count = valid.sum().item<int64_t>(), correct = prediction.eq(truth).logical_and(valid).sum().item<int64_t>();
  std::ostringstream out; out << std::setprecision(17) << "{\"total\":" << truth.numel() << ",\"valid\":" << count
      << ",\"correct\":" << correct << ",\"coverage\":" << double(count) / truth.numel() << ",\"accuracy\":";
  if (count) out << double(correct) / count; else out << "null";
  return out.str() + '}';
}
torch::Tensor normalized(const torch::Tensor &x, const torch::Tensor &valid, const torch::Tensor &mean, const torch::Tensor &scale) {
  auto out = torch::zeros_like(x, torch::kFloat64); const auto rows = valid.nonzero().flatten();
  if (rows.numel()) out.index_copy_(0, rows, (x.to(torch::kFloat64).index_select(0, rows) - mean) / scale);
  finite(out, "normalized input"); return out;
}
void map_schema(const torch::Tensor &mean, const torch::Tensor &scale, int64_t width, const std::string &name) {
  tensor(mean, torch::kFloat64, {width}, name + " mean"); tensor(scale, torch::kFloat64, {width}, name + " scale");
  finite(mean, name); finite(scale, name); require(scale.gt(0).all().item<bool>(), "saved scale must be positive");
}
void map_statistics(const torch::Tensor &x, const torch::Tensor &valid, const torch::Tensor &mean, const torch::Tensor &scale) {
  const auto selected = x.index_select(0, valid.nonzero().flatten());
  require(selected.size(0) >= 2, "saved TRAIN fit needs two valid rows");
  const auto expected_mean = selected.mean(0), expected_scale = (selected - expected_mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
  near(mean, expected_mean, 2e-9, 2e-9, "TRAIN mean"); near(scale, expected_scale, 2e-9, 2e-9, "TRAIN population SD/floor");
}
// A small strict JSON codec for source IDs and the retained TRAIN trace only.
// No external package, executable model code or filename discovery is involved.
struct Json {
  enum class Kind { null, boolean, number, string, array, object } kind{Kind::null};
  bool boolean{false}; double number{0}; std::string text;
  std::vector<Json> array; std::map<std::string, Json> object;
  const Json &at(const std::string &key) const {
    require(kind == Kind::object && object.count(key), "missing JSON field: " + key); return object.at(key);
  }
  int64_t integer() const {
    require(kind == Kind::number && std::isfinite(number) && number >= 0 && number <= 9007199254740991. && std::floor(number) == number,
            "exact nonnegative JSON integer required"); return static_cast<int64_t>(number);
  }
};
class Parser {
  const std::string &s; size_t p{0};
  void space() { while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p; }
  char take() { require(p < s.size(), "truncated JSON"); return s[p++]; }
  static void utf8(std::string &out, uint32_t c) {
    if (c < 128) out += char(c);
    else if (c < 2048) { out += char(0xc0 | (c >> 6)); out += char(0x80 | (c & 63)); }
    else if (c < 65536) { out += char(0xe0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 63)); out += char(0x80 | (c & 63)); }
    else { out += char(0xf0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 63)); out += char(0x80 | ((c >> 6) & 63)); out += char(0x80 | (c & 63)); }
  }
  uint32_t hex4() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) { const char c = take(); v *= 16;
      if (c >= '0' && c <= '9') v += c - '0'; else if (c >= 'a' && c <= 'f') v += c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v += c - 'A' + 10; else require(false, "invalid Unicode escape"); }
    return v;
  }
  std::string string() {
    require(take() == '"', "JSON string required"); std::string out;
    while (true) {
      const unsigned char c = take(); if (c == '"') return out; require(c >= 32, "raw JSON control character");
      if (c != '\\') { out += char(c); continue; } const char e = take();
      if (e == '"' || e == '\\' || e == '/') out += e;
      else if (e == 'n') out += '\n'; else if (e == 'r') out += '\r'; else if (e == 't') out += '\t';
      else if (e == 'b') out += '\b'; else if (e == 'f') out += '\f';
      else if (e == 'u') { uint32_t cp = hex4();
        if (cp >= 0xd800 && cp <= 0xdbff) { require(take() == '\\' && take() == 'u', "missing low surrogate"); const auto low = hex4();
          require(low >= 0xdc00 && low <= 0xdfff, "bad low surrogate"); cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00; }
        else { require(cp < 0xdc00 || cp > 0xdfff, "unpaired surrogate"); }
        utf8(out, cp);
      } else require(false, "unknown JSON escape");
    }
  }
  Json value(int depth) {
    require(depth <= 32, "JSON nesting limit"); space(); require(p < s.size(), "empty JSON value"); Json out;
    if (s[p] == '"') { out.kind = Json::Kind::string; out.text = string(); return out; }
    if (s[p] == '[') { ++p; out.kind = Json::Kind::array; space();
      if (p < s.size() && s[p] == ']') { ++p; return out; }
      while (true) { out.array.push_back(value(depth + 1)); space(); const char c = take(); if (c == ']') return out; require(c == ',', "array delimiter"); }
    }
    if (s[p] == '{') { ++p; out.kind = Json::Kind::object; space();
      if (p < s.size() && s[p] == '}') { ++p; return out; }
      while (true) { space(); const auto key = string(); space(); require(take() == ':', "object delimiter");
        require(out.object.emplace(key, value(depth + 1)).second, "duplicate JSON field"); space(); const char c = take();
        if (c == '}') return out;
        require(c == ',', "object delimiter"); }
    }
    for (const auto &literal : {std::string("true"), std::string("false"), std::string("null")})
      if (s.compare(p, literal.size(), literal) == 0) { p += literal.size(); out.kind = literal == "null" ? Json::Kind::null : Json::Kind::boolean; out.boolean = literal == "true"; return out; }
    const auto begin = p; if (s[p] == '-') ++p; require(p < s.size(), "truncated JSON number");
    if (s[p] == '0') ++p; else { require(s[p] >= '1' && s[p] <= '9', "JSON number"); while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p; }
    if (p < s.size() && s[p] == '.') { ++p; const auto start = p; while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p; require(p > start, "fraction digits"); }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) { ++p; if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
      const auto start = p; while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p; require(p > start, "exponent digits"); }
    out.kind = Json::Kind::number; out.number = std::stod(s.substr(begin, p - begin)); require(std::isfinite(out.number), "nonfinite JSON number"); return out;
  }
public:
  explicit Parser(const std::string &text) : s(text) {}
  Json parse() { auto out = value(0); space(); require(p == s.size(), "trailing JSON"); return out; }
};
std::vector<std::string> source_ids(const std::string &text) {
  const auto parsed = Parser(text).parse(); require(parsed.kind == Json::Kind::array, "source array required");
  std::vector<std::string> out; for (const auto &value : parsed.array) {
    require(value.kind == Json::Kind::string && !value.text.empty(), "nonempty source string required"); out.push_back(value.text); }
  return out;
}
torch::Tensor get(torch::serialize::InputArchive &in, const std::string &key) { torch::Tensor x; in.read(key, x, true); return x; }
SavedFeatureFit load_fit(const std::string &path, const std::vector<std::string> &ids, int64_t width, uint64_t rep) {
  torch::serialize::InputArchive in; in.load_from(path, torch::kCPU); SavedFeatureFit fit;
  fit.feature_mean = get(in, "feature_mean"); fit.feature_scale = get(in, "feature_scale");
  fit.ridge_mean = get(in, "ridge_mean"); fit.ridge_scale = get(in, "ridge_scale"); fit.ridge_weights = get(in, "ridge_weights"); fit.ridge_intercept = get(in, "ridge_intercept");
  fit.tiny_mean = get(in, "tiny_mean"); fit.tiny_scale = get(in, "tiny_scale"); fit.tiny_w1 = get(in, "tiny_w1"); fit.tiny_b1 = get(in, "tiny_b1"); fit.tiny_w2 = get(in, "tiny_w2"); fit.tiny_b2 = get(in, "tiny_b2");
  require(source_ids(archive::tensor_text(get(in, "training_source_ids_json"))) == ids, "fit TRAIN source order changed");
  auto applied = get(in, "outer_normalizer_applied"); tensor(applied, torch::kBool, {}, "outer fit flag"); require(applied.item<bool>(), "native outer normalizer must exist");
  for (const auto &key : {"outer_fitted_rows", "fitted_rows"}) { auto count = get(in, key); tensor(count, torch::kInt64, {}, key); require(count.item<int64_t>() == int64_t(ids.size()), "TRAIN fitted-row count"); }
  auto penalty = get(in, "ridge_penalty"), lr = get(in, "tiny_learning_rate"); tensor(penalty, torch::kFloat64, {}, "ridge penalty"); tensor(lr, torch::kFloat64, {}, "tiny rate");
  require(penalty.item<double>() == 1 && lr.item<double>() == .01, "fixed head scalar recipe");
  for (const auto &[key, expected] : std::vector<std::pair<std::string, int64_t>>{{"tiny_hidden", 16}, {"tiny_steps", 100}}) {
    auto x = get(in, key); tensor(x, torch::kInt64, {}, key); require(x.item<int64_t>() == expected, "fixed head budget"); }
  tensor(fit.tiny_w1, torch::kFloat64, {width, 16}, "fixed hidden16 weights");
  uint64_t seed = rep + 0x9e3779b97f4a7c15ULL * uint64_t(width);
  seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL; seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL; seed ^= seed >> 31;
  require(archive::tensor_text(get(in, "actual_probe_seed_decimal")) == std::to_string(seed), "paired declared probe seed"); return fit;
}
SavedFeaturePrediction load_prediction(const std::string &path, const torch::Tensor &labels, const std::vector<std::string> &ids) {
  torch::serialize::InputArchive in; in.load_from(path, torch::kCPU);
  require(torch::equal(get(in, "labels_scoring_only"), labels) && source_ids(archive::tensor_text(get(in, "source_ids_json"))) == ids,
          "prediction TRAIN labels/source order changed");
  return {get(in, "valid"), get(in, "ridge"), get(in, "tiny_secondary"), get(in, "probe_input_features"),
          get(in, "ridge_logits"), get(in, "tiny_hidden_preactivation"), get(in, "tiny_logits")};
}
SavedTrainingTrace load_trace(const std::string &path, int64_t expected, int64_t batch, int64_t parameters) {
  const auto root = Parser(bytes(path)).parse(); SavedTrainingTrace trace;
  trace.attempted = root.at("attempted").integer(); trace.completed = root.at("completed").integer();
  trace.sampled_rows = root.at("sampled_rows").integer(); trace.batch_size = batch;
  require(trace.completed == expected, "declared saved encoder update budget");
  require(root.at("training_device").kind == Json::Kind::string && root.at("training_device").text.starts_with("cuda") &&
          root.at("parameter_count").integer() == parameters && parameters > 0 && root.at("parameter_count").integer() == root.at("cuda_parameter_count").integer(),
          "retained CUDA trace provenance");
  for (const auto &key : {"last_input_cuda", "last_loss_cuda", "finite_gradients", "weights_changed"})
    require(root.at(key).kind == Json::Kind::boolean && root.at(key).boolean, "retained actual CUDA trace flag: " + std::string(key));
  const auto &losses = root.at("losses"); require(losses.kind == Json::Kind::array, "loss trace array");
  for (const auto &row : losses.array) { require(row.kind == Json::Kind::array && row.array.size() == 5, "five-field loss trace row");
    require(row.array[3].kind == Json::Kind::number && row.array[4].kind == Json::Kind::number, "numeric loss/gradient trace");
    trace.rows.push_back({row.array[0].integer(), row.array[1].integer(), row.array[2].integer(), row.array[3].number, row.array[4].number}); }
  return trace;
}
struct Threads {
  int old{at::get_num_threads()}; Threads() { at::set_num_threads(1); }
  ~Threads() noexcept { try { at::set_num_threads(old); } catch (...) { std::terminate(); } }
};
void save_replay(const fs::path &path, const SavedFeatureAnalysis &input, const SavedFeatureReplay &r) {
  torch::serialize::OutputArchive out;
  out.write("features", input.features, true); out.write("valid", input.valid, true); out.write("labels", input.labels, true);
  out.write("source_ids_json", archive::text_tensor(strings(input.source_ids)), true);
  for (const auto &[key, value] : std::vector<std::pair<std::string, torch::Tensor>>{
      {"outer_input", r.outer_input}, {"ridge_input", r.ridge_input}, {"tiny_input", r.tiny_input},
      {"ridge_logits", r.ridge_logits}, {"tiny_hidden_preactivation", r.tiny_hidden_preactivation}, {"tiny_logits", r.tiny_logits},
      {"ridge_signed_margin", r.ridge_signed_margin}, {"tiny_signed_margin", r.tiny_signed_margin},
      {"pair_row0", r.pair_row0}, {"pair_row1", r.pair_row1}, {"pair_valid", r.pair_valid},
      {"served_pair_distance", r.served_pair_distance}, {"outer_pair_distance", r.outer_pair_distance}, {"ridge_pair_distance", r.ridge_pair_distance},
      {"served_pair_midpoint_norm", r.served_pair_midpoint_norm}, {"outer_pair_midpoint_norm", r.outer_pair_midpoint_norm}, {"ridge_pair_midpoint_norm", r.ridge_pair_midpoint_norm},
      {"ridge_pair_order_margin", r.ridge_pair_order_margin}, {"tiny_pair_order_margin", r.tiny_pair_order_margin}}) out.write(key, value, true);
  for (const auto &[prefix, value] : std::vector<std::pair<std::string, SavedFeatureGeometry>>{
      {"served", r.served_geometry}, {"outer", r.outer_geometry}, {"ridge", r.ridge_geometry}}) {
    out.write(prefix + "_coordinate_mean", value.coordinate_mean, true);
    out.write(prefix + "_population_sd", value.population_sd, true);
    out.write(prefix + "_row_norm", value.row_norm, true);
    out.write(prefix + "_centered_row_norm", value.centered_row_norm, true);
    out.write(prefix + "_pooled_label_mean_distance", torch::tensor(value.pooled_label_mean_distance, torch::kFloat64), true);
  }
  out.write("saved_ridge_classes", input.saved.ridge, true); out.write("saved_tiny_classes", input.saved.tiny, true);
  out.write("pair_source_ids_json", archive::text_tensor(strings(r.pair_source_ids)), true); save_new(path, out);
}
} // namespace

SavedFeatureReplay replay_saved_feature_analysis(const SavedFeatureAnalysis &input, double atol, double rtol) {
  require(std::isfinite(atol) && std::isfinite(rtol) && atol >= 0 && rtol >= 0, "finite nonnegative replay tolerances");
  torch::NoGradGuard guard;
  require(input.features.defined() && input.features.dim() == 2 && input.features.size(0) > 0 && input.features.size(1) > 0, "feature dimensions");
  const auto b = input.features.size(0), d = input.features.size(1);
  require(input.features.device().is_cpu() && (input.features.scalar_type() == torch::kFloat32 || input.features.scalar_type() == torch::kFloat64), "CPU saved feature dtype");
  tensor(input.valid, torch::kBool, {b}, "valid"); tensor(input.labels, torch::kInt64, {b}, "labels");
  require(input.labels.ge(0).logical_and(input.labels.le(1)).all().item<bool>() && input.source_ids.size() == size_t(b), "binary labels/source rows");
  const auto x = torch::where(input.valid.unsqueeze(1), input.features.to(torch::kFloat64), torch::zeros({b, d}, torch::kFloat64)); finite(x, "valid native features");
  const auto &fit = input.fit; map_schema(fit.feature_mean, fit.feature_scale, d, "outer");
  map_schema(fit.ridge_mean, fit.ridge_scale, d, "ridge"); map_schema(fit.tiny_mean, fit.tiny_scale, d, "tiny");
  tensor(fit.ridge_weights, torch::kFloat64, {d, 2}, "ridge weights"); tensor(fit.ridge_intercept, torch::kFloat64, {2}, "ridge intercept");
  require(fit.tiny_w1.defined() && fit.tiny_w1.dim() == 2 && fit.tiny_w1.size(0) == d && fit.tiny_w1.size(1) > 0, "tiny hidden dimensions");
  const auto hidden = fit.tiny_w1.size(1); tensor(fit.tiny_w1, torch::kFloat64, {d, hidden}, "tiny w1"); tensor(fit.tiny_b1, torch::kFloat64, {hidden}, "tiny b1");
  tensor(fit.tiny_w2, torch::kFloat64, {hidden, 2}, "tiny w2"); tensor(fit.tiny_b2, torch::kFloat64, {2}, "tiny b2");
  for (const auto &v : {fit.ridge_weights, fit.ridge_intercept, fit.tiny_w1, fit.tiny_b1, fit.tiny_w2, fit.tiny_b2}) finite(v, "saved head weight");
  SavedFeatureReplay out; out.outer_input = normalized(x, input.valid, fit.feature_mean, fit.feature_scale);
  out.ridge_input = normalized(out.outer_input, input.valid, fit.ridge_mean, fit.ridge_scale);
  out.tiny_input = normalized(out.outer_input, input.valid, fit.tiny_mean, fit.tiny_scale);
  out.ridge_logits = out.ridge_input.matmul(fit.ridge_weights) + fit.ridge_intercept;
  out.tiny_hidden_preactivation = out.tiny_input.matmul(fit.tiny_w1) + fit.tiny_b1;
  out.tiny_logits = torch::tanh(out.tiny_hidden_preactivation).matmul(fit.tiny_w2) + fit.tiny_b2;
  const auto &saved = input.saved;
  tensor(saved.valid, torch::kBool, {b}, "saved validity"); tensor(saved.ridge, torch::kInt64, {b}, "saved ridge classes"); tensor(saved.tiny, torch::kInt64, {b}, "saved tiny classes");
  tensor(saved.probe_input, torch::kFloat64, {b, d}, "saved outer input"); tensor(saved.ridge_logits, torch::kFloat64, {b, 2}, "saved ridge logits");
  tensor(saved.tiny_hidden_preactivation, torch::kFloat64, {b, hidden}, "saved tiny hidden"); tensor(saved.tiny_logits, torch::kFloat64, {b, 2}, "saved tiny logits");
  require(torch::equal(saved.valid, input.valid), "saved prediction support changed");
  near(out.outer_input, saved.probe_input, atol, rtol, "outer input"); near(out.ridge_logits, saved.ridge_logits, atol, rtol, "ridge logits");
  near(out.tiny_hidden_preactivation, saved.tiny_hidden_preactivation, atol, rtol, "tiny hidden"); near(out.tiny_logits, saved.tiny_logits, atol, rtol, "tiny logits");
  require(torch::equal(saved.ridge, saved.ridge_logits.argmax(1)) && torch::equal(saved.tiny, saved.tiny_logits.argmax(1)), "saved class differs from own saved-logit argmax");
  const auto sign = input.labels.to(torch::kFloat64) * 2 - 1;
  // Margins refer to saved numerical decisions; separately replayed logits are
  // preserved too, without allowing a last-bit tie to alter the saved class.
  const auto ridge_gap = saved.ridge_logits.select(1, 1) - saved.ridge_logits.select(1, 0);
  const auto tiny_gap = saved.tiny_logits.select(1, 1) - saved.tiny_logits.select(1, 0);
  out.ridge_signed_margin = sign * ridge_gap; out.tiny_signed_margin = sign * tiny_gap;
  std::map<std::string, std::array<int64_t, 2>> groups;
  for (int64_t row = 0; row < b; ++row) {
    const auto &id = input.source_ids.at(row); require(!id.empty(), "empty source ID");
    if (!groups.count(id)) groups[id] = {-1, -1};
    auto &position = groups.at(id)[input.labels[row].item<int64_t>()]; require(position == -1, "duplicate source label variant"); position = row;
  }
  std::vector<int64_t> row0, row1;
  for (const auto &[id, rows] : groups) { require(rows[0] >= 0 && rows[1] >= 0, "each source needs one label0 and one label1"); out.pair_source_ids.push_back(id); row0.push_back(rows[0]); row1.push_back(rows[1]); }
  out.pair_row0 = torch::tensor(row0, torch::kInt64); out.pair_row1 = torch::tensor(row1, torch::kInt64);
  out.pair_valid = input.valid.index_select(0, out.pair_row0).logical_and(input.valid.index_select(0, out.pair_row1));
  const auto difference = [&](const torch::Tensor &values) { return values.index_select(0, out.pair_row1) - values.index_select(0, out.pair_row0); };
  const auto midpoint = [&](const torch::Tensor &values) { return (values.index_select(0, out.pair_row1) + values.index_select(0, out.pair_row0)) * .5; };
  out.served_pair_distance = row_l2_norm(difference(x)); out.outer_pair_distance = row_l2_norm(difference(out.outer_input)); out.ridge_pair_distance = row_l2_norm(difference(out.ridge_input));
  out.served_pair_midpoint_norm = row_l2_norm(midpoint(x)); out.outer_pair_midpoint_norm = row_l2_norm(midpoint(out.outer_input)); out.ridge_pair_midpoint_norm = row_l2_norm(midpoint(out.ridge_input));
  out.served_geometry = geometry_values(x, input.valid, input.labels);
  out.outer_geometry = geometry_values(out.outer_input, input.valid, input.labels);
  out.ridge_geometry = geometry_values(out.ridge_input, input.valid, input.labels);
  out.ridge_pair_order_margin = difference(ridge_gap); out.tiny_pair_order_margin = difference(tiny_gap);
  finite(out.ridge_signed_margin, "ridge signed margin"); finite(out.tiny_signed_margin, "tiny signed margin");
  const auto rows = input.valid.nonzero().flatten(), pairs = out.pair_valid.nonzero().flatten();
  std::ostringstream json; json << "{\"TRAIN_rows\":" << b << ",\"valid_rows\":" << rows.numel() << ",\"source_groups\":" << groups.size()
      << ",\"valid_pair_groups\":" << pairs.numel() << ",\"served\":" << geometry(out.served_geometry, input.valid, input.labels)
      << ",\"outer_input\":" << geometry(out.outer_geometry, input.valid, input.labels) << ",\"ridge_input\":" << geometry(out.ridge_geometry, input.valid, input.labels)
      << ",\"ridge\":" << accuracy(saved.ridge, input.labels, input.valid) << ",\"tiny_secondary\":" << accuracy(saved.tiny, input.labels, input.valid)
      << ",\"ridge_signed_margin\":" << distribution(out.ridge_signed_margin.index_select(0, rows))
      << ",\"tiny_signed_margin\":" << distribution(out.tiny_signed_margin.index_select(0, rows))
      << ",\"pairs\":{\"ordering\":\"lexical-source-ID;label1-minus-label0\",\"served_distance\":" << distribution(out.served_pair_distance.index_select(0, pairs))
      << ",\"outer_distance\":" << distribution(out.outer_pair_distance.index_select(0, pairs)) << ",\"ridge_distance\":" << distribution(out.ridge_pair_distance.index_select(0, pairs))
      << ",\"served_midpoint_norm\":" << distribution(out.served_pair_midpoint_norm.index_select(0, pairs))
      << ",\"outer_midpoint_norm\":" << distribution(out.outer_pair_midpoint_norm.index_select(0, pairs))
      << ",\"ridge_midpoint_norm\":" << distribution(out.ridge_pair_midpoint_norm.index_select(0, pairs))
      << ",\"near_zero_served_pairs\":" << out.served_pair_distance.index_select(0, pairs).le(near_zero).sum().item<int64_t>()
      << ",\"near_zero_outer_pairs\":" << out.outer_pair_distance.index_select(0, pairs).le(near_zero).sum().item<int64_t>()
      << ",\"near_zero_ridge_pairs\":" << out.ridge_pair_distance.index_select(0, pairs).le(near_zero).sum().item<int64_t>()
      << ",\"ridge_order_margin\":" << distribution(out.ridge_pair_order_margin.index_select(0, pairs))
      << ",\"tiny_order_margin\":" << distribution(out.tiny_pair_order_margin.index_select(0, pairs)) << ",\"ridge_positive_order_fraction\":";
  if (pairs.numel()) json << std::setprecision(17) << double(out.ridge_pair_order_margin.index_select(0, pairs).gt(0).sum().item<int64_t>()) / pairs.numel(); else json << "null";
  json << ",\"ridge_tied_order_pairs\":" << out.ridge_pair_order_margin.index_select(0, pairs).eq(0).sum().item<int64_t>()
       << ",\"ridge_negative_order_pairs\":" << out.ridge_pair_order_margin.index_select(0, pairs).lt(0).sum().item<int64_t>() << "}}";
  out.json = json.str(); return out;
}

std::string saved_training_trace_json(const SavedTrainingTrace &trace, int64_t block) {
  require(trace.completed > 0 && trace.attempted == trace.completed && trace.batch_size > 0 &&
          trace.completed <= INT64_MAX / trace.batch_size && trace.sampled_rows == trace.completed * trace.batch_size &&
          trace.rows.size() == size_t(trace.completed) && block > 0 && trace.completed % block == 0, "complete unskipped trace counters/blocks");
  std::vector<double> losses, gradients;
  for (size_t i = 0; i < trace.rows.size(); ++i) { const auto &row = trace.rows[i];
    require(row.attempted == int64_t(i + 1) && row.completed == int64_t(i + 1) && row.target_cells > 0 &&
            std::isfinite(row.loss) && row.loss >= 0 && std::isfinite(row.gradient_norm) && row.gradient_norm >= 0, "absolute trace prefix/finite supported loss");
    losses.push_back(row.loss); gradients.push_back(row.gradient_norm);
  }
  std::ostringstream out; out << "{\"attempted\":" << trace.attempted << ",\"completed\":" << trace.completed << ",\"sampled_rows\":" << trace.sampled_rows << ",\"skips\":0,\"blocks\":[";
  for (int64_t first = 0; first < trace.completed; first += block) {
    if (first) out << ',';
    std::vector<double> loss(losses.begin() + first, losses.begin() + first + block), grad(gradients.begin() + first, gradients.begin() + first + block);
    int64_t cells = 0; for (int64_t i = first; i < first + block; ++i) { require(cells <= INT64_MAX - trace.rows[size_t(i)].target_cells, "target count overflow"); cells += trace.rows[size_t(i)].target_cells; }
    out << "{\"first_absolute_update\":" << first + 1 << ",\"last_absolute_update\":" << first + block << ",\"updates\":" << block << ",\"target_cells_sum\":" << cells
        << ",\"sampled_loss\":" << distribution(torch::tensor(loss, torch::kFloat64)) << ",\"gradient_norm\":" << distribution(torch::tensor(grad, torch::kFloat64)) << '}';
  }
  return out.str() + "],\"interpretation\":\"sampled optimization loss and scalar gradient norm;not fixed-query MAE or causal gradient alignment\"}";
}

std::string run_saved_feature_reliability(const SavedFeatureReliabilityRun &run) {
  require(!run.inputs.empty() && run.training_rows > 0 && run.training_rows % 2 == 0 && run.native_width > 0 &&
          run.expected_updates >= 4 && run.expected_updates % 4 == 0 && run.batch_size > 0 && run.expected_parameter_count > 0,
          "declared run geometry/budgets");
  require(run.shape.device.is_cpu() && run.shape.dtype == torch::kFloat64 && run.shape.channel_count > 0 && run.shape.history_length > 0 && run.shape.input_width > 0,
          "legal CPU float64 observation shape");
  std::set<std::string> ids; for (const auto &input : run.inputs) require(safe(input.id) && !input.display_tag.empty() && ids.insert(input.id).second, "safe unique input IDs");
  std::set<std::string> comparison_ids; for (const auto &pair : run.comparisons)
    require(safe(pair.id) && !ids.count(pair.id) && comparison_ids.insert(pair.id).second && ids.count(pair.candidate_id) && ids.count(pair.comparator_id) && pair.candidate_id != pair.comparator_id,
            "declared unique comparison/input IDs");
  const fs::path output(run.output_directory); require(!output.empty() && !fs::exists(output) && fs::create_directories(output), "new exclusive output directory");
  const Threads threads; torch::NoGradGuard guard;
  struct Result { uint64_t master; torch::Tensor observed, mask, labels; std::vector<std::string> sources; std::array<SavedFeatureReplay, 3> replay; };
  std::map<std::string, Result> results;
  std::ostringstream report; report << "{\"protocol\":\"saved-native-reliability-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"human_card_sha256\":" << quote(run.card_sha256) << ",\"recipe\":{\"absolute_tolerance\":2e-9,\"relative_tolerance\":2e-9,\"near_zero_threshold\":1e-12,\"quantile_probabilities\":[0,0.05,0.25,0.5,0.75,0.95,1],\"quantile_law\":\"linear-interpolation-(n-1)p\",\"bootstrap\":false},\"instances\":[";
  bool first = true;
  for (const auto &input : run.inputs) {
    torch::serialize::InputArchive observations, features; observations.load_from(input.controlled_training_path, torch::kCPU); features.load_from(input.native_training_path, torch::kCPU);
    const auto observed = get(observations, "observations"), mask = get(observations, "feature_mask"), labels = get(observations, "labels_scoring_only");
    const auto sources = source_ids(archive::tensor_text(get(observations, "source_ids_json")));
    tensor(observed, torch::kFloat64, {run.training_rows, run.shape.channel_count, run.shape.history_length, run.shape.input_width}, "legal observations");
    tensor(mask, torch::kBool, observed.sizes().vec(), "legal mask"); tensor(labels, torch::kInt64, {run.training_rows}, "TRAIN labels");
    finite(observed, "legal observed/hidden storage"); require(observed.masked_select(mask.logical_not()).eq(0).all().item<bool>(), "hidden storage must be zero");
    SavedFeatureAnalysis analysis; analysis.features = get(features, "features"); analysis.valid = get(features, "valid"); analysis.labels = labels; analysis.source_ids = sources;
    tensor(analysis.features, torch::kFloat32, {run.training_rows, run.native_width}, "served native features"); tensor(analysis.valid, torch::kBool, {run.training_rows}, "native support");
    require(analysis.valid.all().item<bool>() && mask.flatten(1).any(1).all().item<bool>(), "retained protocol requires full legal/native TRAIN support");
    require(torch::equal(get(features, "labels_scoring_only"), labels) && source_ids(archive::tensor_text(get(features, "source_ids_json"))) == sources, "native TRAIN source/label association");
    const auto provenance = archive::tensor_text(get(features, "provenance")); require(!provenance.empty(), "native provenance required");
    Result result{input.master_seed, observed, mask, labels, sources, {}};
    const auto directory = output / input.id; require(fs::create_directory(directory), "new instance directory");
    std::ostringstream instance; instance << "{\"id\":" << quote(input.id) << ",\"display_tag\":" << quote(input.display_tag) << ",\"master_seed\":" << input.master_seed
        << ",\"provenance\":" << quote(provenance) << ",\"repetitions\":[";
    SavedFeatureFit reference;
    for (size_t r = 0; r < repetitions.size(); ++r) {
      analysis.fit = load_fit(input.fit_paths[r], sources, run.native_width, repetitions[r]); analysis.saved = load_prediction(input.training_prediction_paths[r], labels, sources);
      result.replay[r] = replay_saved_feature_analysis(analysis);
      require(result.replay[r].pair_source_ids.size() == size_t(run.training_rows / 2) && result.replay[r].pair_valid.all().item<bool>(), "complete paired TRAIN sources");
      map_statistics(analysis.features.to(torch::kFloat64), analysis.valid, analysis.fit.feature_mean, analysis.fit.feature_scale);
      map_statistics(result.replay[r].outer_input, analysis.valid, analysis.fit.ridge_mean, analysis.fit.ridge_scale);
      map_statistics(result.replay[r].outer_input, analysis.valid, analysis.fit.tiny_mean, analysis.fit.tiny_scale);
      if (r) for (const auto &[a, b] : std::vector<std::pair<torch::Tensor, torch::Tensor>>{{reference.feature_mean, analysis.fit.feature_mean}, {reference.feature_scale, analysis.fit.feature_scale},
          {reference.ridge_mean, analysis.fit.ridge_mean}, {reference.ridge_scale, analysis.fit.ridge_scale}, {reference.ridge_weights, analysis.fit.ridge_weights}, {reference.ridge_intercept, analysis.fit.ridge_intercept}})
        require(torch::equal(a, b), "deterministic Ridge/outer map differs across declared repetitions");
      else reference = analysis.fit;
      const auto rep = directory / ("rep-" + std::to_string(repetitions[r])); require(fs::create_directory(rep), "new repetition directory");
      save_replay(rep / "analysis.pt", analysis, result.replay[r]); write_new(rep / "analysis.json", result.replay[r].json + "\n");
      if (r) instance << ',';
      instance << "{\"repetition\":" << repetitions[r] << ",\"saved_fit\":" << quote(input.fit_paths[r])
          << ",\"saved_TRAIN_predictions\":" << quote(input.training_prediction_paths[r]) << ",\"analysis\":" << result.replay[r].json << '}';
    }
    const auto trace = load_trace(input.encoder_progress_path, run.expected_updates, run.batch_size, run.expected_parameter_count);
    const auto trace_json = saved_training_trace_json(trace, run.expected_updates / 4);
    torch::serialize::OutputArchive trace_out; std::vector<int64_t> counters; std::vector<double> values;
    for (const auto &row : trace.rows) { counters.insert(counters.end(), {row.attempted, row.completed, row.target_cells}); values.insert(values.end(), {row.loss, row.gradient_norm}); }
    trace_out.write("counters", torch::tensor(counters, torch::kInt64).reshape({run.expected_updates, 3}), true);
    trace_out.write("values", torch::tensor(values, torch::kFloat64).reshape({run.expected_updates, 2}), true); save_new(directory / "trace.pt", trace_out);
    instance << "],\"trace\":" << trace_json << '}'; write_new(directory / "instance.json", instance.str() + "\n");
    if (!first) report << ',';
    first = false; report << instance.str(); results.emplace(input.id, std::move(result));
  }
  report << "],\"comparisons\":["; first = true;
  for (const auto &pair : run.comparisons) {
    const auto &a = results.at(pair.candidate_id), &b = results.at(pair.comparator_id);
    require(a.master == b.master && a.sources == b.sources && torch::equal(a.labels, b.labels) && torch::equal(a.observed, b.observed) && torch::equal(a.mask, b.mask), "paired methods must share exact legal TRAIN source/row/mask/labels");
    const auto directory = output / pair.id; require(fs::create_directory(directory), "new paired-comparison directory");
    std::ostringstream value; value << "{\"id\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.candidate_id) << ",\"comparator\":" << quote(pair.comparator_id) << ",\"repetitions\":[";
    for (size_t r = 0; r < repetitions.size(); ++r) {
      const auto &left = a.replay[r], &right = b.replay[r]; require(left.pair_source_ids == right.pair_source_ids, "paired lexical source ordering");
      const auto common = left.pair_valid.logical_and(right.pair_valid), selected = common.nonzero().flatten();
      const auto served = left.served_pair_distance - right.served_pair_distance, outer = left.outer_pair_distance - right.outer_pair_distance,
          ridge_distance = left.ridge_pair_distance - right.ridge_pair_distance, ridge_margin = left.ridge_pair_order_margin - right.ridge_pair_order_margin,
          tiny_margin = left.tiny_pair_order_margin - right.tiny_pair_order_margin;
      for (const auto &x : {served, outer, ridge_distance, ridge_margin, tiny_margin}) finite(x, "paired difference");
      torch::serialize::OutputArchive out; out.write("source_ids_json", archive::text_tensor(strings(left.pair_source_ids)), true); out.write("common_pair_valid", common, true);
      out.write("served_distance_delta", served, true); out.write("outer_distance_delta", outer, true); out.write("ridge_distance_delta", ridge_distance, true);
      out.write("ridge_order_margin_delta", ridge_margin, true); out.write("tiny_order_margin_delta", tiny_margin, true);
      save_new(directory / ("rep-" + std::to_string(repetitions[r]) + ".pt"), out);
      if (r) value << ',';
      value << "{\"repetition\":" << repetitions[r] << ",\"common_source_groups\":" << selected.numel()
          << ",\"served_distance_delta\":" << distribution(served.index_select(0, selected)) << ",\"outer_distance_delta\":" << distribution(outer.index_select(0, selected))
          << ",\"ridge_distance_delta\":" << distribution(ridge_distance.index_select(0, selected)) << ",\"ridge_order_margin_delta\":" << distribution(ridge_margin.index_select(0, selected))
          << ",\"tiny_order_margin_delta\":" << distribution(tiny_margin.index_select(0, selected)) << '}';
    }
    value << "]}"; write_new(directory / "comparison.json", value.str() + "\n"); if (!first) report << ','; first = false; report << value.str();
  }
  report << "],\"encoder_updates\":0,\"decoder_updates\":0,\"head_refits\":0,\"PCA_fits\":0,\"encoder_forwards\":0,\"heldout_analysis_payloads\":0,\"selection\":false,\"promotion\":false,";
  report << "\"limits\":[\"saved TRAIN descriptive arithmetic;not new fitted quality\",\"source-paired Euclidean distances depend on representation coordinates and scale\",\"low margins or sampled training traces do not establish collapse,memory,overfit or causality\"]}";
  write_new(output / "report.json", report.str() + "\n"); return report.str();
}
} // namespace embedding::evaluation
