// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_native_feature_adapter.h"
#include "embedding/shared/fixed_feature_readouts.h"
#include "frozen_role_guard.h"
#include <torch/cuda.h>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <sys/stat.h>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace {
namespace ev = embedding::evaluation;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace frozen = embedding::evaluation::frozen_inputs;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr std::array<uint64_t, 5> old_masters{9109, 10210, 11311, 12412, 13513};
constexpr std::array<uint64_t, 5> new_masters{14614, 15715, 16816, 17917, 19018};
const std::string protocol = "frozen-amplitude-transfer-v1";
const std::string card_sha = "901247488a9588e4b44dd0ae29169bb6c8d59c8af9e3f0b0a84d2568007ed52e";
const fs::path parent_relative = "output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b";
const std::string parent_source = "6463926ae71ee5e5547aa660d654afb02832fa843c978c41609cbd814266d78c";
const std::string inventory_sha = "a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90";
const embedding::input_shape_t shape{3, 32, 3, torch::kFloat64, torch::kCPU};
const std::vector<int64_t> channel_ids{0, 1, 2};
const std::string units = "unitless,unitless,unitless";
void require(bool value, const std::string &why) { frozen::require(value, why); }
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void synchronize() { torch::cuda::synchronize(); }
std::string number(double value) {
  require(std::isfinite(value) && value >= 0, "finite nonnegative cost required");
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
std::string plan() {
  return "{\"protocol\":" + frozen::quote(protocol) +
      ",\"old_master_seeds\":[9109,10210,11311,12412,13513],\"new_data_master_seeds\":[14614,15715,16816,17917,19018],"
      "\"display_tags\":[\"RPB-v4.alt-01\",\"RPB-v7.alt-01\"],\"retained_snapshots\":15,\"unique_parent_roles\":80,"
      "\"encoder_updates\":0,\"decoder_updates\":0,\"planned_pipelines\":90,\"planned_heads\":180,"
      "\"train_pairs\":128,\"validation_pairs\":64,\"test_pairs\":0,\"shape\":[3,32,3],\"native_width\":32,"
      "\"head_repetitions\":[2701,2802,2903],\"deletion_rate\":0.30,\"encoder_inference_device\":\"CUDA\","
      "\"old_head_reuse\":false,\"testing_accessed\":false,\"stress_accessed\":false,\"quality_generated\":false,"
      "\"selection\":false,\"promotion\":false,\"source_fingerprint\":" + frozen::quote(EVALUATION_SOURCE_ID) +
      ",\"human_card_sha256\":" + frozen::quote(card_sha) + ",\"parent_source_fingerprint\":" + frozen::quote(parent_source) +
      ",\"parent_inventory_sha256\":" + frozen::quote(inventory_sha) + "}";
}

// Restricted metadata/source-ID JSON parser. It never discovers payload paths.
struct Json {
  enum class Kind { null, boolean, number, string, array, object } kind{Kind::null};
  double number{0}; std::string text; std::vector<Json> array; std::map<std::string, Json> object;
  const Json &at(const std::string &key) const {
    require(kind == Kind::object && object.count(key), "missing metadata field: " + key); return object.at(key);
  }
  std::string string() const { require(kind == Kind::string, "metadata string required"); return text; }
  uint64_t integer() const {
    require(kind == Kind::number && std::isfinite(number) && number >= 0 && number <= 9007199254740991. &&
        std::floor(number) == number, "exact nonnegative metadata integer required"); return static_cast<uint64_t>(number);
  }
};
class Parser {
  const std::string &s; size_t p{0};
  void space() { while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p; }
  char take() { require(p < s.size(), "truncated metadata JSON"); return s[p++]; }
  std::string string() {
    require(take() == '"', "metadata string required"); std::string out;
    while (true) {
      const unsigned char c = take(); if (c == '"') return out; require(c >= 32, "metadata control character");
      if (c != '\\') { out += char(c); continue; }
      const char e = take();
      if (e == '"' || e == '\\' || e == '/') out += e;
      else if (e == 'n') out += '\n'; else if (e == 'r') out += '\r'; else if (e == 't') out += '\t';
      else if (e == 'b') out += '\b'; else if (e == 'f') out += '\f';
      else if (e == 'u') {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) { const char h = take(); value *= 16;
          if (h >= '0' && h <= '9') value += h - '0'; else if (h >= 'a' && h <= 'f') value += h - 'a' + 10;
          else if (h >= 'A' && h <= 'F') value += h - 'A' + 10; else require(false, "bad metadata Unicode escape"); }
        require(value < 128, "only ASCII escaped metadata is permitted"); out += char(value);
      } else require(false, "unknown metadata escape");
    }
  }
  Json value(int depth) {
    require(depth <= 32, "metadata nesting limit"); space(); require(p < s.size(), "empty metadata JSON"); Json out;
    if (s[p] == '"') { out.kind = Json::Kind::string; out.text = string(); return out; }
    if (s[p] == '[') {
      ++p; out.kind = Json::Kind::array; space(); if (p < s.size() && s[p] == ']') { ++p; return out; }
      while (true) { out.array.push_back(value(depth + 1)); space(); const char c = take();
        if (c == ']') return out;
        require(c == ',', "metadata array delimiter"); }
    }
    if (s[p] == '{') {
      ++p; out.kind = Json::Kind::object; space(); if (p < s.size() && s[p] == '}') { ++p; return out; }
      while (true) { space(); const auto key = string(); space(); require(take() == ':', "metadata object delimiter");
        require(out.object.emplace(key, value(depth + 1)).second, "duplicate metadata field"); space(); const char c = take();
        if (c == '}') return out;
        require(c == ',', "metadata object delimiter"); }
    }
    for (const auto &literal : {std::string("true"), std::string("false"), std::string("null")})
      if (s.compare(p, literal.size(), literal) == 0) { p += literal.size(); out.kind = literal == "null" ? Json::Kind::null : Json::Kind::boolean; return out; }
    const auto start = p; if (s[p] == '-') ++p; require(p < s.size(), "truncated metadata number");
    if (s[p] == '0') ++p; else { require(s[p] >= '1' && s[p] <= '9', "metadata number"); while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p; }
    if (p < s.size() && s[p] == '.') { ++p; const auto first = p; while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p; require(p > first, "metadata fraction"); }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) { ++p; if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
      const auto first = p; while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p; require(p > first, "metadata exponent"); }
    out.kind = Json::Kind::number; out.number = std::stod(s.substr(start, p - start)); require(std::isfinite(out.number), "nonfinite metadata number"); return out;
  }
public:
  explicit Parser(const std::string &text) : s(text) {}
  Json parse() { auto out = value(0); space(); require(p == s.size(), "trailing metadata JSON"); return out; }
};
std::vector<std::string> source_ids(const std::string &text) {
  const auto parsed = Parser(text).parse(); require(parsed.kind == Json::Kind::array, "source-ID array required");
  std::vector<std::string> out; for (const auto &value : parsed.array) {
    const auto id = value.string(); require(!id.empty(), "empty source ID"); out.push_back(id); }
  return out;
}
bool inside(const fs::path &path, const fs::path &root) {
  auto p = path.begin(); for (auto r = root.begin(); r != root.end(); ++r, ++p) if (p == path.end() || *p != *r) return false; return true;
}
fs::path absolute(const std::string &value) {
  const fs::path path(value); require(path.is_absolute() && path.lexically_normal() == path, "absolute normalized path required"); return path;
}
void metadata_path(const fs::path &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(path) && fs::canonical(path) == path, "direct regular metadata required");
  struct stat v {}; require(::stat(path.c_str(), &v) == 0 && S_ISREG(v.st_mode) && v.st_nlink == 1, "metadata hardlink rejected");
}
struct Instance {
  std::string id, tag, method, policy; uint64_t old_master{0}, new_master{0}; int64_t updates{0};
  fs::path controlled, checkpoint, audit, scaler, raw, native;
  std::vector<std::string> fields() const {
    return {id, tag, std::to_string(old_master), std::to_string(new_master), method, policy, std::to_string(updates),
      controlled.string(), checkpoint.string(), audit.string(), scaler.string(), raw.string(), native.string()};
  }
};
std::vector<Instance> expected_instances(const fs::path &root) {
  std::vector<Instance> out;
  for (size_t i = 0; i < old_masters.size(); ++i) for (const std::string method : {"untrained_native", "native_v4", "native_v7"}) {
    const auto base = root / parent_relative / "results" / ("seed-" + std::to_string(old_masters[i]) + "-lag_sign");
    Instance v; v.id = method + '-' + std::to_string(old_masters[i]); v.method = method; v.old_master = old_masters[i]; v.new_master = new_masters[i];
    v.tag = method == "untrained_native" ? "Untrained encoder" : method == "native_v4" ? "RPB-v4.alt-01" : "RPB-v7.alt-01";
    v.policy = method == "native_v7" ? "coordinate15_v7" : "ordinary_v4"; v.updates = method == "untrained_native" ? 0 : 512;
    v.controlled = base / "controlled-training.pt";
    v.checkpoint = base / (method == "native_v7" ? "v7" : "v4") / (v.updates ? "point-512" : "point-0") / "checkpoint.pt";
    v.audit = v.checkpoint.string() + ".audit.pt"; v.scaler = v.checkpoint.string() + ".scaler.pt"; v.raw = v.checkpoint.string() + ".training-raw.pt";
    v.native = base / "readouts" / method / "training-features.pt"; out.push_back(v);
  }
  return out;
}
std::vector<std::string> split(const std::string &line) {
  std::vector<std::string> out; size_t start = 0;
  while (true) { const auto end = line.find('\t', start); out.push_back(line.substr(start, end - start)); if (end == std::string::npos) return out; start = end + 1; }
}
std::map<std::string, fs::path> admit_tsv(const fs::path &path, const std::vector<Instance> &instances) {
  std::istringstream in(frozen::bytes(path)); std::string line;
  require(bool(std::getline(in, line)), "instances TSV required"); if (!line.empty() && line.back() == '\r') line.pop_back();
  require(line == "id\tdisplay_tag\told_master\tnew_master\tmethod\tpolicy\tupdates\tcontrolled_training\tcheckpoint\taudit\tscaler\ttraining_raw\tnative_training", "exact 13-column TSV header required");
  for (const auto &instance : instances) {
    require(bool(std::getline(in, line)), "all fifteen ordered instances required"); if (!line.empty() && line.back() == '\r') line.pop_back();
    require(split(line) == instance.fields(), "TSV row must equal exact ordered role template");
  }
  require(!std::getline(in, line) && in.eof(), "undeclared trailing TSV row");
  std::map<std::string, fs::path> roles;
  for (const auto &instance : instances) for (const auto &path : {instance.controlled, instance.checkpoint, instance.audit, instance.scaler, instance.raw, instance.native})
    roles.emplace(path.string(), absolute(path.string()));
  require(roles.size() == 80, "exact eighty unique parent roles required"); return roles;
}
struct InventoryEntry { uint64_t bytes; std::string sha; };
std::map<std::string, InventoryEntry> inventory_entries(const std::string &text, const fs::path &parent) {
  const auto json = Parser(text).parse(); const auto &files = json.at("files"); require(files.kind == Json::Kind::array, "inventory files array required");
  std::map<std::string, InventoryEntry> out;
  for (const auto &item : files.array) {
    const fs::path relative(item.at("path").string()); const auto path = parent / relative;
    require(!relative.empty() && !relative.is_absolute() && relative.lexically_normal() == relative && inside(path, parent), "inventory relative path escapes");
    const auto sha = item.at("sha256").string(); require(frozen::is_sha(sha) && out.emplace(path.string(), InventoryEntry{item.at("bytes").integer(), sha}).second, "duplicate/invalid inventory role");
  }
  require(out.size() == json.at("file_count").integer(), "inventory file count differs"); return out;
}
void whole_matrix(const std::map<std::string, fs::path> &roles, const fs::path &parent,
                  const std::map<std::string, InventoryEntry> &inventory) {
  require(roles.size() == 80, "closed eighty-role matrix required"); std::set<std::pair<dev_t, ino_t>> identities;
  for (const auto &[name, path] : roles) {
    require(name == path.string() && inside(path, parent) && inventory.count(name) && fs::is_regular_file(path) && !fs::is_symlink(path) &&
        fs::canonical(path) == path, "unadmitted/escaping/redirected parent role: " + name);
    struct stat v {}; require(::stat(path.c_str(), &v) == 0 && S_ISREG(v.st_mode) && v.st_nlink == 1 && v.st_size > 0 &&
        uint64_t(v.st_size) == inventory.at(name).bytes && identities.emplace(v.st_dev, v.st_ino).second, "size/inode/hardlink parent matrix differs");
  }
}
void checksum_matrix(const std::string &text, const std::map<std::string, fs::path> &roles,
                     const std::map<std::string, InventoryEntry> &inventory) {
  std::istringstream in(text); std::string line; std::set<std::string> seen;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    require(line.size() > 66 && line.substr(64, 2) == "  " && frozen::is_sha(line.substr(0, 64)), "exact SHA role syntax required");
    const auto key = line.substr(66); require(roles.count(key) && seen.insert(key).second && inventory.at(key).sha == line.substr(0, 64), "SHA roles must equal pinned inventory");
  }
  require(in.eof() && seen.size() == 80, "complete SHA role matrix required before first payload hash");
}
torch::Tensor get(torch::serialize::InputArchive &archive, const std::string &key) { torch::Tensor out; archive.read(key, out, true); return out; }
std::string get_text(torch::serialize::InputArchive &archive, const std::string &key) { return embedding::archive::tensor_text(get(archive, key)); }
void tensor(const torch::Tensor &x, torch::Dtype dtype, const std::vector<int64_t> &dimensions, const std::string &why) {
  require(x.defined() && x.device().is_cpu() && x.scalar_type() == dtype && x.sizes().vec() == dimensions, why + " dtype/shape");
}
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &why) {
  require(a.device().is_cpu() && b.device().is_cpu() && a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes(), why + " schema");
  const auto x = a.contiguous(), y = b.contiguous();
  require(!x.numel() || std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel() * x.element_size()) == 0, why + " bytes");
}
void save_new(const fs::path &path, torch::serialize::OutputArchive &archive) { require(!fs::exists(path), "new archive required"); archive.save_to(path.string()); }
void write_text(torch::serialize::OutputArchive &archive, const std::string &key, const std::string &text) { archive.write(key, embedding::archive::text_tensor(text), true); }
ev::ControlledDataset original_training(const fs::path &path, uint64_t master) {
  torch::serialize::InputArchive archive; archive.load_from(path.string(), torch::kCPU); ev::ControlledDataset out;
  out.observed = {get(archive, "observations"), get(archive, "feature_mask")}; out.clean = out.observed;
  out.labels = get(archive, "labels_scoring_only"); out.source_ids = source_ids(get_text(archive, "source_ids_json"));
  tensor(out.observed.data, torch::kFloat64, {256, 3, 32, 3}, "original observations"); tensor(out.observed.feature_mask, torch::kBool, {256, 3, 32, 3}, "original mask");
  tensor(out.labels, torch::kInt64, {256}, "original labels"); require(out.source_ids.size() == 256, "original source order length");
  require(torch::isfinite(out.observed.data).all().item<bool>() && out.observed.data.masked_select(out.observed.feature_mask.logical_not()).eq(0).all().item<bool>(), "original hidden storage/finite values");
  std::map<std::string, std::array<int64_t, 2>> counts;
  const auto prefix = "fresh-decoder-replication-v1/seed-" + std::to_string(master) + "/lag_sign/source-";
  for (int64_t row = 0; row < 256; ++row) {
    const auto label = out.labels[row].item<int64_t>(); require(label == 0 || label == 1, "original binary label");
    require(out.source_ids[row].rfind(prefix, 0) == 0, "original source namespace/seed"); ++counts[out.source_ids[row]][label];
  }
  require(counts.size() == 128, "original128 source groups required"); for (const auto &[id, count] : counts) { (void)id; require(count[0] == 1 && count[1] == 1, "original complete source pair"); }
  return out;
}
ev::FeatureSurface extract(const ev::FeatureProvider &provider, const embedding::Batch &batch) {
  const auto values = provider.extract(batch); require(values.size() == 1 && values.count("curve_global"), "sole exact served global surface");
  auto out = values.at("curve_global"); tensor(out.values, torch::kFloat32, {batch.data.size(0), 32}, "CUDA native32 evidence");
  tensor(out.valid, torch::kBool, {batch.data.size(0)}, "CUDA support evidence"); ev::validate_features(out);
  require(torch::isfinite(out.values).all().item<bool>(), "finite native evidence");
  return out;
}
ev::ProviderFitInput original_fit(const ev::ControlledDataset &training, uint64_t master) {
  return {training.observed, shape, master, training.source_ids, channel_ids, units, "fresh-decoder-replication-v1/lag_sign", 1, 31};
}
void save_observations(const fs::path &path, const ev::ControlledDataset &data, const torch::Tensor &erasure = {}) {
  torch::serialize::OutputArchive archive; archive.write("observations", data.observed.data, true); archive.write("feature_mask", data.observed.feature_mask, true);
  archive.write("labels_scoring_only", data.labels, true); write_text(archive, "source_ids_json", frozen::strings(data.source_ids));
  if (erasure.defined()) archive.write("requested_erasure", erasure, true);
  save_new(path, archive);
}
void sanitize(ev::ControlledDataset &data) {
  data.observed.data = torch::where(data.observed.feature_mask, data.observed.data, torch::zeros_like(data.observed.data)).detach().clone();
  data.observed.feature_mask = data.observed.feature_mask.clone(); data.clean = data.observed; for (auto &id : data.source_ids) id = protocol + '/' + id;
}
ev::FeatureSurface raw_surface(const ev::ObservationScaler &scaler, const embedding::Batch &batch) {
  const auto normalized = scaler.transform(batch);
  return {torch::cat({normalized.data.flatten(1), normalized.feature_mask.flatten(1).to(torch::kFloat64)}, 1), batch.feature_mask.flatten(1).any(1), "amplitude TRAIN ObservationScaler; legal values+visibility"};
}
ev::FeatureSurface mask_surface(const embedding::Batch &batch) {
  return {batch.feature_mask.flatten(1).to(torch::kFloat64), batch.feature_mask.flatten(1).any(1), "visibility metadata only; amplitude observations"};
}
std::vector<ev::FixedFeatureMethod> controls(const fs::path &directory, const ev::ControlledDataset &train,
    const ev::ControlledDataset &validation, const ev::ControlledDataset &deleted) {
  ev::ObservationScaler scaler(train.observed); torch::serialize::OutputArchive saved;
  saved.write("mean", scaler.mean, true); saved.write("scale", scaler.scale, true); saved.write("counts", scaler.counts, true); save_new(directory / "raw-scaler.pt", saved);
  auto a = raw_surface(scaler, train.observed), b = raw_surface(scaler, validation.observed), c = raw_surface(scaler, deleted.observed);
  std::vector<ev::FixedFeatureMethod> methods{{"raw", a, b, c, false, {}}};
  const auto zero = [](const ev::FeatureSurface &x) { return ev::FeatureSurface{torch::zeros({x.values.size(0), 32}, torch::kFloat64), x.valid.clone(), "unsupported raw TRAIN PCA32; no fit"}; };
  try {
    ev::FeatureNormalizer outer(a); const auto prepared = outer.transform(a); ev::TrainPca pca(prepared, 32);
    torch::serialize::OutputArchive archive; archive.write("feature_mean", outer.mean, true); archive.write("feature_scale", outer.scale, true);
    archive.write("fitted_rows", torch::tensor(outer.fitted_rows), true); archive.write("pca_mean", pca.mean, true); archive.write("pca_components", pca.components, true);
    archive.write("pca_singular_values", pca.singular_values, true); archive.write("pca_numerical_rank", torch::tensor(pca.numerical_rank), true);
    save_new(directory / "pca-preprocessing.pt", archive); frozen::write_new(directory / "pca-status.json", "{\"status\":\"measured\",\"width\":32,\"native_PCA\":false}\n");
    methods.push_back({"pca_only", pca.transform(prepared), pca.transform(outer.transform(b)), pca.transform(outer.transform(c)), true, {}});
  } catch (const std::exception &error) {
    const std::string why = error.what();
    require(why.find("PCA dimensions exceed numerical training rank") != std::string::npos || why.find("PCA dimensions exceed valid centered training-row bound") != std::string::npos ||
        why.find("feature fit requires two valid training rows") != std::string::npos, "unexpected raw PCA error: " + why);
    frozen::write_new(directory / "pca-status.json", "{\"status\":\"unsupported_fit\",\"width\":32,\"native_PCA\":false,\"reason\":" + frozen::quote(why) + "}\n");
    methods.push_back({"pca_only", zero(a), zero(b), zero(c), true, why});
  }
  methods.push_back({"mask_metadata", mask_surface(train.observed), mask_surface(validation.observed), mask_surface(deleted.observed), false, {}});
  return methods;
}
} // namespace

int main(int argc, char **argv) try {
  if (argc == 2 && std::string(argv[1]) == "--source-id") { std::cout << EVALUATION_SOURCE_ID << '\n'; return 0; }
  if (argc == 2 && std::string(argv[1]) == "--plan") { std::cout << plan() << '\n'; return 0; }
  std::map<std::string, std::string> options;
  const std::set<std::string> names{"--instances", "--checksums", "--input-root", "--output", "--admission-log", "--admission-sha256", "--card", "--card-sha256"};
  require(argc == 17, "exact eight measurement options required");
  for (int i = 1; i < argc; i += 2) require(names.count(argv[i]) && options.emplace(argv[i], argv[i + 1]).second, "unknown/duplicate option");
  require(options.size() == names.size() && frozen::is_sha(EVALUATION_SOURCE_ID) && options.at("--card-sha256") == card_sha, "compiled source and frozen card binding");
  const auto root = absolute(options.at("--input-root")), output = absolute(options.at("--output")); const auto parent = root / parent_relative;
  require(fs::is_directory(root) && fs::canonical(root) == root && !fs::is_symlink(root), "direct input root required");
  require(inside(output, root / "output/runs/rpb-frozen-amplitude-transfer") && !inside(output, parent) && !fs::exists(output) &&
      fs::is_directory(output.parent_path()) && fs::canonical(output.parent_path()) == output.parent_path(), "exclusive additive output root required");
  const auto tsv = absolute(options.at("--instances")), checksums = absolute(options.at("--checksums")), card = absolute(options.at("--card")), log = absolute(options.at("--admission-log"));
  const auto inventory_path = parent / "artifact-integrity.json";
  require(!inside(tsv, parent) && !inside(checksums, parent) && !inside(log, parent) &&
      card == output.parent_path() / "source/code/evaluation/cards/frozen_amplitude_transfer_v1.md", "captured card and metadata cannot alias old payload inputs");
  for (const auto &path : {tsv, checksums, card, log, inventory_path}) metadata_path(path);
  const auto card_bytes = frozen::bytes(card), log_bytes = frozen::bytes(log), tsv_bytes = frozen::bytes(tsv), checksum_bytes = frozen::bytes(checksums), inventory_bytes = frozen::bytes(inventory_path);
  require(frozen::sha256(card_bytes) == card_sha && frozen::is_sha(options.at("--admission-sha256")) && frozen::sha256(log_bytes) == options.at("--admission-sha256") &&
      log_bytes.find("Frozen native feature CUDA admission passed") != std::string::npos && log_bytes.find("Fresh decoder replication CUDA admission passed") != std::string::npos &&
      log_bytes.find("Fixed feature readout tests passed") != std::string::npos && log_bytes.find("Frozen role guard checks passed") != std::string::npos && log_bytes.find(EVALUATION_SOURCE_ID) != std::string::npos,
      "actual source-bound CUDA/shared fixture admission required before parent access");
  require(frozen::sha256(inventory_bytes) == inventory_sha, "pinned original inventory differs");
  const auto instances = expected_instances(root); const auto roles = admit_tsv(tsv, instances); const auto inventory = inventory_entries(inventory_bytes, parent);
  whole_matrix(roles, parent, inventory); checksum_matrix(checksum_bytes, roles, inventory); // Entire metadata matrix before first payload hash.
  const auto parent_hash_start = Clock::now(); const frozen::Guard guard(checksums, roles); const auto parent_hash_seconds = seconds(parent_hash_start);
  require(torch::cuda::is_available(), "CUDA mandatory; no CPU encoder fallback"); torch::set_num_threads(1);
  require(fs::create_directory(output) && fs::create_directory(output / "role-parity"), "exclusive new output directories required");
  frozen::write_new(output / "recipe-plan.json", plan() + "\n"); frozen::write_new(output / "input-manifest.json", guard.json() + "\n");
  frozen::write_new(output / "admission-binding.json", "{\"source_fingerprint\":" + frozen::quote(EVALUATION_SOURCE_ID) + ",\"human_card_sha256\":" + frozen::quote(card_sha) +
      ",\"admission_log_sha256\":" + frozen::quote(options.at("--admission-sha256")) + ",\"instances_sha256\":" + frozen::quote(frozen::sha256(tsv_bytes)) +
      ",\"checksums_sha256\":" + frozen::quote(frozen::sha256(checksum_bytes)) + ",\"parent_inventory_sha256\":" + frozen::quote(inventory_sha) + "}\n");
  std::vector<ev::FeatureProvider> providers; std::vector<std::string> parity_records;
  std::string common_core, common_producer; double factory_seconds = 0, parity_seconds = 0, parity_io_seconds = 0;
  for (size_t cohort = 0; cohort < old_masters.size(); ++cohort) {
    const auto original_io_start = Clock::now();
    const auto original = original_training(instances[cohort * 3].controlled, old_masters[cohort]); const auto fit = original_fit(original, old_masters[cohort]);
    torch::serialize::InputArchive initial_audit; initial_audit.load_from(instances[cohort * 3].audit.string(), torch::kCPU);
    const auto core = get_text(initial_audit, "core_writer_source_fingerprint"), producer = get_text(initial_audit, "training_producer_source_fingerprint");
    require(frozen::is_sha(core) && frozen::is_sha(producer), "original core/producer identity must be explicit SHA256");
    if (cohort == 0) { common_core = core; common_producer = producer; }
    require(core == common_core && producer == common_producer, "all original masters must share producer scopes");
    parity_io_seconds += seconds(original_io_start);
    for (size_t method = 0; method < 3; ++method) {
      const auto &instance = instances[cohort * 3 + method];
      rpb::FrozenNativeFeatureOptions opts; opts.parent_checkpoint_path = instance.checkpoint.string(); opts.expected_parent_updates = instance.updates;
      opts.parent_policy = instance.policy == "ordinary_v4" ? rpb::FrozenDecoderParentPolicy::ordinary_v4 : rpb::FrozenDecoderParentPolicy::coordinate15_v7;
      opts.expected_parent_core_source_fingerprint = core; opts.expected_training_producer_source_fingerprint = producer;
      synchronize(); const auto factory_start = Clock::now(); auto provider = rpb::make_frozen_native_feature_provider(opts, fit); synchronize(); const auto factory_cost = seconds(factory_start); factory_seconds += factory_cost;
      require(provider.surfaces.size() == 1 && provider.surfaces.count("curve_global") && provider.surfaces.at("curve_global").kind == ev::SurfaceKind::global, "sole native global provider contract");
      synchronize(); const auto parity_start = Clock::now(); const auto actual = extract(provider, original.observed); synchronize(); const auto parity_cost = seconds(parity_start); parity_seconds += parity_cost;
      const auto witness_io_start = Clock::now();
      torch::serialize::InputArchive old; old.load_from(instance.native.string(), torch::kCPU); const auto saved_features = get(old, "features"), saved_valid = get(old, "valid");
      tensor(saved_features, torch::kFloat32, {256, 32}, "saved native TRAIN"); tensor(saved_valid, torch::kBool, {256}, "saved native support");
      exact(actual.values, saved_features, "original TRAIN CUDA native"); exact(actual.valid, saved_valid, "original TRAIN CUDA support");
      exact(get(old, "labels_scoring_only"), original.labels, "original TRAIN label order"); require(source_ids(get_text(old, "source_ids_json")) == original.source_ids, "original TRAIN source order differs");
      const auto saved_provenance = get_text(old, "provenance");
      torch::serialize::OutputArchive evidence; evidence.write("cuda_features", actual.values, true); evidence.write("saved_features", saved_features, true);
      evidence.write("cuda_valid", actual.valid, true); evidence.write("saved_valid", saved_valid, true); evidence.write("labels_scoring_only", original.labels, true);
      write_text(evidence, "source_ids_json", frozen::strings(original.source_ids)); write_text(evidence, "saved_provenance", saved_provenance); write_text(evidence, "extractor_provenance", actual.provenance);
      write_text(evidence, "snapshot_audit_json", frozen::fields(provider.audit_fields)); evidence.write("exact_bytes", torch::tensor(true), true); evidence.write("cuda_inference", torch::tensor(true), true);
      save_new(output / "role-parity" / (instance.id + ".pt"), evidence);
      const auto audit_directory = output / "role-parity" / instance.id; require(fs::create_directory(audit_directory), "new provider audit directory"); require(bool(provider.save_assets), "provider audit callback required"); provider.save_assets(audit_directory.string());
      const auto record = "{\"id\":" + frozen::quote(instance.id) + ",\"display_tag\":" + frozen::quote(instance.tag) + ",\"method\":" + frozen::quote(instance.method) +
          ",\"old_master\":" + std::to_string(instance.old_master) + ",\"new_master\":" + std::to_string(instance.new_master) + ",\"updates\":" + std::to_string(instance.updates) +
          ",\"checkpoint\":" + frozen::quote(instance.checkpoint.string()) + ",\"checkpoint_sha256\":" + frozen::quote(guard.expected.at(instance.checkpoint.string())) +
          ",\"training_witness_sha256\":" + frozen::quote(guard.expected.at(instance.native.string())) + ",\"audit\":" + frozen::fields(provider.audit_fields) +
          ",\"rows\":256,\"width\":32,\"exact_native_bytes\":true,\"exact_support_bytes\":true,\"source_and_label_order_exact\":true,\"encoder_inference_device\":\"CUDA\","
          "\"quality_generated\":false,\"factory_CUDA_and_binding_wall_seconds\":" + number(factory_cost) + ",\"TRAIN_CUDA_inference_transfer_and_parent_verification_wall_seconds\":" + number(parity_cost) + "}";
      frozen::write_new(output / "role-parity" / (instance.id + ".json"), record + "\n"); parity_records.push_back(record); providers.push_back(std::move(provider));
      parity_io_seconds += seconds(witness_io_start);
    }
  }
  require(providers.size() == 15, "all15 parity gates before any amplitude generation");
  std::ostringstream parity; parity << "{\"protocol\":" << frozen::quote(protocol) << ",\"status\":\"passed\",\"parity_snapshots\":15,\"unique_parent_roles\":80,\"quality_generated\":false,\"records\":[";
  for (size_t i = 0; i < parity_records.size(); ++i) { if (i) parity << ','; parity << parity_records[i]; } parity << "]}";
  frozen::write_new(output / "parity-before-generation.json", parity.str() + "\n");
  std::ostringstream report; report << "{\"protocol\":" << frozen::quote(protocol) << ",\"source_fingerprint\":" << frozen::quote(EVALUATION_SOURCE_ID)
      << ",\"human_card_sha256\":" << frozen::quote(card_sha) << ",\"encoder_updates\":0,\"decoder_updates\":0,\"old_head_reuse\":false,\"cohorts\":[";
  double generation_seconds = 0, cuda_features_seconds = 0, baseline_seconds = 0, head_seconds = 0;
  for (size_t cohort = 0; cohort < new_masters.size(); ++cohort) {
    const auto master = new_masters[cohort]; const auto directory = output / ("seed-" + std::to_string(master) + "-amplitude"); require(fs::create_directory(directory), "new amplitude cohort directory");
    const auto generation_start = Clock::now(); auto data = ev::make_controlled_development_protocol(ev::Task::amplitude, shape, 128, 64, master, .1);
    require(!data.testing.observed.data.defined() && data.testing.source_ids.empty(), "TEST must not be generated"); sanitize(data.training); sanitize(data.validation);
    auto view = ev::make_coordinate_deletion_view(data.validation.observed, data.validation.source_ids, ev::Task::amplitude,
        ev::stream_seed(master, 0x616d702d76616c30ULL), .30, protocol + "/validation-coordinate-dropout");
    auto deleted = data.validation; deleted.observed = view.observations; deleted.clean = deleted.observed;
    save_observations(directory / "controlled-training.pt", data.training); save_observations(directory / "controlled-validation.pt", data.validation);
    save_observations(directory / "controlled-validation-deleted.pt", deleted, view.requested_erasure); const auto generation_cost = seconds(generation_start); generation_seconds += generation_cost;
    const auto baseline_start = Clock::now(); auto methods = controls(directory, data.training, data.validation, deleted); const auto baseline_cost = seconds(baseline_start); baseline_seconds += baseline_cost;
    synchronize(); const auto feature_start = Clock::now();
    for (size_t method = 0; method < 3; ++method) {
      const auto &provider = providers[cohort * 3 + method]; methods.push_back({instances[cohort * 3 + method].method,
        extract(provider, data.training.observed), extract(provider, data.validation.observed), extract(provider, deleted.observed), false, {}});
    }
    synchronize(); const auto feature_cost = seconds(feature_start); cuda_features_seconds += feature_cost;
    ev::FixedFeatureReadoutRun heads; heads.output_directory = (directory / "readouts").string(); heads.master_seed = master;
    heads.training_labels = data.training.labels; heads.validation_labels = data.validation.labels;
    heads.training_source_ids = data.training.source_ids; heads.validation_source_ids = data.validation.source_ids; heads.methods = std::move(methods);
    const auto heads_start = Clock::now(); const auto readouts = ev::run_fixed_feature_readouts(heads); const auto heads_cost = seconds(heads_start); head_seconds += heads_cost;
    const auto record = "{\"old_master\":" + std::to_string(old_masters[cohort]) + ",\"new_data_master\":" + std::to_string(master) +
        ",\"task\":\"amplitude\",\"training_rows\":256,\"validation_rows\":128,\"test_rows\":0,\"encoder_updates\":0,\"decoder_updates\":0,"
        "\"planned_pipelines\":18,\"planned_heads\":36,\"generation_and_observation_IO_wall_seconds\":" + number(generation_cost) +
        ",\"CPU_raw_scaler_PCA_and_asset_IO_wall_seconds\":" + number(baseline_cost) + ",\"CUDA_native_inference_transfer_and_parent_verification_wall_seconds\":" + number(feature_cost) +
        ",\"CPU_head_fit_prediction_bootstrap_and_asset_IO_wall_seconds\":" + number(heads_cost) + ",\"fixed_readouts\":" + readouts + "}";
    frozen::write_new(directory / "cohort.json", record + "\n"); if (cohort) report << ','; report << record;
    std::cout << "Frozen amplitude transfer data_master=" << master << " parent_master=" << old_masters[cohort] << " encoder_updates=0 decoder_updates=0\n" << std::flush;
  }
  report << "],\"costs\":{\"GPU_training_seconds\":0,\"factory_CUDA_and_binding_wall_seconds\":" << number(factory_seconds)
      << ",\"old_TRAIN_CUDA_inference_transfer_and_parent_verification_wall_seconds\":" << number(parity_seconds)
      << ",\"new_CUDA_native_inference_transfer_and_parent_verification_wall_seconds\":" << number(cuda_features_seconds)
      << ",\"parent_role_SHA256_wall_seconds\":" << number(parent_hash_seconds) << ",\"old_TRAIN_archive_parity_evidence_IO_wall_seconds\":" << number(parity_io_seconds)
      << ",\"generation_and_observation_IO_wall_seconds\":" << number(generation_seconds) << ",\"CPU_raw_scaler_PCA_and_asset_IO_wall_seconds\":" << number(baseline_seconds)
      << ",\"CPU_head_fit_prediction_bootstrap_and_asset_IO_wall_seconds\":" << number(head_seconds)
      << ",\"scope\":\"synchronized mixed whole-callback wall time; CUDA forwards plus CPU transfer and immutable-parent hash verification/IO; pure GPU compute time unmeasured; factory includes parent bindings and archive loads; CPU head time includes bootstrap and evidence IO\"},"
      "\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false}";
  whole_matrix(roles, parent, inventory); guard.verify();
  require(frozen::bytes(tsv) == tsv_bytes && frozen::bytes(checksums) == checksum_bytes && frozen::bytes(card) == card_bytes && frozen::bytes(log) == log_bytes &&
      frozen::bytes(inventory_path) == inventory_bytes, "declared metadata changed during transfer");
  frozen::write_new(output / "report.json", report.str() + "\n");
  frozen::write_new(output / "input-integrity-after.json", "{\"passed\":true,\"roles\":80,\"all_bytes_exact\":true,\"whole_matrix_before_first_hash\":true}\n");
  frozen::write_new(output / "complete.json", "{\"protocol\":" + frozen::quote(protocol) + ",\"status\":\"complete\",\"cohorts\":5,\"retained_snapshots\":15,"
      "\"original_TRAIN_parity_snapshots\":15,\"unique_parent_roles\":80,\"encoder_updates\":0,\"decoder_updates\":0,\"planned_pipelines\":90,\"planned_heads\":180,"
      "\"test_rows\":0,\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false}\n");
  std::cout << "Frozen amplitude transfer complete: five new cohorts, fifteen frozen CUDA snapshots, zero model updates.\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
