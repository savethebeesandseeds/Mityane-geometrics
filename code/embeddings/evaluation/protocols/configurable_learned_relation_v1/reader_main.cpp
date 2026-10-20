// SPDX-License-Identifier: MIT
// Read only saved CPU features. This translation unit has no encoder includes,
// construction, forwards, optimization, PCA, or raw-data baseline fitting.
#include "embedding/shared/fixed_feature_readouts.h"
#include "frozen_role_guard.h"
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sys/stat.h>

#ifndef CONFIGURABLE_LEARNED_RELATION_SOURCE_ID
#define CONFIGURABLE_LEARNED_RELATION_SOURCE_ID "unrecorded"
#endif
#ifndef CONFIGURABLE_LEARNED_RELATION_CARD_SHA256
#define CONFIGURABLE_LEARNED_RELATION_CARD_SHA256 "unrecorded"
#endif
namespace {
namespace ev = embedding::evaluation;
namespace fi = ev::frozen_inputs;
namespace fs = std::filesystem;
using fi::require; using fi::quote; using fi::bytes; using fi::sha256; using fi::write_new;
constexpr const char *protocol = "configurable-learned-relation-v1";
constexpr const char *original_source = "8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1";
constexpr const char *original_card = "562e685c6e1283a7bc22bad7ddcf9fb138c75221cb3dbbd8cd576f8edcfe144e";
constexpr const char *original_config = "c136f42f4cf89e50d00b17f6d9ad13bbbe39c2f59ba71dc180eb1c63ef077c8b";
constexpr const char *original_inputs = "6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091";
constexpr const char *original_inventory = "62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83";
constexpr int64_t masters[]{941101, 941102, 941103};
constexpr int64_t repetitions[]{2701, 2802, 2903};
constexpr const char *methods[]{"native_512", "learned_20_512"};
constexpr const char *views[]{"training", "validation-intact", "validation-deleted"};

torch::Tensor read(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor t; a.read(key, t, true); return t;
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  return embedding::archive::tensor_text(read(a, key));
}
void direct(const fs::path &p, bool file = false) {
  require(p.is_absolute() && fs::canonical(p) == p && !fs::is_symlink(p), "direct absolute input: " + p.string());
  if (file) { struct stat s{}; require(::stat(p.c_str(), &s) == 0 && S_ISREG(s.st_mode) && s.st_nlink == 1, "single-link regular input"); }
}
void new_leaf(const fs::path &p) {
  require(p.is_absolute() && p.lexically_normal() == p && !p.filename().empty() && !fs::exists(p) && !fs::is_symlink(p), "new absolute output leaf");
  direct(p.parent_path()); require(fs::create_directory(p), "reserve output");
}
std::string string_field(const std::string &json, const std::string &key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"\\\\]*)\"");
  std::sregex_iterator i(json.begin(), json.end(), pattern), end;
  require(i != end, "missing plain JSON field " + key); const auto value = (*i)[1].str();
  require(++i == end, "duplicate JSON field " + key); return value;
}
int64_t integer_field(const std::string &json, const std::string &key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)(?=\\s*[,}])");
  std::sregex_iterator i(json.begin(), json.end(), pattern), end;
  require(i != end, "missing JSON integer " + key); const auto value = std::stoll((*i)[1].str());
  require(++i == end, "duplicate JSON integer " + key); return value;
}
std::vector<std::string> source_strings(const std::string &s) {
  require(s.size() >= 2 && s.front() == '[' && s.back() == ']', "source ID JSON array");
  std::vector<std::string> out; size_t i = 1;
  while (i < s.size() - 1) {
    require(s[i] == '"', "source ID quote"); const auto end = s.find('"', i + 1);
    require(end != std::string::npos, "source ID closing quote"); const auto value = s.substr(i + 1, end - i - 1);
    require(!value.empty() && value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/") == std::string::npos, "closed source ID string");
    out.push_back(value); i = end + 1;
    if (i < s.size() - 1) { require(s[i++] == ',', "source ID separator"); require(i < s.size() - 1, "no trailing separator"); }
  }
  return out;
}
struct BoundInputs {
  fs::path root;
  std::map<std::string, std::string> inventory, used;
  explicit BoundInputs(const fs::path &capsule) : root(capsule) {
    direct(root); direct(root / "artifacts.sha256", true);
    require(sha256(bytes(root / "artifacts.sha256")) == original_inventory, "original completed artifact inventory");
    std::istringstream in(bytes(root / "artifacts.sha256")); std::string line;
    while (std::getline(in, line)) {
      require(line.size() > 66 && fi::is_sha(line.substr(0, 64)) && line.substr(64, 2) == "  ", "closed original inventory line");
      const auto role = line.substr(66); const fs::path p(role);
      require(!p.is_absolute() && p.lexically_normal() == p && role.find("..") == std::string::npos && inventory.emplace(role, line.substr(0, 64)).second, "safe unique original role");
    }
    require(inventory.size() == 815, "original closed artifact count");
    const auto complete = bytes(root / "complete.json"); direct(root / "complete.json", true);
    require(string_field(complete, "protocol") == "configurable-capacity-v1" && string_field(complete, "status") == "complete" &&
        string_field(complete, "artifacts_manifest_sha256") == original_inventory && integer_field(complete, "encoder_trajectories") == 9 &&
        integer_field(complete, "completed_updates_per_run") == 512 && integer_field(complete, "saved_head_fits") == 81, "original completion receipt");
    const auto receipt = bytes(bind("freeze/receipt.json"));
    require(string_field(receipt, "source_id") == original_source && string_field(receipt, "card_sha256") == original_card &&
        string_field(receipt, "config_sha256") == original_config, "original frozen source/card/settings receipt");
    require(sha256(bytes(bind("freeze/inputs.sha256"))) == original_inputs, "original quality input manifest");
  }
  fs::path bind(const std::string &role) {
    require(inventory.count(role), "declared original input role: " + role);
    const auto p = root / role; direct(p, true);
    require(sha256(bytes(p)) == inventory.at(role), "original saved input SHA: " + role);
    used.emplace(role, inventory.at(role)); return p;
  }
  void verify() const {
    require(sha256(bytes(root / "artifacts.sha256")) == original_inventory, "original inventory changed");
    for (const auto &[role, digest] : used) { direct(root / role, true); require(sha256(bytes(root / role)) == digest, "original input changed: " + role); }
  }
  std::string json() const {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto &[role, digest] : used) { if (!first) out << ','; first = false;
      out << "{\"role\":" << quote(role) << ",\"path\":" << quote((root / role).string()) << ",\"sha256\":" << quote(digest) << '}'; }
    return out.str() + ']';
  }
};
struct Split { torch::Tensor labels, source_ids, mask; std::vector<std::string> ids; };
struct Cohort { Split training, validation, deleted; };
std::vector<int64_t> expected_source_ids(int64_t master, bool training) {
  // Match only the membership assignment in the original frozen benchmark.
  // It shuffles 384 sources before assigning256 to TRAIN; each role is then
  // emitted in original-source order. No signal or observation is regenerated.
  constexpr uint64_t task_stream = 0x74656d706f352d31ULL, split_stream = 0x6361702d73706c31ULL;
  std::mt19937_64 rng(ev::stream_seed(ev::stream_seed(master, task_stream), split_stream));
  std::vector<uint64_t> assignment(384); std::iota(assignment.begin(), assignment.end(), uint64_t(0));
  for (uint64_t i = 383; i > 0; --i) {
    const auto limit = i + 1, threshold = (uint64_t(0) - limit) % limit;
    uint64_t draw; do { draw = rng(); } while (draw < threshold);
    std::swap(assignment[i], assignment[draw % limit]);
  }
  std::vector<bool> in_training(384, false); for (size_t i = 0; i < 256; ++i) in_training[assignment[i]] = true;
  std::vector<int64_t> ids;
  for (uint64_t source = 0; source < 384; ++source) if (in_training[source] == training)
    ids.push_back(static_cast<int64_t>((uint64_t(master) << 31) | (source << 1)));
  require(ids.size() == size_t(training ? 256 : 128), "exact original shuffled role population"); return ids;
}
void check_role_source(int64_t id, int64_t master, size_t pair, const std::vector<int64_t> &expected) {
  require(id >= 0 && !(id & 1) && (uint64_t(id) >> 31) == uint64_t(master) && pair < expected.size() && id == expected[pair],
      "packed source cohort/exact shuffled ordered role");
}
Split load_split(const fs::path &p, int64_t master, int64_t rows, bool training) {
  torch::serialize::InputArchive a; a.load_from(p.string(), torch::kCPU);
  const auto keys = a.keys(); require(std::set<std::string>(keys.begin(), keys.end()) ==
      std::set<std::string>{"observed", "feature_mask", "labels_scoring_only", "source_ids", "source_ids_json"}, "exact controlled input keys");
  Split s{read(a, "labels_scoring_only"), read(a, "source_ids"), read(a, "feature_mask"), source_strings(text(a, "source_ids_json"))};
  const auto observed = read(a, "observed"); const std::vector<int64_t> shape{rows, 3, 128, 9};
  require(observed.device().is_cpu() && observed.scalar_type() == torch::kFloat64 && observed.sizes() == torch::IntArrayRef(shape) &&
      s.mask.device().is_cpu() && s.mask.scalar_type() == torch::kBool && s.mask.sizes() == observed.sizes(), "legal CPU64 values/Boolean mask geometry");
  require(torch::isfinite(observed).all().item<bool>() && observed.masked_select(~s.mask).eq(0).all().item<bool>(), "finite observation storage and zero masked cells");
  require(s.labels.scalar_type() == torch::kInt64 && s.labels.sizes() == torch::IntArrayRef({rows}) && s.source_ids.scalar_type() == torch::kInt64 &&
      s.source_ids.sizes() == s.labels.sizes() && s.ids.size() == size_t(rows), "label/source row schema");
  std::set<int64_t> groups; const auto expected = expected_source_ids(master, training);
  require(expected.size() * 2 == size_t(rows), "exact role rows");
  for (int64_t row = 0; row < rows; row += 2) {
    const auto id = s.source_ids[row].item<int64_t>(); check_role_source(id, master, size_t(row / 2), expected);
    require(s.source_ids[row + 1].item<int64_t>() == id && s.ids[row] == "TEMPO-5/" + std::to_string(id) && s.ids[row + 1] == s.ids[row] &&
        groups.insert(id).second && s.labels[row].item<int64_t>() >= 0 && s.labels[row].item<int64_t>() <= 1 &&
        s.labels[row + 1].item<int64_t>() == 1 - s.labels[row].item<int64_t>() && torch::equal(s.mask[row], s.mask[row + 1]), "complete unique source pairs/opposite classes/independent masks");
  }
  return s;
}
Cohort load_cohort(BoundInputs &inputs, int64_t master) {
  const auto pre = "quality-data/master-" + std::to_string(master) + '/';
  Cohort c{load_split(inputs.bind(pre + "training.pt"), master, 512, true), load_split(inputs.bind(pre + "validation.pt"), master, 256, false),
      load_split(inputs.bind(pre + "validation-deleted.pt"), master, 256, false)};
  require(c.validation.ids == c.deleted.ids && torch::equal(c.validation.labels, c.deleted.labels) && torch::equal(c.validation.source_ids, c.deleted.source_ids), "deleted role association");
  require(!c.deleted.mask.logical_and(~c.validation.mask).any().item<bool>(), "deleted support only removes observations");
  return c;
}
struct NativeViews { ev::FeatureSurface training, validation, deleted; std::string scaler_identity; };
NativeViews load_native(BoundInputs &inputs, int64_t master, int64_t point, const Cohort &c) {
  const auto role = "profiles/A/master-" + std::to_string(master) + "/point-" + std::to_string(point) + "/features.pt";
  torch::serialize::InputArchive a; a.load_from(inputs.bind(role).string(), torch::kCPU);
  require(text(a, "protocol") == "configurable-capacity-v1" && text(a, "artifact_kind") == "configurable_capacity_native_feature_views_v1" &&
      text(a, "dataset_id") == "TEMPO-5" && text(a, "profile") == "A" && text(a, "model_tag") == "RPB-v20.alt-01" &&
      read(a, "master").item<int64_t>() == master && read(a, "point").item<int64_t>() == point && read(a, "native_width").item<int64_t>() == 32,
      "profile A original native geometry/cohort/point");
  require(text(a, "source_fingerprint") == original_source && text(a, "card_sha") == original_card && text(a, "input_manifest_sha") == original_inputs &&
      text(a, "config_sha") == original_config, "original saved-feature source/card/input/config binding");
  const auto surface = [&](const std::string &prefix, const Split &s) {
    ev::FeatureSurface f{read(a, prefix + "_values"), read(a, prefix + "_valid"), role + "; saved original CUDA native export"}; ev::validate_features(f);
    require(f.values.scalar_type() == torch::kFloat64 && f.values.sizes() == torch::IntArrayRef({s.labels.size(0), 32}) && torch::isfinite(f.values).all().item<bool>() &&
        torch::equal(f.valid, s.mask.flatten(1).any(1)), "saved native geometry/finiteness/exact legal row support");
    // Historical exports precede explicit per-view IDs. Their exact original
    // byte digest binds the row order; any newer explicit IDs must agree too.
    const auto keys = a.keys(); if (std::find(keys.begin(), keys.end(), prefix + "_source_ids") != keys.end())
      require(torch::equal(read(a, prefix + "_source_ids"), s.source_ids) && text(a, prefix + "_source_ids_json") == fi::strings(s.ids), "explicit native row identities");
    return f;
  };
  return {surface("training", c.training), surface("validation", c.validation), surface("deleted", c.deleted), text(a, "scaler_identity")};
}
ev::FeatureSurface slice(const ev::FeatureSurface &s, int64_t start, int64_t width, const std::string &name) {
  ev::validate_features(s); require(s.values.size(1) == 32 && ((start == 0 && width == 20) || (start == 20 && width == 12)), "closed native branch slice");
  return {s.values.narrow(1, start, width).clone(), s.valid.clone(), s.provenance + "; exact coordinate slice " + name + "; no PCA"};
}
int64_t fixed_exact(const NativeViews &initial, const NativeViews &trained) {
  require(initial.scaler_identity == trained.scaler_identity && !initial.scaler_identity.empty(), "unchanged original TRAIN scaler identity"); int64_t cells = 0;
  for (const auto &pair : std::vector<std::pair<ev::FeatureSurface, ev::FeatureSurface>>{{initial.training, trained.training}, {initial.validation, trained.validation}, {initial.deleted, trained.deleted}}) {
    require(torch::equal(pair.first.valid, pair.second.valid) && torch::equal(pair.first.values.narrow(1, 20, 12), pair.second.values.narrow(1, 20, 12)), "fixed last12 exact at0/512 on every view");
    cells += pair.first.values.size(0) * 12;
  }
  return cells;
}
// Physical retained locations; historical receipt identities stay unchanged.
constexpr const char *original_root = "/mityane/code/embeddings/output/runs/rpb-v20-capacity/admission-0gN4uD";
constexpr const char *branch_root = "/mityane/code/embeddings/output/runs/rpb-v20-branches/diagnostic-YA3Nws";
constexpr const char *branch_inventory = "cefa413c560bdf2ab1eee5be2e3f34feca6e2f723a99627f9eeae683c4c11d93";
struct BoundBranch {
  fs::path root;
  std::map<std::string, std::string> inventory, used;
  explicit BoundBranch(const fs::path &p) : root(p) {
    direct(root); direct(root / "artifacts.sha256", true);
    require(sha256(bytes(root / "artifacts.sha256")) == branch_inventory, "pinned branch controls manifest");
    std::istringstream lines(bytes(root / "artifacts.sha256")); std::string line;
    while (std::getline(lines, line)) {
      require(line.size() > 66 && fi::is_sha(line.substr(0, 64)) && line.substr(64, 2) == "  ", "closed branch inventory line");
      const auto role = line.substr(66); const fs::path relative(role);
      require(!relative.is_absolute() && relative.lexically_normal() == relative && role.find("..") == std::string::npos &&
          inventory.emplace(role, line.substr(0, 64)).second, "safe unique saved branch role");
    }
    const auto done = bytes(root / "complete.json"); direct(root / "complete.json", true);
    require(string_field(done, "protocol") == "configurable-branch-v1" && string_field(done, "status") == "complete" &&
        string_field(done, "artifacts_manifest_sha256") == branch_inventory && integer_field(done, "new_head_fit_pairs") == 27 &&
        integer_field(done, "encoder_calls") == 0, "completed saved branch controls");
  }
  fs::path bind(const std::string &role) {
    require(inventory.count(role), "declared saved branch role"); const auto p = root / role; direct(p, true);
    require(sha256(bytes(p)) == inventory.at(role), "saved branch bytes"); used.emplace(role, inventory.at(role)); return p;
  }
  void verify() const {
    require(sha256(bytes(root / "artifacts.sha256")) == branch_inventory, "saved branch inventory unchanged");
    for (const auto &[role, digest] : used) { direct(root / role, true); require(sha256(bytes(root / role)) == digest, "saved branch control unchanged"); }
  }
  std::string json() const {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto &[role, digest] : used) { if (!first) out << ','; first = false;
      out << "{\"path\":" << quote((root / role).string()) << ",\"sha256\":" << quote(digest) << '}'; }
    return out.str() + ']';
  }
};
struct CandidateFiles {
  std::map<fs::path, std::string> files;
  fs::path bind(const fs::path &p) { direct(p, true); const auto hash = sha256(bytes(p));
    require(!files.count(p) || files.at(p) == hash, "candidate bytes stable during readout"); files[p] = hash; return p; }
  void verify() const { for (const auto &[p, digest] : files) { direct(p, true); require(sha256(bytes(p)) == digest, "candidate saved input unchanged"); } }
  std::string json() const { std::ostringstream out; out << '['; bool first = true;
    for (const auto &[p, digest] : files) { if (!first) out << ','; first = false; out << "{\"path\":" << quote(p.string()) << ",\"sha256\":" << quote(digest) << '}'; }
    return out.str() + ']'; }
};
NativeViews load_candidate(CandidateFiles &files, const fs::path &producer, int64_t master, int64_t point,
    const Cohort &c, const std::string &source_sha, const std::string &card_sha, const std::string &config_sha) {
  const auto p = producer / "candidates" / ("master-" + std::to_string(master)) / ("point-" + std::to_string(point)) / "features.pt";
  torch::serialize::InputArchive a; a.load_from(files.bind(p).string(), torch::kCPU);
  require(text(a, "protocol") == protocol && text(a, "artifact_kind") == "configurable_capacity_native_feature_views_v1" &&
      text(a, "model_tag") == "RPB-v21.alt-01" && text(a, "dataset_id") == "TEMPO-5" && text(a, "profile") == "A" &&
      read(a, "master").item<int64_t>() == master && read(a, "point").item<int64_t>() == point && read(a, "native_width").item<int64_t>() == 32,
      "candidate saved feature geometry/tag/master/point");
  require(text(a, "source_fingerprint") == source_sha && text(a, "card_sha") == card_sha &&
      text(a, "input_manifest_sha") == original_inputs && text(a, "config_sha") == config_sha,
      "candidate saved native source/card/config/original data binding");
  const auto surface = [&](const std::string &prefix, const Split &s) {
    ev::FeatureSurface f{read(a, prefix + "_values"), read(a, prefix + "_valid"), p.string() + "; saved candidate CUDA native32"}; ev::validate_features(f);
    require(f.values.scalar_type() == torch::kFloat64 && f.values.sizes() == torch::IntArrayRef({s.labels.size(0), 32}) &&
        torch::isfinite(f.values).all().item<bool>() && torch::equal(f.valid, s.mask.flatten(1).any(1)) &&
        torch::equal(read(a, prefix + "_source_ids"), s.source_ids) && text(a, prefix + "_source_ids_json") == fi::strings(s.ids),
        "candidate native exact original row/source/validity association"); return f;
  };
  return {surface("training", c.training), surface("validation", c.validation), surface("deleted", c.deleted), text(a, "scaler_identity")};
}
bool exact_native(const torch::Tensor &a,const torch::Tensor &b){
  if(!a.device().is_cpu()||!b.device().is_cpu()||a.scalar_type()!=b.scalar_type()||a.sizes()!=b.sizes())return false;
  const auto x=a.contiguous(),y=b.contiguous();return !x.numel()||!std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size());
}
void initial_exact(const NativeViews &a, const NativeViews &b) {
  require(a.scaler_identity == b.scaler_identity, "initialized candidate uses original TRAIN scaler");
  for (const auto &pair : std::vector<std::pair<ev::FeatureSurface, ev::FeatureSurface>>{{a.training,b.training},{a.validation,b.validation},{a.deleted,b.deleted}})
    require(exact_native(pair.first.values, pair.second.values) && exact_native(pair.first.valid, pair.second.valid),
        "candidate initialized complete32/learned20 values and validity exact saved v20 controls");
}
void fixed_control_exact(const NativeViews &a, const NativeViews &b) {
  require(a.scaler_identity == b.scaler_identity, "candidate retains original TRAIN scaler");
  for (const auto &pair : std::vector<std::pair<ev::FeatureSurface, ev::FeatureSurface>>{{a.training,b.training},{a.validation,b.validation},{a.deleted,b.deleted}})
    require(torch::equal(pair.first.values.narrow(1,20,12), pair.second.values.narrow(1,20,12)) && torch::equal(pair.first.valid, pair.second.valid),
        "candidate fixed12/support matches saved control");
}
ev::FixedFeatureReadoutRun heads(const fs::path &out, int64_t master, const Cohort &c, const NativeViews &trained) {
  ev::FixedFeatureReadoutRun run; run.output_directory = out.string(); run.master_seed = master;
  run.training_labels = c.training.labels; run.validation_labels = c.validation.labels;
  run.training_source_ids = c.training.ids; run.validation_source_ids = c.validation.ids;
  run.methods.push_back({"native_512",trained.training,trained.validation,trained.deleted,false,{}});
  run.methods.push_back({"learned_20_512",slice(trained.training,0,20,"learned_20_512"),
      slice(trained.validation,0,20,"learned_20_512"),slice(trained.deleted,0,20,"learned_20_512"),false,{}});
  return run;
}
void close(const torch::Tensor &a, const torch::Tensor &b, const std::string &why) {
  require(a.device().is_cpu() && b.device().is_cpu() && a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes(), why + " geometry");
  if (a.is_floating_point()) require(torch::allclose(a,b,2e-9,2e-9),why + " arithmetic"); else require(torch::equal(a,b),why + " exact");
}
ev::FeatureSurface load_surface(const fs::path &p) {
  direct(p,true); torch::serialize::InputArchive a; a.load_from(p.string(),torch::kCPU);
  ev::FeatureSurface f{read(a,"features"),read(a,"valid"),text(a,"provenance")}; ev::validate_features(f); return f;
}
uint64_t named(const std::string &s) { uint64_t h=14695981039346656037ULL; for(unsigned char c:s){h^=c;h*=1099511628211ULL;} return h; }
std::string score_json(const ev::Score &s) {
  std::ostringstream out; out<<std::setprecision(17)<<"{\"total\":"<<s.total<<",\"valid\":"<<s.valid<<",\"correct\":"<<s.correct<<",\"accuracy\":";
  if(s.supported)out<<s.accuracy;else out<<"null";out<<",\"coverage\":"<<s.coverage<<",\"full_population_correctness\":"<<double(s.correct)/s.total<<'}';return out.str();
}
std::string interval_json(const ev::GroupedInterval &i) {
  std::ostringstream out;out<<std::setprecision(17)<<"{\"source_groups\":"<<i.source_groups<<",\"estimate\":";
  if(i.source_groups)out<<i.estimate;else out<<"null";out<<",\"lower\":";if(i.supported)out<<i.lower;else out<<"null";
  out<<",\"upper\":";if(i.supported)out<<i.upper;else out<<"null";return out.str()+'}';
}
struct Pair { std::string candidate,reference,id; fs::path reference_root; };
std::vector<Pair> comparisons(int64_t master) {
  const auto m="master-"+std::to_string(master);
  return {{"native_512","native_A_0","candidate_native_512_minus_native_A_0",fs::path(original_root)/"readouts"/m/"models"/"native_A_0"},
          {"native_512","native_A_512","candidate_native_512_minus_native_A_512",fs::path(original_root)/"readouts"/m/"models"/"native_A_512"},
          {"learned_20_512","learned_20_0","candidate_learned_20_512_minus_learned_20_0",fs::path(branch_root)/"readouts"/m/"readouts"/"learned_20_0"},
          {"learned_20_512","learned_20_512","candidate_learned_20_512_minus_learned_20_512",fs::path(branch_root)/"readouts"/m/"readouts"/"learned_20_512"}};
}
int64_t audit_directory(const fs::path &root, const fs::path &output={}) {
  direct(root);int64_t fits=0,predictions=0,paired=0;bool first=true;
  std::ostringstream replay;replay<<std::setprecision(17)<<"{\"protocol\":\"saved-learned-relation-arithmetic-v1\",\"encoder_calls\":0,\"head_fits\":0,\"records\":[";
  std::vector<std::pair<int64_t,fs::path>> cohorts;
  for(const auto &entry:fs::directory_iterator(root))if(entry.is_directory()&&entry.path().filename().string().rfind("master-",0)==0){
    const auto master=std::stoll(entry.path().filename().string().substr(7));
    require(master==941000||std::find(std::begin(masters),std::end(masters),master)!=std::end(masters),"closed saved master");cohorts.push_back({master,entry.path()/"models"});}
  std::sort(cohorts.begin(),cohorts.end());require(!cohorts.empty(),"saved candidate cohorts required");
  for(const auto &[master,models]:cohorts)for(const std::string name:methods){
    const auto method=models/name;const auto training=load_surface(method/"training-features.pt");const int64_t width=name=="native_512"?32:20;
    require(training.values.size(1)==width,"saved candidate coordinate width");const auto rows=torch::nonzero(training.valid).flatten(),x=training.values.index_select(0,rows);
    const auto mean=x.mean(0),scale=(x-mean).pow(2).mean(0).sqrt().clamp_min(1e-8),normalized=(x-mean)/scale;
    const auto probe_mean=normalized.mean(0),probe_scale=(normalized-probe_mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
    torch::serialize::InputArchive train;train.load_from((method/"training-features.pt").string(),torch::kCPU);
    for(const auto repetition:repetitions){const auto rep="rep-"+std::to_string(repetition);const auto dir=method/rep;direct(dir/"fit.pt",true);
      torch::serialize::InputArchive fit;fit.load_from((dir/"fit.pt").string(),torch::kCPU);
      require(read(fit,"outer_normalizer_applied").item<bool>()&&read(fit,"outer_fitted_rows").item<int64_t>()==rows.numel()&&read(fit,"fitted_rows").item<int64_t>()==rows.numel(),"TRAIN-only fitted population");
      require(text(fit,"training_source_ids_json")==text(train,"source_ids_json")&&text(fit,"actual_probe_seed_decimal")==std::to_string(ev::stream_seed(repetition,uint64_t(width)))&&
        read(fit,"ridge_penalty").item<double>()==1&&read(fit,"tiny_hidden").item<int64_t>()==16&&read(fit,"tiny_steps").item<int64_t>()==100&&read(fit,"tiny_learning_rate").item<double>()==.01,"fixed recipe/seed/identity");
      close(read(fit,"feature_mean"),mean,"TRAIN outer mean");close(read(fit,"feature_scale"),scale,"TRAIN outer scale");
      for(const std::string head:{"ridge","tiny"}){close(read(fit,head+"_mean"),probe_mean,"TRAIN probe mean");close(read(fit,head+"_scale"),probe_scale,"TRAIN probe scale");}
      for(const std::string view:views){const auto surface=load_surface(method/(view+"-features.pt"));direct(dir/(view+"-predictions.pt"),true);
        torch::serialize::InputArchive prediction,original;prediction.load_from((dir/(view+"-predictions.pt")).string(),torch::kCPU);original.load_from((method/(view+"-features.pt")).string(),torch::kCPU);
        const auto input=torch::where(surface.valid.unsqueeze(1),(surface.values-mean)/scale,torch::zeros_like(surface.values));
        close(read(prediction,"probe_input_features"),input,"saved outer map");close(read(prediction,"valid"),surface.valid,"saved support");
        const auto normalized_input=[&](const std::string &head){return torch::where(surface.valid.unsqueeze(1),(input-read(fit,head+"_mean"))/read(fit,head+"_scale"),torch::zeros_like(input));};
        const auto ridge=normalized_input("ridge").matmul(read(fit,"ridge_weights"))+read(fit,"ridge_intercept");
        const auto hidden=normalized_input("tiny").matmul(read(fit,"tiny_w1"))+read(fit,"tiny_b1"),tiny=torch::tanh(hidden).matmul(read(fit,"tiny_w2"))+read(fit,"tiny_b2");
        close(read(prediction,"ridge_logits"),ridge,"saved ridge logits");close(read(prediction,"tiny_hidden_preactivation"),hidden,"saved tanh input");close(read(prediction,"tiny_logits"),tiny,"saved neural logits");
        close(read(prediction,"ridge"),ridge.argmax(1),"saved ridge decisions");close(read(prediction,"tiny_secondary"),tiny.argmax(1),"saved neural decisions");
        const auto truth=read(prediction,"labels_scoring_only");close(truth,read(original,"labels_scoring_only"),"saved scoring labels");require(text(prediction,"source_ids_json")==text(original,"source_ids_json"),"saved source identity");
        const auto ids=source_strings(text(prediction,"source_ids_json"));const auto seed=ev::stream_seed(master,named(name+'/'+rep+'/'+view));
        if(!first)replay<<',';first=false;replay<<"{\"prediction\":"<<quote(fs::relative(dir/(view+"-predictions.pt"),root).string())<<",\"ridge\":"<<score_json(ev::score(ridge.argmax(1),truth,surface.valid))
          <<",\"tiny_secondary\":"<<score_json(ev::score(tiny.argmax(1),truth,surface.valid));
        if(view!="training")replay<<",\"bootstrap_seed_decimal\":"<<quote(std::to_string(seed))<<",\"ridge_grouped_interval\":"<<interval_json(ev::grouped_accuracy_interval(ridge.argmax(1),truth,surface.valid,ids,seed,1000))
          <<",\"tiny_grouped_interval\":"<<interval_json(ev::grouped_accuracy_interval(tiny.argmax(1),truth,surface.valid,ids,seed,1000));
        replay<<'}';++predictions;}
      ++fits;}
  }
  replay<<"],\"pairs\":[";first=true;
  for(const auto &[master,models]:cohorts){if(master==941000)continue;
    BoundInputs old(original_root);BoundBranch branch(branch_root);
    for(const auto &pair:comparisons(master))for(const auto repetition:repetitions)for(const std::string view:{"validation-intact","validation-deleted"}){
      const auto rep="rep-"+std::to_string(repetition);const auto candidate=models/pair.candidate/rep/(view+"-predictions.pt"),reference=pair.reference_root/rep/(view+"-predictions.pt");
      if(pair.reference.rfind("native_",0)==0)old.bind(fs::relative(reference,original_root).string());else branch.bind(fs::relative(reference,branch_root).string());
      direct(candidate,true);torch::serialize::InputArchive a,b;a.load_from(candidate.string(),torch::kCPU);b.load_from(reference.string(),torch::kCPU);
      const auto truth=read(a,"labels_scoring_only");close(truth,read(b,"labels_scoring_only"),"paired scoring labels");require(text(a,"source_ids_json")==text(b,"source_ids_json"),"paired source identities");
      const auto ids=source_strings(text(a,"source_ids_json"));const auto common=read(a,"valid").logical_and(read(b,"valid"));auto view_name=view;std::replace(view_name.begin(),view_name.end(),'-','_');
      const auto seed=ev::stream_seed(master,named(pair.id+'/'+rep+'/'+view_name));if(!first)replay<<',';first=false;
      replay<<"{\"master\":"<<master<<",\"id\":"<<quote(pair.id)<<",\"repetition\":"<<quote(rep)<<",\"view\":"<<quote(view_name)<<",\"candidate_prediction\":"<<quote(candidate.string())
        <<",\"reference_prediction\":"<<quote(reference.string())<<",\"valid_rows\":"<<common.sum().item<int64_t>()<<",\"bootstrap_seed_decimal\":"<<quote(std::to_string(seed))
        <<",\"ridge\":"<<interval_json(ev::grouped_accuracy_interval(read(a,"ridge"),truth,common,ids,seed,1000,read(b,"ridge")))
        <<",\"tiny_secondary\":"<<interval_json(ev::grouped_accuracy_interval(read(a,"tiny_secondary"),truth,common,ids,seed,1000,read(b,"tiny_secondary")))<<'}';++paired;}
    old.verify();branch.verify();}
  require(fits==int64_t(cohorts.size())*6&&predictions==fits*3,"exact trained-only candidate fits/prediction matrix");
  replay<<"],\"fitted_pairs_replayed\":"<<fits<<",\"prediction_archives\":"<<predictions<<",\"paired_comparisons\":"<<paired<<'}';if(!output.empty())write_new(output,replay.str()+'\n');return fits;
}
std::string reuse_receipts(BoundInputs &old,BoundBranch &branch,int64_t master){
  const auto m="master-"+std::to_string(master);std::ostringstream files,methods_json;files<<'[';methods_json<<'[';bool first_file=true,first_method=true;std::set<fs::path> receipted;
  const auto record=[&](const fs::path &p){if(!receipted.insert(p).second)return;if(!first_file)files<<',';first_file=false;files<<"{\"path\":"<<quote(p.string())<<",\"sha256\":"<<quote(sha256(bytes(p)))<<'}';};
  for(const std::string method:{"native_A_0","native_A_512","learned_20_0","learned_20_512","fixed_12"}){
    const bool original=method.rfind("native_",0)==0;const auto root=original?old.root:branch.root;
    const std::string prefix="readouts/"+m+(original?"/models/":"/readouts/");const auto bind=[&](const std::string &role){return original?old.bind(role):branch.bind(role);};
    const auto report=bind(prefix+"report.json");record(report);const auto method_root=root/prefix/method;
    if(!first_method)methods_json<<',';first_method=false;methods_json<<"{\"method\":"<<quote(method)<<",\"root\":"<<quote(method_root.string())<<",\"report_path\":"<<quote(report.string())<<",\"head_fits_reused\":3}";
    for(const std::string view:views)record(bind(prefix+method+'/'+view+"-features.pt"));
    for(const auto repetition:repetitions){const auto rep="rep-"+std::to_string(repetition);record(bind(prefix+method+'/'+rep+"/fit.pt"));
      for(const std::string view:views)record(bind(prefix+method+'/'+rep+'/'+view+"-predictions.pt"));}
  }
  return "{\"files\":"+files.str()+"],\"methods\":"+methods_json.str()+"]}";
}
void self_test(const fs::path &out){new_leaf(out);torch::manual_seed(941000);
  const auto split=[](int64_t groups,const std::string &prefix){Split s;std::vector<int64_t> labels;for(int64_t i=0;i<groups;++i){s.ids.push_back(prefix+std::to_string(i));s.ids.push_back(prefix+std::to_string(i));labels.push_back(0);labels.push_back(1);}s.labels=torch::tensor(labels,torch::kInt64);return s;};
  Cohort c{split(8,"train-"),split(4,"val-"),{}};c.deleted=c.validation;
  const auto f=[](const Split &s){auto x=torch::randn({s.labels.size(0),32},torch::kFloat64);x.select(1,0).add_(s.labels.to(torch::kFloat64)*2-1);return ev::FeatureSurface{x,torch::ones({s.labels.size(0)},torch::kBool),"artificial CPU fixed-head engineering"};};
  NativeViews v{f(c.training),f(c.validation),{},"fixture-scaler"};v.deleted=v.validation;initial_exact(v,v);fixed_control_exact(v,v);
  auto wrong=v;wrong.training.values=v.training.values.clone();wrong.training.values[0][0].add_(1);bool rejected=false;try{initial_exact(v,wrong);}catch(const std::exception &){rejected=true;}require(rejected,"reject nonexact initialized reuse");
  wrong=v;wrong.deleted.valid=v.deleted.valid.clone();wrong.deleted.valid.narrow(0,0,1).zero_();rejected=false;try{fixed_control_exact(v,wrong);}catch(const std::exception &){rejected=true;}require(rejected,"reject changed control validity");
  const auto dir=out/"master-941000";require(fs::create_directory(dir),"new engineering master");ev::run_fixed_feature_readouts(heads(dir/"models",941000,c,v));require(audit_directory(out,out/"arithmetic-audit.json")==6,"six artificial paired fits only");
  write_new(out/"complete.json","{\"protocol\":\"configurable-learned-relation-v1-engineering\",\"status\":\"passed\",\"fitted_pairs\":6,\"encoder_calls\":0}\n");
}
} // namespace
int main(int argc,char **argv){try{torch::set_num_threads(1);
  if(argc==2&&std::string(argv[1])=="--source-id"){std::cout<<CONFIGURABLE_LEARNED_RELATION_SOURCE_ID<<'\n';return 0;}
  std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2)require(i+1<argc&&args.emplace(argv[i],argv[i+1]).second,"unique key/value CLI");
  if(args.count("--self-test")){require(args.size()==1,"closed engineering CLI");self_test(args.at("--self-test"));return 0;}
  if(args.count("--audit-only")){require(args.size()==1||(args.size()==2&&args.count("--audit-output")),"closed saved-audit CLI");fs::path output;
    if(args.count("--audit-output")){output=args.at("--audit-output");require(output.is_absolute()&&output.lexically_normal()==output&&!fs::exists(output)&&!fs::is_symlink(output),"new audit output");direct(output.parent_path());}
    std::cout<<"Saved arithmetic: "<<audit_directory(args.at("--audit-only"),output)<<" candidate paired fits replayed; zero encoder calls/head fits\n";return 0;}
  const std::set<std::string> required{"--original","--branch","--producer-root","--output","--stage","--card","--sources-sha256"};require(args.size()==required.size(),"closed quality reader CLI");for(const auto &key:required)require(args.count(key),"missing "+key);
  require(args.at("--original")==original_root&&args.at("--branch")==branch_root,"exact declared saved control capsules");
  const auto stage=args.at("--stage");require(stage=="screen"||stage=="third","closed stage");const std::vector<int64_t> stage_masters=stage=="screen"?std::vector<int64_t>{941101,941102}:std::vector<int64_t>{941103};
  const fs::path producer=args.at("--producer-root"),output=args.at("--output"),card=args.at("--card"),sources=args.at("--sources-sha256");direct(producer);direct(card,true);direct(sources,true);
  const auto source_sha=sha256(bytes(sources)),card_sha=sha256(bytes(card));require(source_sha==CONFIGURABLE_LEARNED_RELATION_SOURCE_ID&&card_sha==CONFIGURABLE_LEARNED_RELATION_CARD_SHA256,"compiled source/card binding");
  direct(producer/"freeze/receipt.json",true);const auto config_sha=string_field(bytes(producer/"freeze/receipt.json"),"config_sha256");require(fi::is_sha(config_sha),"frozen candidate settings");
  BoundInputs old(original_root);BoundBranch branch(branch_root);CandidateFiles files;std::map<int64_t,Cohort> cohorts;std::map<int64_t,NativeViews> trained;
  for(const auto master:stage_masters){auto cohort=load_cohort(old,master);auto initial=load_candidate(files,producer,master,0,cohort,source_sha,card_sha,config_sha),final=load_candidate(files,producer,master,512,cohort,source_sha,card_sha,config_sha);
    const auto control_initial=load_native(old,master,0,cohort),control_final=load_native(old,master,512,cohort);initial_exact(initial,control_initial);fixed_control_exact(initial,control_final);fixed_control_exact(final,control_initial);fixed_control_exact(final,control_final);fixed_exact(initial,final);
    cohorts.emplace(master,std::move(cohort));trained.emplace(master,std::move(final));}
  old.verify();branch.verify();files.verify();new_leaf(output);std::ostringstream report;
  report<<std::setprecision(17)<<"{\"protocol\":"<<quote(protocol)<<",\"dataset\":\"TEMPO-5\",\"model_tag\":\"RPB-v21.alt-01\",\"stage\":"<<quote(stage)<<",\"masters\":[";
  for(size_t i=0;i<stage_masters.size();++i){if(i)report<<',';report<<stage_masters[i];}
  report<<"],\"source_fingerprint\":"<<quote(source_sha)<<",\"card_sha\":"<<quote(card_sha)<<",\"config_sha\":"<<quote(config_sha)<<",\"input_manifest_sha\":"<<quote(original_inputs)
    <<",\"original_source_fingerprint\":"<<quote(original_source)<<",\"original_artifact_manifest_sha\":"<<quote(original_inventory)<<",\"branch_artifact_manifest_sha\":"<<quote(branch_inventory)
    <<",\"encoder_calls\":0,\"encoder_updates\":0,\"head_fitted_pairs\":"<<stage_masters.size()*6<<",\"head_fits\":"<<stage_masters.size()*12
    <<",\"control_head_fits\":0,\"validation_fits\":0,\"pca_fits\":0,\"raw_fits\":0,\"initial_control_exact\":true,\"fixed_control_exact\":true,\"cohorts\":[";bool first=true;
  for(const auto master:stage_masters){const auto dir=output/("master-"+std::to_string(master));require(fs::create_directory(dir),"new candidate head master");const auto begin=std::chrono::steady_clock::now();
    const auto result=ev::run_fixed_feature_readouts(heads(dir/"models",master,cohorts.at(master),trained.at(master))),reuse=reuse_receipts(old,branch,master);
    if(!first)report<<',';first=false;report<<"{\"master\":"<<master<<",\"CPU_head_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<",\"candidate_heads\":"<<result<<",\"reuse\":"<<reuse<<'}';
    std::cout<<"Candidate trained-only heads master="<<master<<" six paired fits; initialized/control heads reused\n"<<std::flush;}
  report<<"],\"saved_input_files\":[";first=true;
  for(const auto &ledger:std::vector<std::string>{old.json(),branch.json(),files.json()})if(ledger.size()>2){if(!first)report<<',';first=false;report<<ledger.substr(1,ledger.size()-2);}
  report<<"],\"testing_accessed\":false,\"promotion\":false}";
  old.verify();branch.verify();files.verify();require(sha256(bytes(card))==card_sha&&sha256(bytes(sources))==source_sha,"source/card unchanged");write_new(output/"report.json",report.str()+'\n');
  const auto fits=audit_directory(output,output/"arithmetic-audit.json");require(fits==int64_t(stage_masters.size())*6,"exact declared candidate fit inventory");
  write_new(output/"complete.json","{\"protocol\":"+quote(protocol)+",\"status\":\"complete\",\"saved_arithmetic_fits_replayed\":"+std::to_string(fits)+",\"head_fitted_pairs\":"+std::to_string(fits)+",\"encoder_calls\":0,\"report_sha256\":"+quote(sha256(bytes(output/"report.json")))+",\"arithmetic_audit_sha256\":"+quote(sha256(bytes(output/"arithmetic-audit.json")))+"}\n");return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
