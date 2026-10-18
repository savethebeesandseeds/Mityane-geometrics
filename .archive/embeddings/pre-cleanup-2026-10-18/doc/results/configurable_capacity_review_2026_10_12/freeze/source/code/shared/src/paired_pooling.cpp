// SPDX-License-Identifier: MIT
#include "embedding/shared/paired_pooling.h"
#include "embedding/shared/feature_stress.h"
#include <ATen/Context.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

namespace embedding::evaluation {
namespace {
namespace fs=std::filesystem;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[paired pooling] " + message);
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
template <typename T> std::string numbers(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  return out.str() + ']';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
std::string fields(const std::map<std::string, std::string> &values) {
  std::ostringstream out;
  out << '{';
  bool first = true;
  for (const auto &[key, value] : values) {
    if (!first) out << ',';
    first = false;
    out << quote(key) << ':' << quote(value);
  }
  return out.str() + '}';
}
void write_text(const fs::path &path, const std::string &text) {
  require(!fs::exists(path), "artifact exists: " + path.string());
  std::ofstream output(path);
  require(bool(output), "cannot create artifact: " + path.string());
  output << text;
  output.close();
  require(bool(output), "cannot save artifact: " + path.string());
}
void save_archive(const fs::path &path, torch::serialize::OutputArchive &output) {
  require(!fs::exists(path), "archive exists: " + path.string());
  archive::save_archive(path.string(), output);
}
uint64_t named_stream(const std::string &name) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char c : name) { hash ^= c; hash *= 1099511628211ULL; }
  return hash;
}
std::string checksum(const torch::Tensor &value) {
  const auto tensor = value.detach().to(torch::kCPU).contiguous();
  const auto *bytes = static_cast<const unsigned char *>(tensor.const_data_ptr());
  uint64_t hash = 14695981039346656037ULL;
  for (int64_t i = 0; i < tensor.numel() * tensor.element_size(); ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
Batch legal_clone(const Batch &batch) {
  return {torch::where(batch.feature_mask, batch.data.detach(), torch::zeros_like(batch.data)).clone(),
          batch.feature_mask.clone()};
}
std::string score_json(const Score &score) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total\":" << score.total << ",\"valid\":" << score.valid
      << ",\"correct\":" << score.correct << ",\"abstained\":" << score.total-score.valid
      << ",\"full_population_correctness\":" << (score.total?double(score.correct)/score.total:0)
      << ",\"coverage\":" << score.coverage << ",\"accuracy\":";
  if (score.supported) out << score.accuracy; else out << "null";
  return out.str() + '}';
}
std::string interval_json(const GroupedInterval &interval,int64_t replicates) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"method\":\"source-group percentile bootstrap; within-run conditional on fitted checkpoint/readouts\","
      << "\"replicates\":" << replicates << ",\"confidence\":0.95,\"source_groups\":" << interval.source_groups << ",\"estimate\":";
  if (interval.source_groups) out << interval.estimate; else out << "null";
  out << ",\"lower\":";
  if (interval.supported) out << interval.lower; else out << "null";
  out << ",\"upper\":";
  if (interval.supported) out << interval.upper; else out << "null";
  return out.str() + '}';
}
std::string population_json(const torch::Tensor &valid, const ControlledDataset &split) {
  std::set<std::string> all, selected, complete;
  std::map<std::string, std::pair<int64_t, int64_t>> groups;
  int64_t classes[2]{0, 0};
  for (int64_t row = 0; row < valid.size(0); ++row) {
    const auto &source = split.source_ids.at(row);
    all.insert(source); ++groups[source].first;
    if (valid[row].item<bool>()) {
      selected.insert(source); ++groups[source].second; ++classes[split.labels[row].item<int64_t>()];
    }
  }
  for (const auto &[id, count] : groups) if (count.first == count.second) complete.insert(id);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total_rows\":" << valid.size(0) << ",\"valid_rows\":" << classes[0] + classes[1]
      << ",\"class_valid_rows\":[" << classes[0] << ',' << classes[1] << "],\"total_source_groups\":" << all.size()
      << ",\"valid_source_groups\":" << selected.size() << ",\"complete_source_pairs\":" << complete.size()
      << ",\"coverage\":" << double(classes[0] + classes[1]) / valid.size(0) << '}';
  return out.str();
}
void validate_split(const ControlledDataset &split, const NativeCurveRun &run, int64_t pairs,
                    std::set<std::string> &all_sources) {
  require(split.observed.data.defined() && split.observed.data.device().is_cpu() &&
          split.observed.data.scalar_type() == torch::kFloat64 &&
          split.observed.data.sizes() == torch::IntArrayRef({2 * pairs, run.card.shape.channel_count, run.card.shape.history_length, run.card.shape.input_width}) &&
          split.observed.feature_mask.defined() && split.observed.feature_mask.device().is_cpu() &&
          split.observed.feature_mask.scalar_type() == torch::kBool &&
          split.observed.feature_mask.sizes() == split.observed.data.sizes() &&
          torch::isfinite(split.observed.data.masked_select(split.observed.feature_mask)).all().item<bool>() &&
          split.labels.defined() && split.labels.device().is_cpu() &&
          split.labels.scalar_type() == torch::kInt64 && split.labels.sizes() == torch::IntArrayRef({2 * pairs}) &&
          split.labels.ge(0).logical_and(split.labels.le(1)).all().item<bool>() &&
          split.source_ids.size() == size_t(2 * pairs), "controlled split shape/dtype/label contract");
  std::map<std::string, std::vector<int64_t>> groups;
  for (int64_t row = 0; row < 2 * pairs; ++row) groups[split.source_ids.at(row)].push_back(row);
  require(groups.size() == size_t(pairs), "controlled source pair count differs");
  for (const auto &[id, rows] : groups) {
    require(!id.empty() && rows.size() == 2 && split.labels[rows[0]].item<int64_t>() != split.labels[rows[1]].item<int64_t>() &&
            torch::equal(split.observed.feature_mask[rows[0]], split.observed.feature_mask[rows[1]]) &&
            all_sources.insert(id).second, "source groups overlap or paired labels/masks differ");
  }
}
void save_split(const fs::path &path, const ControlledDataset &split) {
  torch::serialize::OutputArchive output;
  output.write("observed", split.observed.data, true);
  output.write("feature_mask", split.observed.feature_mask, true);
  output.write("labels_scoring_only", split.labels, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  save_archive(path, output);
}
std::string split_manifest(const ControlledDataset &split, const std::string &name) {
  return "{\"split\":" + quote(name) + ",\"source_ids\":" + strings(split.source_ids) +
      ",\"observed_checksum_fnv1a64\":" + quote(checksum(split.observed.data)) +
      ",\"mask_checksum_fnv1a64\":" + quote(checksum(split.observed.feature_mask)) +
      ",\"labels_checksum_fnv1a64\":" + quote(checksum(split.labels)) + '}';
}

std::string reconstruction(const fs::path &path, const ControlledDataset &split,
                           const CurveSnapshot &snapshot, const NativeCurveRun &run) {
  const auto B = split.observed.data.size(0), C = run.card.shape.channel_count;
  const auto H = run.card.shape.history_length, F = run.card.shape.input_width, K = H / run.patch_length;
  std::vector<torch::Tensor> predictions, targets, queries, requested, visible_masks, eligibility;
  for (int64_t trial = 0; trial < K; ++trial) {
    auto artificial = torch::zeros_like(split.observed.feature_mask);
    artificial.narrow(2, trial * run.patch_length, run.patch_length).fill_(true);
    const auto requested_query = split.observed.feature_mask.logical_and(artificial);
    const auto visible = split.observed.feature_mask.logical_and(artificial.logical_not());
    const auto fixed_eligible = visible.reshape({B, C, K, -1}).any(-1).sum(-1).ge(2)
        .logical_and(requested_query.flatten(2).any(2));
    // A channel that cannot leave2observed patch groups has no reconstruction
    // trial. This prevents invalid whole-patch masks reaching a strict provider.
    artificial = artificial.logical_and(fixed_eligible.unsqueeze(-1).unsqueeze(-1));
    CurveReconstruction value;
    {
      torch::NoGradGuard no_grad;
      value = snapshot.reconstruct(legal_clone(split.observed), artificial.clone());
    }
    for (const auto &tensor : {value.prediction, value.target})
      require(tensor.defined() && tensor.device().is_cpu() && tensor.is_floating_point() &&
              !tensor.requires_grad() && tensor.sizes() == split.observed.data.sizes(),
              "reconstruction must return detached CPU standardized prediction/target[B,C,H,F]");
    require(value.eligible.defined() && value.eligible.device().is_cpu() && value.eligible.scalar_type() == torch::kBool &&
            value.eligible.sizes() == torch::IntArrayRef({B, C}) &&
            !value.eligible.logical_and(fixed_eligible.logical_not()).any().item<bool>(),
            "reconstruction eligibility exceeds fixed observed target/visible-patch support");
    const auto query = split.observed.feature_mask.logical_and(artificial)
        .logical_and(value.eligible.unsqueeze(-1).unsqueeze(-1));
    require(torch::isfinite(value.prediction.masked_select(query)).all().item<bool>() &&
            torch::isfinite(value.target.masked_select(query)).all().item<bool>(), "nonfinite standardized reconstruction on valid target");
    predictions.push_back(torch::where(query, value.prediction.to(torch::kFloat64), torch::zeros({B,C,H,F}, torch::kFloat64)));
    targets.push_back(torch::where(query, value.target.to(torch::kFloat64), torch::zeros({B,C,H,F}, torch::kFloat64)));
    queries.push_back(query); requested.push_back(requested_query);
    visible_masks.push_back(split.observed.feature_mask.logical_and(artificial.logical_not()));
    eligibility.push_back(value.eligible.detach().clone());
  }
  const auto prediction = torch::stack(predictions), target = torch::stack(targets);
  const auto query = torch::stack(queries), requested_query = torch::stack(requested);
  const auto error = prediction - target;
  require(torch::isfinite(error.masked_select(query)).all().item<bool>(), "standardized reconstruction difference overflow");
  const auto absolute = error.abs();
  const auto huber = torch::where(absolute.le(run.huber_delta), .5 * error.clamp(-run.huber_delta,run.huber_delta).square(),
                                  run.huber_delta * (absolute - .5 * run.huber_delta));
  const std::vector<int64_t> axes{0,3,4};
  const auto counts = query.sum(axes), channel_valid = counts.gt(0);
  const auto channel_mae = absolute.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto channel_huber = huber.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto example_valid = channel_valid.any(1);
  const auto channel_count = channel_valid.sum(1).clamp_min(1).to(torch::kFloat64);
  const auto example_mae = channel_mae.sum(1) / channel_count, example_huber = channel_huber.sum(1) / channel_count;
  require(torch::isfinite(example_mae).all().item<bool>() && torch::isfinite(example_huber).all().item<bool>(),
          "reconstruction hierarchical reduction overflow");
  const auto retained = example_valid.sum().item<int64_t>(), target_cells = counts.sum().item<int64_t>();
  const auto full_targets = requested_query.sum().item<int64_t>();
  double mae = 0, loss = 0;
  if (retained) {
    mae = example_mae.masked_select(example_valid).mean().item<double>();
    loss = example_huber.masked_select(example_valid).mean().item<double>();
    require(std::isfinite(mae) && std::isfinite(loss), "reconstruction summary overflow");
  }
  torch::serialize::OutputArchive output;
  output.write("standardized_prediction", prediction, true); output.write("standardized_target", target, true);
  output.write("target_mask", query, true); output.write("requested_observed_target_mask", requested_query, true);
  output.write("visible_mask", torch::stack(visible_masks), true); output.write("trial_channel_eligible", torch::stack(eligibility), true);
  output.write("channel_target_counts", counts, true); output.write("channel_valid", channel_valid, true);
  output.write("channel_standardized_mae", channel_mae, true); output.write("channel_standardized_huber", channel_huber, true);
  output.write("example_valid", example_valid, true); output.write("example_standardized_mae", example_mae, true);
  output.write("example_standardized_huber", example_huber, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  save_archive(path, output);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"status\":" << quote(retained ? "measured" : "unsupported_zero_support")
      << ",\"units\":\"frozen training-scaler standardized\",\"total_examples\":" << B << ",\"valid_examples\":" << retained
      << ",\"example_coverage\":" << double(retained)/B << ",\"valid_channels\":" << channel_valid.sum().item<int64_t>()
      << ",\"valid_target_cells\":" << target_cells << ",\"requested_observed_target_cells\":" << full_targets
      << ",\"target_coverage\":";
  if (full_targets) out << double(target_cells)/full_targets; else out << "null";
  out << ",\"standardized_mae\":";
  if (retained) out << mae; else out << "null";
  out << ",\"standardized_huber\":";
  if (retained) out << loss; else out << "null";
  out << ",\"query_checksum_fnv1a64\":" << quote(checksum(query))
      << ",\"artifact\":" << quote(path.filename().string()) << '}';
  return out.str();
}
void check_progress(const CurveProgress &progress, int64_t budget, const CurveProgress *previous) {
  require(progress.completed == budget && progress.attempted >= progress.completed && progress.parameter_count > 0 &&
          progress.cuda_parameter_count >= 0 && progress.cuda_parameter_count <= progress.parameter_count &&
          progress.sampled_rows >= 0 && std::isfinite(progress.training_seconds) && progress.training_seconds >= 0 &&
          !progress.training_device.empty() && !progress.preprocessing_id.empty() && !progress.training_dataset_id.empty(),
          "trainer counters/device/scaler/dataset/timing contract");
  if (previous)
    require(progress.attempted >= previous->attempted && progress.sampled_rows >= previous->sampled_rows &&
            progress.training_seconds >= previous->training_seconds && progress.parameter_count == previous->parameter_count &&
            progress.cuda_parameter_count == previous->cuda_parameter_count &&
            progress.training_device == previous->training_device && progress.preprocessing_id == previous->preprocessing_id &&
            progress.training_dataset_id == previous->training_dataset_id, "trainer reset its absolute counter/model/scaler state");
  else require(progress.completed == 0 && progress.attempted == 0, "point0 must be exact fresh initialization");
  if (previous) {
    require(progress.losses.size() >= previous->losses.size(), "trainer discarded its cumulative loss trace");
    for (size_t i = 0; i < previous->losses.size(); ++i) {
      const auto &a = previous->losses[i], &b = progress.losses[i];
      require(a.attempted == b.attempted && a.completed == b.completed && a.target_cells == b.target_cells &&
              a.loss == b.loss && a.gradient_norm == b.gradient_norm, "trainer changed an earlier counter/loss prefix");
    }
  }
  int64_t attempted = 0, completed = 0;
  for (const auto &loss : progress.losses) {
    require(loss.attempted > attempted && loss.completed > completed && loss.completed <= progress.completed &&
            loss.attempted <= progress.attempted && loss.target_cells > 0 && std::isfinite(loss.loss) &&
            std::isfinite(loss.gradient_norm) && loss.gradient_norm >= 0, "nonfinite/unordered loss trace");
    attempted = loss.attempted; completed = loss.completed;
  }
}
std::string progress_json(const CurveProgress &progress) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"attempted\":" << progress.attempted << ",\"completed\":" << progress.completed
      << ",\"parameter_count\":" << progress.parameter_count << ",\"cuda_parameter_count\":" << progress.cuda_parameter_count
      << ",\"sampled_rows\":" << progress.sampled_rows << ",\"cumulative_training_seconds\":" << progress.training_seconds
      << ",\"training_device\":" << quote(progress.training_device) << ",\"preprocessing_id\":" << quote(progress.preprocessing_id)
      << ",\"training_dataset_id\":" << quote(progress.training_dataset_id)
      << ",\"last_input_cuda\":" << (progress.last_input_cuda ? "true":"false")
      << ",\"last_loss_cuda\":" << (progress.last_loss_cuda ? "true":"false")
      << ",\"finite_gradients\":" << (progress.finite_gradients ? "true":"false")
      << ",\"weights_changed_from_initialization\":" << (progress.weights_changed ? "true":"false") << ",\"loss_trace\":[";
  for (size_t i = 0; i < progress.losses.size(); ++i) {
    if (i) out << ',';
    const auto &loss = progress.losses[i];
    out << "{\"attempted\":" << loss.attempted << ",\"completed\":" << loss.completed
        << ",\"loss\":" << loss.loss << ",\"gradient_norm\":" << loss.gradient_norm << ",\"target_cells\":" << loss.target_cells << '}';
  }
  return out.str() + "]}";
}

struct MeasurementIsolation {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  MeasurementIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i)generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &generator:generators)states.push_back(generator.get_state().clone());
  }
  ~MeasurementIsolation() noexcept {
    try{for(size_t i=0;i<generators.size();++i)generators[i].set_state(states[i]);}catch(...){std::terminate();}
  }
};
struct ThreadsIsolation {
  int previous{at::get_num_threads()};
  ~ThreadsIsolation() noexcept {try{at::set_num_threads(previous);}catch(...){std::terminate();}}
};
void durable_selection(const fs::path &path,const std::string &value) {
  write_text(path,value);
  const int file=::open(path.c_str(),O_RDONLY);
  require(file>=0,"cannot open persisted selection for fsync");
  const int synced=::fsync(file),closed=::close(file);
  require(synced==0 && closed==0,"cannot durably flush persisted selection");
  const int directory=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
  require(directory>=0,"cannot open selection directory for fsync");
  const int directory_synced=::fsync(directory),directory_closed=::close(directory);
  require(directory_synced==0 && directory_closed==0,"cannot durably flush selection directory");
}
std::string bytes(const fs::path &path) {
  std::ifstream input(path, std::ios::binary); require(bool(input), "cannot read declared file: " + path.string());
  std::ostringstream out; out << input.rdbuf(); require(!input.bad(), "file read failed: " + path.string());
  return out.str();
}
// SHA-256 over exact file bytes; no platform process or additional dependency.
std::string sha256(const std::string &input) {
  static constexpr std::array<uint32_t,64> constants{
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  require(input.size() <= std::numeric_limits<uint64_t>::max()/8, "file too large for SHA-256");
  std::vector<uint8_t> padded(input.begin(),input.end()); padded.push_back(0x80);
  while (padded.size()%64 != 56) padded.push_back(0);
  const uint64_t bits=uint64_t(input.size())*8;
  for (int shift=56;shift>=0;shift-=8) padded.push_back(uint8_t(bits >> shift));
  std::array<uint32_t,8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  auto rotate=[](uint32_t value,int count) { return (value>>count)|(value<<(32-count)); };
  for (size_t offset=0;offset<padded.size();offset+=64) {
    std::array<uint32_t,64> schedule{};
    for (size_t i=0;i<16;++i) for (size_t j=0;j<4;++j) schedule[i]=(schedule[i]<<8)|padded[offset+4*i+j];
    for (size_t i=16;i<64;++i) {
      const auto x=schedule[i-15], y=schedule[i-2];
      schedule[i]=schedule[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+schedule[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));
    }
    auto a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for (size_t i=0;i<64;++i) {
      const auto first=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+constants[i]+schedule[i];
      const auto second=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
      h=g;g=f;f=e;e=d+first;d=c;c=b;b=a;a=first+second;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
  }
  std::ostringstream out; out << std::hex << std::setfill('0');
  for (const auto value:state) out << std::setw(8) << value;
  return out.str();
}
bool testing_filename(const fs::path &path) {
  std::string token;
  auto test=[&] { return token=="test" || token=="testing"; };
  for (const unsigned char c:path.filename().string()) {
    if (std::isalnum(c)) token+=char(std::tolower(c));
    else { if (test()) return true; token.clear(); }
  }
  return test();
}
struct InputGuard {
  std::map<std::string,std::string> originals;
  void add(const std::string &path,const std::string &expected) {
    require(!path.empty() && fs::is_regular_file(path) && !testing_filename(path), "missing/non-file/testing-marked declared archive: "+path);
    const auto canonical=fs::canonical(path).string();
    require(!testing_filename(canonical), "resolved archive is testing-marked: "+canonical);
    if (!originals.count(canonical)) originals.emplace(canonical,bytes(canonical));
    if (!expected.empty()) require(expected==sha256(originals.at(canonical)), "declared SHA-256 differs: "+path);
  }
  void verify() const {
    for (const auto &[path,original]:originals) require(bytes(path)==original, "input archive bytes changed: "+path);
  }
  std::string manifest() const {
    std::ostringstream out;out << "{\"checksum_algorithm\":\"sha256-file-bytes\",\"preservation_check\":\"exact file-byte equality\",\"files\":[";
    bool first=true;
    for (const auto &[path,original]:originals) {
      if (!first) out << ',';
      first=false;out << "{\"path\":" << quote(path) << ",\"bytes\":" << original.size() << ",\"sha256\":" << quote(sha256(original)) << '}';
    }
    return out.str()+"]}";
  }
};
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

FeatureSurface global_surface(const CurveSnapshot &snapshot,const Batch &batch,const NativeCurveRun &run) {
  require(bool(snapshot.features.extract) && !snapshot.features.provenance.empty(),"snapshot has no frozen extraction/provenance");
  std::optional<std::string> global;
  for(const auto &[name,description]:snapshot.features.surfaces)if(description.kind==SurfaceKind::global) {
    require(!global && safe_name(name) && description.channel_order.empty() && !description.support_rule.empty(),"snapshot requires one declared typed global export");global=name;
  }
  require(global.has_value(),"snapshot has no typed global export");
  auto all=snapshot.features.extract(legal_clone(batch));require(all.count(*global),"snapshot omitted global export");
  auto out=all.at(*global);validate_features(out);
  require(out.values.size(0)==batch.data.size(0) && out.values.size(1)==run.export_width &&
      !out.valid.logical_and(batch.feature_mask.flatten(1).any(1).logical_not()).any().item<bool>(),"native global geometry/support mismatch");
  return {out.values.detach().clone(),out.valid.clone(),out.provenance+"; "+snapshot.features.provenance};
}
bool fit_supported(const FeatureSurface &surface,const torch::Tensor &labels) {
  const auto selected=labels.masked_select(surface.valid);
  return selected.numel()>=2 && selected.eq(0).any().item<bool>() && selected.eq(1).any().item<bool>();
}
FeatureSurface mask_surface(const Batch &batch) {
  return {batch.feature_mask.to(torch::kFloat64).flatten(1),torch::ones({batch.data.size(0)},torch::kBool),"class-independent source-paired observation masks only"};
}
torch::Tensor read_tensor(torch::serialize::InputArchive &archive,const std::string &key) {
  torch::Tensor out;archive.read(key,out,true);return out.detach().clone();
}
void finite_tensor(const torch::Tensor &value,const std::vector<int64_t> &shape,const std::string &label) {
  require(value.defined() && value.device().is_cpu() && value.scalar_type()==torch::kFloat64 && value.sizes()==torch::IntArrayRef(shape) &&
      torch::isfinite(value).all().item<bool>(),"invalid retained float64 tensor: "+label);
}
ControlledDataset load_observations(const std::string &path) {
  torch::serialize::InputArchive archive;archive.load_from(path,torch::kCPU);ControlledDataset out;
  out.observed={read_tensor(archive,"observed"),read_tensor(archive,"feature_mask")};
  out.labels=read_tensor(archive,"labels_scoring_only");out.source_ids=SourceParser(archive::tensor_text(read_tensor(archive,"source_ids_json"))).parse();
  // Historic legal-observation archives do not contain hidden clean data. The
  // generic stress validator needs a nonempty schema witness only; this legal
  // observation clone is never supplied as a clean target or fitting surface.
  out.clean=legal_clone(out.observed);return out;
}
FeatureSurface load_features(const std::string &path,const ControlledDataset &split,int64_t width) {
  torch::serialize::InputArchive archive;archive.load_from(path,torch::kCPU);
  FeatureSurface out{read_tensor(archive,"features"),read_tensor(archive,"valid"),archive::tensor_text(read_tensor(archive,"provenance"))};validate_features(out);
  require(out.values.sizes()==torch::IntArrayRef({split.labels.size(0),width}) && !out.provenance.empty() &&
      torch::equal(read_tensor(archive,"labels_scoring_only"),split.labels) &&
      SourceParser(archive::tensor_text(read_tensor(archive,"source_ids_json"))).parse()==split.source_ids &&
      !out.valid.logical_and(split.observed.feature_mask.flatten(1).any(1).logical_not()).any().item<bool>(),"retained feature row lineage/support mismatch");
  return out;
}
struct FrozenReadout {
  torch::Tensor feature_mean,feature_scale,pca_mean,pca_components,pca_singular_values,pca_numerical_rank;
  torch::Tensor ridge_mean,ridge_scale,ridge_weights,ridge_intercept,tiny_mean,tiny_scale,w1,b1,w2,b2;
  int64_t fitted_rows{0},input_width{0},probe_width{0};uint64_t seed{0};
  std::vector<torch::Tensor> tensors() const {
    std::vector<torch::Tensor> out{feature_mean,feature_scale,ridge_mean,ridge_scale,ridge_weights,ridge_intercept,tiny_mean,tiny_scale,w1,b1,w2,b2};
    if(pca_components.defined()){out.push_back(pca_mean);out.push_back(pca_components);out.push_back(pca_singular_values);out.push_back(pca_numerical_rank);}
    return out;
  }
  std::vector<torch::Tensor> witness() const {auto out=tensors();for(auto &tensor:out)tensor=tensor.detach().clone();return out;}
  void unchanged(const std::vector<torch::Tensor> &before) const {
    const auto after=tensors();require(before.size()==after.size(),"frozen fit structure changed");
    for(size_t i=0;i<before.size();++i)require(torch::equal(before[i],after[i]),"retained TRAIN-fitted tensor changed");
  }
  static FeatureSurface normalize(const FeatureSurface &input,const torch::Tensor &mean,const torch::Tensor &scale) {
    validate_features(input);require(input.values.size(1)==mean.numel(),"loaded normalizer width mismatch");
    auto out=torch::zeros_like(input.values,torch::kFloat64);const auto selected=input.valid.nonzero().flatten();
    out.index_copy_(0,selected,(input.values.to(torch::kFloat64).index_select(0,selected)-mean)/scale);
    require(torch::isfinite(out).all().item<bool>(),"frozen normalization overflow");return {out,input.valid.clone(),input.provenance};
  }
  StressPredictions predict(const FeatureSurface &input) const {
    torch::NoGradGuard no_grad;auto x=normalize(input,feature_mean,feature_scale);
    if(pca_components.defined()) {
      auto projected=torch::zeros({x.values.size(0),probe_width},torch::kFloat64);const auto rows=x.valid.nonzero().flatten();
      projected.index_copy_(0,rows,(x.values.index_select(0,rows)-pca_mean).matmul(pca_components));
      require(torch::isfinite(projected).all().item<bool>(),"loaded standalone PCA overflow");x.values=projected;
    }
    const auto ridge=normalize(x,ridge_mean,ridge_scale).values.matmul(ridge_weights)+ridge_intercept;
    const auto hidden=normalize(x,tiny_mean,tiny_scale).values.matmul(w1)+b1;
    require(torch::isfinite(ridge).all().item<bool>() && torch::isfinite(hidden).all().item<bool>(),"frozen probe arithmetic overflow");
    const auto tiny=torch::tanh(hidden).matmul(w2)+b2;require(torch::isfinite(tiny).all().item<bool>(),"frozen neural logits overflow");
    return {x.valid.clone(),ridge.argmax(1),tiny.argmax(1)};
  }
  void save(const fs::path &path) const {
    torch::serialize::OutputArchive out;
    out.write("feature_mean",feature_mean,true);out.write("feature_scale",feature_scale,true);out.write("fitted_rows",torch::tensor(fitted_rows),true);
    if(pca_components.defined()){out.write("pca_mean",pca_mean,true);out.write("pca_components",pca_components,true);out.write("pca_singular_values",pca_singular_values,true);out.write("pca_numerical_rank",pca_numerical_rank,true);}
    out.write("ridge_mean",ridge_mean,true);out.write("ridge_scale",ridge_scale,true);out.write("ridge_weights",ridge_weights,true);out.write("ridge_intercept",ridge_intercept,true);
    out.write("tiny_mean",tiny_mean,true);out.write("tiny_scale",tiny_scale,true);out.write("tiny_w1",w1,true);out.write("tiny_b1",b1,true);out.write("tiny_w2",w2,true);out.write("tiny_b2",b2,true);
    out.write("actual_probe_seed_decimal",archive::text_tensor(std::to_string(seed)),true);save_archive(path,out);
  }
  static FrozenReadout load(const std::string &path,int64_t input,int64_t width,bool pca,const NativeCurveRun &recipe,uint64_t actual_seed,int64_t rows) {
    torch::serialize::InputArchive archive;archive.load_from(path,torch::kCPU);FrozenReadout out;out.input_width=input;out.probe_width=width;out.seed=actual_seed;
    out.feature_mean=read_tensor(archive,"feature_mean");out.feature_scale=read_tensor(archive,"feature_scale");out.fitted_rows=read_tensor(archive,"fitted_rows").item<int64_t>();
    require(out.fitted_rows==rows && archive::tensor_text(read_tensor(archive,"actual_probe_seed_decimal"))==std::to_string(actual_seed),"retained fit TRAIN count/head seed mismatch");
    torch::Tensor projection;const bool has_pca=archive.try_read("pca_components",projection,true);require(has_pca==pca,"native/control PCA contract mismatch");
    if(pca) {
      out.pca_components=projection.detach().clone();out.pca_mean=read_tensor(archive,"pca_mean");out.pca_singular_values=read_tensor(archive,"pca_singular_values");out.pca_numerical_rank=read_tensor(archive,"pca_numerical_rank");
      finite_tensor(out.pca_mean,{input},"PCA mean");finite_tensor(out.pca_components,{input,width},"PCA components");
      require(out.pca_singular_values.device().is_cpu() && out.pca_singular_values.scalar_type()==torch::kFloat64 && out.pca_singular_values.dim()==1 &&
          torch::isfinite(out.pca_singular_values).all().item<bool>() && out.pca_numerical_rank.item<int64_t>()>=width,"retained PCA spectrum/rank");
    }
    out.ridge_mean=read_tensor(archive,"ridge_mean");out.ridge_scale=read_tensor(archive,"ridge_scale");out.ridge_weights=read_tensor(archive,"ridge_weights");out.ridge_intercept=read_tensor(archive,"ridge_intercept");
    out.tiny_mean=read_tensor(archive,"tiny_mean");out.tiny_scale=read_tensor(archive,"tiny_scale");out.w1=read_tensor(archive,"tiny_w1");out.b1=read_tensor(archive,"tiny_b1");out.w2=read_tensor(archive,"tiny_w2");out.b2=read_tensor(archive,"tiny_b2");
    for(const auto &item:std::vector<std::pair<torch::Tensor,std::string>>{{out.feature_mean,"outer mean"},{out.feature_scale,"outer scale"}})finite_tensor(item.first,{input},item.second);
    for(const auto &item:std::vector<std::pair<torch::Tensor,std::string>>{{out.ridge_mean,"ridge mean"},{out.ridge_scale,"ridge scale"},{out.tiny_mean,"tiny mean"},{out.tiny_scale,"tiny scale"}})finite_tensor(item.first,{width},item.second);
    finite_tensor(out.ridge_weights,{width,2},"ridge weights");finite_tensor(out.ridge_intercept,{2},"ridge intercept");finite_tensor(out.w1,{width,recipe.tiny_hidden},"tiny W1");finite_tensor(out.b1,{recipe.tiny_hidden},"tiny B1");finite_tensor(out.w2,{recipe.tiny_hidden,2},"tiny W2");finite_tensor(out.b2,{2},"tiny B2");
    require(out.feature_scale.gt(0).all().item<bool>() && out.ridge_scale.gt(0).all().item<bool>() && out.tiny_scale.gt(0).all().item<bool>(),"nonpositive retained scale");return out;
  }
};
struct FrozenRaw {
  torch::Tensor mean,scale,counts;
  static FrozenRaw load(const std::string &path,const NativeCurveRun &recipe) {
    torch::serialize::InputArchive archive;archive.load_from(path,torch::kCPU);FrozenRaw out{read_tensor(archive,"mean"),read_tensor(archive,"scale"),read_tensor(archive,"counts")};
    const std::vector<int64_t> shape{recipe.card.shape.channel_count,recipe.card.shape.input_width};finite_tensor(out.mean,shape,"raw mean");finite_tensor(out.scale,shape,"raw scale");finite_tensor(out.counts,shape,"raw counts");
    require(out.scale.gt(0).all().item<bool>() && out.counts.gt(0).all().item<bool>(),"raw fitted scale/count support");return out;
  }
  FeatureSurface extract(const Batch &batch) const {
    const auto safe=torch::where(batch.feature_mask,batch.data.to(torch::kFloat64),torch::zeros_like(batch.data,torch::kFloat64));
    const auto scaled=(safe-mean.unsqueeze(0).unsqueeze(2))/scale.unsqueeze(0).unsqueeze(2);
    require(torch::isfinite(scaled.masked_select(batch.feature_mask)).all().item<bool>(),"retained raw scaling overflow");
    const auto data=torch::where(batch.feature_mask,scaled,torch::zeros_like(scaled));
    return {torch::cat({data.flatten(1),batch.feature_mask.to(torch::kFloat64).flatten(1)},1),batch.feature_mask.flatten(1).any(1),"retained TRAIN ObservationScaler; raw values plus masks"};
  }
};
void prediction_parity(const FrozenReadout &fit,const FeatureSurface &surface,const ControlledDataset &validation,const std::string &path) {
  const auto actual=fit.predict(surface);torch::serialize::InputArchive archive;archive.load_from(path,torch::kCPU);
  require(torch::equal(actual.valid,read_tensor(archive,"valid")) && torch::equal(actual.ridge,read_tensor(archive,"ridge")) && torch::equal(actual.tiny,read_tensor(archive,"tiny_secondary")) &&
      torch::equal(validation.labels,read_tensor(archive,"labels_scoring_only")) && SourceParser(archive::tensor_text(read_tensor(archive,"source_ids_json"))).parse()==validation.source_ids,"loaded readout differs from original VALIDATION predictions/row lineage");
}
void save_surface(const fs::path &path,const FeatureSurface &surface,const ControlledDataset &split) {
  torch::serialize::OutputArchive out;out.write("features",surface.values,true);out.write("valid",surface.valid,true);out.write("provenance",archive::text_tensor(surface.provenance),true);
  out.write("labels_scoring_only",split.labels,true);out.write("source_ids_json",archive::text_tensor(strings(split.source_ids)),true);save_archive(path,out);
}
void save_predictions(const fs::path &path,const StressPredictions &prediction,const ControlledDataset &split) {
  torch::serialize::OutputArchive out;out.write("ridge",prediction.ridge,true);out.write("tiny_secondary",prediction.tiny,true);out.write("valid",prediction.valid,true);
  out.write("labels_scoring_only",split.labels,true);out.write("source_ids_json",archive::text_tensor(strings(split.source_ids)),true);save_archive(path,out);
}
std::vector<FrozenReadout> fit_candidate(const FeatureSurface &training,const FeatureSurface &validation,const ControlledDataset &train_split,const ControlledDataset &val_split,
                                       const NativeCurveRun &recipe,const fs::path &directory) {
  require(fit_supported(training,train_split.labels),"candidate fixed-budget TRAIN support cannot fit both classes");
  FeatureNormalizer outer(training);const auto transformed=outer.transform(training);std::vector<FrozenReadout> fits;
  for(const auto &rep:recipe.repetitions) {
    const auto seed=stream_seed(rep.probe_seed,uint64_t(recipe.export_width));RidgeProbe ridge(transformed,train_split.labels,recipe.ridge_penalty);TinyProbe tiny(transformed,train_split.labels,seed,recipe.tiny_steps,recipe.tiny_hidden,recipe.tiny_learning_rate);
    FrozenReadout fit;fit.feature_mean=outer.mean.clone();fit.feature_scale=outer.scale.clone();fit.fitted_rows=outer.fitted_rows;fit.input_width=recipe.export_width;fit.probe_width=recipe.export_width;fit.seed=seed;
    fit.ridge_mean=ridge.normalizer.mean.clone();fit.ridge_scale=ridge.normalizer.scale.clone();fit.ridge_weights=ridge.weights.clone();fit.ridge_intercept=ridge.intercept.clone();fit.tiny_mean=tiny.normalizer.mean.clone();fit.tiny_scale=tiny.normalizer.scale.clone();fit.w1=tiny.w1.clone();fit.b1=tiny.b1.clone();fit.w2=tiny.w2.clone();fit.b2=tiny.b2.clone();
    const auto root=directory/rep.id;require(fs::create_directory(root),"candidate fit repetition exists");fit.save(root/"fit.pt");
    save_predictions(root/"training-predictions.pt",fit.predict(training),train_split);save_predictions(root/"validation-predictions.pt",fit.predict(validation),val_split);fits.push_back(std::move(fit));
  }
  write_text(directory/"fit-counts.json","{\"outer_train_normalizer_fits\":1,\"post_encoder_pca_fits\":0,\"ridge_fits\":"+std::to_string(fits.size())+",\"tiny_fits\":"+std::to_string(fits.size())+",\"validation_test_stress_fits\":0}");return fits;
}
struct Method {
  std::string name,label;int64_t width;
  FeatureSurface training,validation;
  std::vector<FrozenReadout> fits;
  std::vector<std::vector<torch::Tensor>> witnesses;
  std::vector<StressPredictions> validation_predictions;
  std::vector<std::string> fit_paths;
};
struct Cohort {
  RetainedPoolingCohort retained;ControlledDataset training,validation;ProviderFitInput metadata;fs::path directory;
  FrozenRaw raw;CurveSnapshot reference,reference_initial,candidate,candidate_initial;
  CurveProgress progress;std::vector<Method> methods;
};
ProviderFitInput legal_fit(const ProviderFitInput &input) {
  auto copy=input;copy.training_observations=legal_clone(input.training_observations);return copy;
}
void same_export(const FeatureSurface &actual,const FeatureSurface &saved,const std::string &label) {
  const auto rows=saved.valid.nonzero().flatten();
  require(torch::equal(actual.valid,saved.valid) && actual.values.sizes()==saved.values.sizes() &&
      torch::equal(actual.values.to(torch::kFloat64).index_select(0,rows),saved.values.to(torch::kFloat64).index_select(0,rows)),label);
}
void retain_witnesses(Method &method) {
  for(const auto &fit:method.fits){method.witnesses.push_back(fit.witness());method.validation_predictions.push_back(fit.predict(method.validation));}
}
void check_fits(const Cohort &cohort) {
  for(const auto &method:cohort.methods)for(size_t i=0;i<method.fits.size();++i) {
    method.fits[i].unchanged(method.witnesses[i]);const auto current=method.fits[i].predict(method.validation),&old=method.validation_predictions[i];
    require(torch::equal(current.valid,old.valid) && torch::equal(current.ridge,old.ridge) && torch::equal(current.tiny,old.tiny),"retained validation predictions changed");
  }
}
void add_loaded(Cohort &cohort,const NativeCurveRun &recipe,const std::string &name,const std::string &label,int64_t width,
                const FeatureSurface &training,const FeatureSurface &validation,const std::vector<std::string> &paths,
                const std::vector<std::string> &predictions,bool pca=false) {
  Method method{name,label,width,training,validation,{},{},{},paths};
  for(size_t i=0;i<paths.size();++i) {
    auto fit=FrozenReadout::load(paths[i],training.values.size(1),width,pca,recipe,stream_seed(recipe.repetitions[i].probe_seed,uint64_t(width)),training.valid.sum().item<int64_t>());
    prediction_parity(fit,validation,cohort.validation,predictions[i]);method.fits.push_back(std::move(fit));
  }
  retain_witnesses(method);cohort.methods.push_back(std::move(method));
}
void register_inputs(const RetainedPoolingCohort &cohort,InputGuard &guard) {
  for(const auto &path:{cohort.training_observations,cohort.validation_observations,cohort.reference_checkpoint,cohort.reference_initial_checkpoint,
      cohort.reference_training_features,cohort.reference_validation_features,cohort.reference_initial_training_features,cohort.reference_initial_validation_features,cohort.raw_scaler}) {
    const auto found=cohort.expected_sha256.find(path);guard.add(path,found==cohort.expected_sha256.end()?"":found->second);
  }
  for(const auto *paths:{&cohort.reference_fits,&cohort.reference_initial_fits,&cohort.raw_fits,&cohort.pca_fits,&cohort.mask_fits,
      &cohort.reference_validation_predictions,&cohort.reference_initial_validation_predictions,&cohort.raw_validation_predictions,&cohort.pca_validation_predictions,&cohort.mask_validation_predictions})
    for(const auto &path:*paths){const auto found=cohort.expected_sha256.find(path);guard.add(path,found==cohort.expected_sha256.end()?"":found->second);}
  // Explicitly pinned companions consumed by the encoder adapter are protected
  // too. The shared engine never discovers checkpoint directories or TEST files.
  for(const auto &[path,sha]:cohort.expected_sha256)guard.add(path,sha);
}
void validate_run(const PairedPoolingRun &run,const NamedCurveFactory &candidate,const RetainedCurveSnapshotLoader &loader,const PoolingInitializationAudit &audit) {
  const auto &recipe=run.recipe;const auto &card=recipe.card;const auto &shape=card.shape;
  require(safe_name(run.protocol_id) && run.fresh_test_namespace==run.protocol_id+"/fresh-testing" &&
      run.fresh_test_stream!=0,"paired protocol/fresh-testing namespace/stream contract");
  require(safe_name(run.training_protocol_namespace),"unsafe or empty explicit training protocol namespace");
  require(card.version==2 && card.policy_version=="1.2" && card.stage=="development" && card.tasks==std::vector<Task>{Task::lag_sign},"lag-only policy1.2 development card required");
  require(shape.channel_count>=2 && shape.history_length>=8 && shape.input_width>0 && shape.dtype==torch::kFloat64 && shape.device.is_cpu() &&
      shape.channel_count<=std::numeric_limits<int64_t>::max()/shape.history_length && shape.channel_count*shape.history_length<=std::numeric_limits<int64_t>::max()/shape.input_width/2,"geometry/precision overflow");
  require(recipe.export_width>0 && recipe.patch_length>0 && shape.history_length%recipe.patch_length==0 && shape.history_length/recipe.patch_length>=3 && run.completed_updates>0,"native/patch/fixed-budget geometry");
  require(card.channel_ids.size()==size_t(shape.channel_count) && std::set<int64_t>(card.channel_ids.begin(),card.channel_ids.end()).size()==card.channel_ids.size() && !card.feature_units.empty() &&
      size_t(1+std::count(card.feature_units.begin(),card.feature_units.end(),','))==size_t(shape.input_width),"channel IDs/feature units");
  require(card.train_pairs>=2 && card.validation_pairs>0 && card.test_pairs>0 && card.train_pairs<=std::numeric_limits<int64_t>::max()/2 && card.validation_pairs<=std::numeric_limits<int64_t>::max()/2 && card.test_pairs<=std::numeric_limits<int64_t>::max()/2 &&
      card.threads>0 && std::isfinite(card.sampling_interval) && card.sampling_interval>0 && std::isfinite((shape.history_length-1)*card.sampling_interval),"split/sampling/thread metadata");
  require(!card.seeds.empty() && run.cohorts.size()==card.seeds.size() && std::set<uint64_t>(card.seeds.begin(),card.seeds.end()).size()==card.seeds.size(),"cohort/master count");
  require(!recipe.repetitions.empty() && recipe.tiny_hidden>0 && recipe.tiny_steps>0 && std::isfinite(recipe.tiny_learning_rate) && recipe.tiny_learning_rate>0 &&
      std::isfinite(recipe.ridge_penalty) && recipe.ridge_penalty>0 && recipe.bootstrap_replicates>=100 && std::isfinite(recipe.huber_delta) && recipe.huber_delta>0 && std::isfinite(recipe.missing_rate) && recipe.missing_rate>=0 && recipe.missing_rate<1,"fixed head/query recipe");
  std::set<std::string> repetitions;for(const auto &rep:recipe.repetitions)require(safe_name(rep.id) && repetitions.insert(rep.id).second,"duplicate/unsafe repetition");
  require(safe_name(candidate.name) && !candidate.recipe.empty() && bool(candidate.factory) && bool(loader) && bool(audit) && !run.reference_tag.empty() && !run.candidate_tag.empty(),"incomplete paired callbacks/tags");
  std::set<uint64_t> seeds;
  for(const auto &cohort:run.cohorts) {
    require(seeds.insert(cohort.master_seed).second && std::find(card.seeds.begin(),card.seeds.end(),cohort.master_seed)!=card.seeds.end() && !cohort.lineage.empty(),"cohort master/lineage missing");
    for(const auto *paths:{&cohort.reference_fits,&cohort.reference_initial_fits,&cohort.raw_fits,&cohort.pca_fits,&cohort.mask_fits,
        &cohort.reference_validation_predictions,&cohort.reference_initial_validation_predictions,&cohort.raw_validation_predictions,&cohort.pca_validation_predictions,&cohort.mask_validation_predictions})
      require(paths->size()==recipe.repetitions.size(),"retained fit/prediction repetition count mismatch");
  }
}
std::string metrics_json(const StressPredictions &predictions,const ControlledDataset &split,const NativeCurveRun &recipe,uint64_t seed) {
  return "{\"population\":"+population_json(predictions.valid,split)+",\"ridge\":"+score_json(score(predictions.ridge,split.labels,predictions.valid))+",\"tiny_secondary\":"+score_json(score(predictions.tiny,split.labels,predictions.valid))+
      ",\"ridge_grouped_interval\":"+interval_json(grouped_accuracy_interval(predictions.ridge,split.labels,predictions.valid,split.source_ids,seed,recipe.bootstrap_replicates),recipe.bootstrap_replicates)+
      ",\"tiny_grouped_interval\":"+interval_json(grouped_accuracy_interval(predictions.tiny,split.labels,predictions.valid,split.source_ids,seed,recipe.bootstrap_replicates),recipe.bootstrap_replicates)+'}';
}
std::string pair_json(const std::string &left,const std::string &right,const std::map<std::string,StressPredictions> &predictions,const ControlledDataset &split,const NativeCurveRun &recipe,uint64_t seed) {
  const auto &candidate=predictions.at(left),&reference=predictions.at(right);const auto common=candidate.valid.logical_and(reference.valid);
  return "{\"candidate\":"+quote(left)+",\"comparator\":"+quote(right)+",\"common_population\":"+population_json(common,split)+
      ",\"ridge_candidate_minus_comparator_grouped_interval\":"+interval_json(grouped_accuracy_interval(candidate.ridge,split.labels,common,split.source_ids,seed,recipe.bootstrap_replicates,reference.ridge),recipe.bootstrap_replicates)+
      ",\"tiny_candidate_minus_comparator_grouped_interval\":"+interval_json(grouped_accuracy_interval(candidate.tiny,split.labels,common,split.source_ids,seed,recipe.bootstrap_replicates,reference.tiny),recipe.bootstrap_replicates)+'}';
}
const std::vector<std::pair<std::string,std::string>> kPairs{{"candidate","reference"},{"candidate","candidate_initial"},
    {"reference","reference_initial"},{"candidate","pca_only"},{"reference","pca_only"}};
EvaluationCard fixed_stress_card(const EvaluationCard &input) {
  auto card=input;card.comparisons.clear();
  for(const auto &[left,right]:kPairs)card.comparisons.push_back({left+"_vs_"+right,left,right,DimensionTier::native});
  return card;
}
std::string card_json(const PairedPoolingRun &run,const NamedCurveFactory &candidate) {
  const auto &r=run.recipe;const auto &c=r.card;std::ostringstream out;out << std::setprecision(17)
      << "{\"protocol\":" << quote(run.protocol_id) << ",\"policy_version\":\"1.2\",\"stage\":\"development\",\"reference_tag\":" << quote(run.reference_tag)
      << ",\"candidate_tag\":" << quote(run.candidate_tag) << ",\"candidate_recipe\":" << quote(candidate.recipe);
  if(run.training_protocol_namespace!="native-curve-v1")out << ",\"training_protocol_namespace\":" << quote(run.training_protocol_namespace);
  out << ",\"source_fingerprint\":" << quote(r.source_fingerprint) << ",\"git_head\":" << quote(r.git_head) << ",\"git_dirty\":" << quote(r.git_dirty)
      << ",\"masters\":" << numbers(c.seeds) << ",\"task\":\"lag_sign\",\"shape\":[" << c.shape.channel_count << ',' << c.shape.history_length << ',' << c.shape.input_width
      << "],\"channel_ids\":" << numbers(c.channel_ids) << ",\"feature_units\":" << quote(c.feature_units) << ",\"sampling_interval\":" << c.sampling_interval
      << ",\"train_pairs\":" << c.train_pairs << ",\"validation_pairs\":" << c.validation_pairs << ",\"test_pairs\":" << c.test_pairs << ",\"threads\":" << c.threads
      << ",\"candidate_milestones\":[0," << run.completed_updates << "],\"completed_updates\":" << run.completed_updates << ",\"patch_length\":" << r.patch_length
      << ",\"export_width\":" << r.export_width << ",\"post_encoder_pca\":false,\"fresh_test_namespace\":" << quote(run.fresh_test_namespace) << ",\"fresh_test_stream\":" << quote(std::to_string(run.fresh_test_stream))
      << ",\"primary_cases\":[\"intact\",\"random_dropout_030\"],\"primary_pair\":\"candidate minus reference\",\"primary_head\":\"ridge\",\"missing_rate\":" << r.missing_rate
      << ",\"fixed_budget_no_selection\":true,\"old_test_access\":false,\"reused_validation_is_diagnostic\":true,\"retained_readout_refits\":0,\"retained_transform_refits\":0"
      << ",\"candidate_fit\":\"TRAIN-only exact native export; outer normalizer once per checkpoint; heads paired by declared seed and width; no encoder PCA\""
      << ",\"ridge_penalty\":" << r.ridge_penalty << ",\"tiny_hidden\":" << r.tiny_hidden << ",\"tiny_steps\":" << r.tiny_steps << ",\"tiny_learning_rate\":" << r.tiny_learning_rate
      << ",\"bootstrap_replicates\":" << r.bootstrap_replicates << ",\"confidence\":0.95,\"uncertainty\":\"within-master source-pair bootstrap conditional on one checkpoint/head; no across-master CI\""
      << ",\"raw_oracle_gate\":0.95,\"stress_sweep\":" << (r.stress_sweep?"true":"false") << ",\"reconstruction_huber_delta\":" << r.huber_delta
      << ",\"observation_archive_clean_placeholder\":\"legal observation clone for schema validation only; hidden clean signals unavailable and unused\",\"repetitions\":[";
  for(size_t i=0;i<r.repetitions.size();++i){if(i)out << ',';
    out << "{\"id\":" << quote(r.repetitions[i].id) << ",\"seed\":" << quote(std::to_string(r.repetitions[i].probe_seed)) << '}';}
  out << "],\"cohorts\":[";
  for(size_t i=0;i<run.cohorts.size();++i) {
    if(i)out << ',';
    const auto &v=run.cohorts[i];out << "{\"master\":" << quote(std::to_string(v.master_seed)) << ",\"lineage\":" << quote(v.lineage)
        << ",\"training\":" << quote(v.training_observations) << ",\"validation\":" << quote(v.validation_observations)
        << ",\"reference_checkpoint\":" << quote(v.reference_checkpoint) << ",\"reference_initial_checkpoint\":" << quote(v.reference_initial_checkpoint)
        << ",\"expected_sha256\":" << fields(v.expected_sha256) << '}';
  }
  out << "],\"comparisons\":[";
  for(size_t i=0;i<kPairs.size();++i) {
    if(i)out << ',';
    out << "{\"id\":" << quote(kPairs[i].first+"_vs_"+kPairs[i].second) << ",\"candidate\":" << quote(kPairs[i].first)
        << ",\"comparator\":" << quote(kPairs[i].second) << ",\"tier\":\"native\"}";
  }
  return out.str()+"]}";
}
std::string validation_json(const Cohort &cohort,const NativeCurveRun &recipe) {
  std::ostringstream out;out << "{\"master\":" << quote(std::to_string(cohort.retained.master_seed)) << ",\"progress\":" << progress_json(cohort.progress)
      << ",\"retained_readout_refits\":0,\"retained_transform_refits\":0,\"original_validation_prediction_parity\":true,\"repetitions\":[";
  for(size_t i=0;i<recipe.repetitions.size();++i) {
    if(i)out << ',';
    out << "{\"id\":" << quote(recipe.repetitions[i].id) << ",\"methods\":[";std::map<std::string,StressPredictions> predictions;
    for(size_t m=0;m<cohort.methods.size();++m) {
      if(m)out << ',';
      const auto &method=cohort.methods[m];const auto value=method.fits[i].predict(method.validation);predictions.emplace(method.name,value);
      out << "{\"method\":" << quote(method.name) << ",\"label\":" << quote(method.label) << ",\"size\":" << method.width << ",\"fit_artifact\":" << quote(method.fit_paths[i])
          << ",\"metrics\":" << metrics_json(value,cohort.validation,recipe,stream_seed(cohort.retained.master_seed,named_stream(method.name+"/"+recipe.repetitions[i].id))) << '}';
    }
    out << "],\"pairs\":[";
    for(size_t p=0;p<kPairs.size();++p){if(p)out << ',';out << pair_json(kPairs[p].first,kPairs[p].second,predictions,cohort.validation,recipe,stream_seed(cohort.retained.master_seed,named_stream("validation/"+kPairs[p].first+"/"+kPairs[p].second)));}
    out << "]}";
  }
  return out.str()+"]}";
}
FeatureSurface extract_method(const std::string &name,const Cohort &cohort,const Batch &batch,const NativeCurveRun &recipe) {
  if(name=="reference")return global_surface(cohort.reference,batch,recipe);
  if(name=="reference_initial")return global_surface(cohort.reference_initial,batch,recipe);
  if(name=="candidate")return global_surface(cohort.candidate,batch,recipe);
  if(name=="candidate_initial")return global_surface(cohort.candidate_initial,batch,recipe);
  if(name=="mask_metadata")return mask_surface(batch);
  require(name=="raw" || name=="pca_only","unknown paired method");return cohort.raw.extract(batch);
}
void verify_snapshots(Cohort &cohort,const NativeCurveRun &recipe) {
  const MeasurementIsolation isolation;
  for(const auto &method:cohort.methods) {
    if(method.name=="raw" || method.name=="pca_only" || method.name=="mask_metadata")continue;
    same_export(extract_method(method.name,cohort,cohort.validation.observed,recipe),method.validation,"retained snapshot changed before TEST");
    const CurveSnapshot *snapshot=method.name=="reference"?&cohort.reference:method.name=="reference_initial"?&cohort.reference_initial:
        method.name=="candidate"?&cohort.candidate:&cohort.candidate_initial;
    const auto original=cohort.directory/(method.name+"-validation-reconstruction.pt"),witness=cohort.directory/(method.name+"-validation-reconstruction-witness.pt");
    reconstruction(witness,cohort.validation,*snapshot,recipe);torch::serialize::InputArchive a,b;a.load_from(original.string(),torch::kCPU);b.load_from(witness.string(),torch::kCPU);
    for(const auto &key:{"standardized_prediction","standardized_target","target_mask","visible_mask","trial_channel_eligible"})
      require(torch::equal(read_tensor(a,key),read_tensor(b,key)),"later training changed retained decoder/support witness");
  }
  check_fits(cohort);
}
std::string test_cohort(Cohort &cohort,const PairedPoolingRun &run,const fs::path &output,std::set<std::string> &universe,std::ostringstream &stress_report,bool &first_stress) {
  const MeasurementIsolation isolation;const auto &r=run.recipe;const auto seed=cohort.retained.master_seed;const auto fresh=stream_seed(seed,run.fresh_test_stream);
  const auto testing=make_controlled_test_dataset(Task::lag_sign,r.card.shape,r.card.test_pairs,fresh,r.missing_rate);validate_split(testing,r,r.card.test_pairs,universe);
  save_split(cohort.directory/"controlled-testing.pt",testing);write_text(cohort.directory/"testing-manifest.json","{\"fresh_test_seed\":"+quote(std::to_string(fresh))+",\"fixed_budget_manifest\":\"../comparison-manifest.json\",\"testing\":"+split_manifest(testing,"testing")+'}');
  const auto oracle=raw_oracle(Task::lag_sign,testing.observed);const auto oracle_score=score(oracle.predictions,testing.labels,oracle.valid);
  require(oracle_score.supported && oracle_score.accuracy>=.95,"fresh legal raw oracle failed solvability gate");
  std::map<std::string,FeatureSurface> surfaces;
  for(const auto &method:cohort.methods){auto surface=extract_method(method.name,cohort,testing.observed,r);save_surface(cohort.directory/(method.name+"-testing.pt"),surface,testing);surfaces.emplace(method.name,std::move(surface));}
  const auto candidate_error=reconstruction(cohort.directory/"candidate-testing-reconstruction.pt",testing,cohort.candidate,r);
  const auto reference_error=reconstruction(cohort.directory/"reference-testing-reconstruction.pt",testing,cohort.reference,r);
  std::ostringstream report;report << "{\"master\":" << quote(std::to_string(seed)) << ",\"fresh_test_seed\":" << quote(std::to_string(fresh))
      << ",\"completed_updates\":" << run.completed_updates << ",\"candidate_progress\":" << progress_json(cohort.progress) << ",\"raw_oracle\":" << score_json(oracle_score)
      << ",\"candidate_reconstruction\":" << candidate_error << ",\"reference_reconstruction\":" << reference_error << ",\"repetitions\":[";
  for(size_t i=0;i<r.repetitions.size();++i) {
    if(i)report << ',';
    const auto directory=cohort.directory/("testing-"+r.repetitions[i].id);require(fs::create_directory(directory),"test repetition directory exists");
    std::map<std::string,StressPredictions> predictions;StressReadouts readouts;std::map<std::string,SurfaceDescription> descriptions;
    report << "{\"id\":" << quote(r.repetitions[i].id) << ",\"methods\":[";
    for(size_t m=0;m<cohort.methods.size();++m) {
      if(m)report << ',';
      const auto &method=cohort.methods[m];const auto &fit=method.fits[i];fit.unchanged(method.witnesses[i]);
      const auto prediction=fit.predict(surfaces.at(method.name));predictions.emplace(method.name,prediction);save_predictions(directory/(method.name+"-predictions.pt"),prediction,testing);
      report << "{\"method\":" << quote(method.name) << ",\"label\":" << quote(method.label) << ",\"size\":" << method.width << ",\"status\":\"measured\",\"fit_artifact\":" << quote(method.fit_paths[i])
          << ",\"ridge_parameters\":" << 2*method.width+2 << ",\"neural_parameters\":" << r.tiny_hidden*(method.width+3)+2
          << ",\"metrics\":" << metrics_json(prediction,testing,r,stream_seed(seed,named_stream(method.name+"/"+r.repetitions[i].id))) << '}';
      FrozenStressReadout frozen;frozen.status="measured";frozen.probe_dimensions=method.width;frozen.native_valid=prediction.valid.clone();frozen.native_ridge=prediction.ridge.clone();frozen.native_tiny=prediction.tiny.clone();
      const auto *fit_pointer=&fit;const auto witness=method.witnesses[i];
      frozen.predict=[fit_pointer,witness,prediction,first=true](const FeatureSurface &surface) mutable {
        fit_pointer->unchanged(witness);const auto value=fit_pointer->predict(surface);fit_pointer->unchanged(witness);
        if(first){require(torch::equal(value.valid,prediction.valid) && torch::equal(value.ridge,prediction.ridge) && torch::equal(value.tiny,prediction.tiny),"intact stress differs from frozen ordinary predictions");first=false;}return value;
      };
      readouts.emplace(std::make_pair(method.name,DimensionTier::native),std::move(frozen));
      descriptions.emplace(method.name,SurfaceDescription{method.name=="raw" || method.name=="pca_only" || method.name=="mask_metadata"?SurfaceKind::control:SurfaceKind::global,
          method.name=="mask_metadata"?"mask metadata, all rows including allmissing":"any observed coordinate; no inferred missing signal",{}});
    }
    report << "],\"pairs\":[";
    for(size_t p=0;p<kPairs.size();++p){if(p)report << ',';report << pair_json(kPairs[p].first,kPairs[p].second,predictions,testing,r,stream_seed(seed,named_stream("test/"+kPairs[p].first+"/"+kPairs[p].second+"/"+r.repetitions[i].id)));}
    report << "]}";
    if(r.stress_sweep) {
      const auto card=fixed_stress_card(r.card);
      const ControlledProtocol protocol{Task::lag_sign,seed,r.card.shape,cohort.training,cohort.validation,testing};
      const auto extractor=[&cohort,&r](const Batch &batch){FeatureMap out;for(const auto &method:cohort.methods)out.emplace(method.name,extract_method(method.name,cohort,batch,r));return out;};
      const auto stress=run_fixed_readout_stress(directory.string(),card,protocol,descriptions,readouts,extractor);
      if(!first_stress)stress_report << ',';
      first_stress=false;
      stress_report << "{\"master\":" << quote(std::to_string(seed)) << ",\"repetition\":" << quote(r.repetitions[i].id) << ",\"directory\":" << quote(fs::relative(directory/"stress",output).generic_string()) << ",\"report\":" << stress << '}';
    }
    check_fits(cohort);
  }
  return report.str()+"]}";
}
void validate_native_geometry(const NativeCurveRun &recipe) {
  const auto &shape=recipe.card.shape;
  require(shape.channel_count>0 && shape.history_length>0 && shape.input_width>0 &&
      shape.dtype==torch::kFloat64 && shape.device.is_cpu() &&
      shape.channel_count<=std::numeric_limits<int64_t>::max()/shape.history_length &&
      shape.channel_count*shape.history_length<=std::numeric_limits<int64_t>::max()/shape.input_width &&
      recipe.export_width>0 && recipe.card.threads>0,"native measurement geometry/precision/thread contract");
}
void validate_native_batch(const Batch &batch,const NativeCurveRun &recipe) {
  const auto &shape=recipe.card.shape;
  require(batch.data.defined() && batch.data.device().is_cpu() &&
      batch.data.scalar_type()==torch::kFloat64 && batch.data.dim()==4 && batch.data.size(0)>0 &&
      batch.data.size(1)==shape.channel_count && batch.data.size(2)==shape.history_length &&
      batch.data.size(3)==shape.input_width && batch.feature_mask.defined() &&
      batch.feature_mask.device().is_cpu() && batch.feature_mask.scalar_type()==torch::kBool &&
      batch.feature_mask.sizes()==batch.data.sizes() &&
      torch::isfinite(batch.data.masked_select(batch.feature_mask)).all().item<bool>(),
      "native measurement requires configured CPU float64 observations and bool masks");
}
void validate_native_split(const ControlledDataset &split,const NativeCurveRun &recipe) {
  validate_native_geometry(recipe);validate_native_batch(split.observed,recipe);
  const auto rows=split.observed.data.size(0);
  require(rows%2==0,"native development split must contain complete source pairs");
  std::set<std::string> sources;validate_split(split,recipe,rows/2,sources);
}
void validate_new_archive(const std::string &destination) {
  const fs::path path(destination);
  const auto parent=path.has_parent_path()?path.parent_path():fs::path(".");
  require(!path.empty() && !path.filename().empty() && !fs::exists(path) && fs::is_directory(parent),
      "native measurement archive requires a new destination in an existing directory");
}
} // namespace

ControlledDataset load_native_development_observations(const std::string &path) {
  const MeasurementIsolation isolation;
  require(!path.empty() && fs::is_regular_file(path),"native development observation archive missing");
  auto split=load_observations(path);
  require(split.observed.data.defined() && split.observed.data.dim()==4,
      "native development observations must be BCHF");
  NativeCurveRun recipe;
  recipe.card.shape={split.observed.data.size(1),split.observed.data.size(2),
      split.observed.data.size(3),torch::kFloat64,torch::kCPU};
  validate_native_split(split,recipe);return split;
}
FeatureSurface extract_native_global(const CurveSnapshot &snapshot,const Batch &batch,
                                     const NativeCurveRun &recipe) {
  validate_native_geometry(recipe);validate_native_batch(batch,recipe);
  const ThreadsIsolation threads;const MeasurementIsolation isolation;
  torch::set_num_threads(recipe.card.threads);return global_surface(snapshot,batch,recipe);
}
void save_native_feature_archive(const std::string &path,const FeatureSurface &surface,
                                 const ControlledDataset &split,const NativeCurveRun &recipe) {
  validate_new_archive(path);validate_native_split(split,recipe);validate_features(surface);
  require(surface.values.sizes()==torch::IntArrayRef({split.observed.data.size(0),recipe.export_width}) &&
      !surface.provenance.empty() &&
      !surface.valid.logical_and(split.observed.feature_mask.flatten(1).any(1).logical_not()).any().item<bool>(),
      "native feature archive width/row lineage/support mismatch");
  const MeasurementIsolation isolation;save_surface(path,surface,split);
}
std::string write_native_patch_reconstruction(const std::string &path,const ControlledDataset &split,
                                             const CurveSnapshot &snapshot,const NativeCurveRun &recipe) {
  validate_new_archive(path);validate_native_split(split,recipe);
  require(recipe.patch_length>0 && recipe.card.shape.history_length%recipe.patch_length==0 &&
      recipe.card.shape.history_length/recipe.patch_length>=3 && std::isfinite(recipe.huber_delta) &&
      recipe.huber_delta>0 && bool(snapshot.reconstruct),"native patch reconstruction geometry/Huber/callback contract");
  const ThreadsIsolation threads;const MeasurementIsolation isolation;
  torch::set_num_threads(recipe.card.threads);return reconstruction(path,split,snapshot,recipe);
}

void run_paired_pooling(const PairedPoolingRun &run,const NamedCurveFactory &candidate,const RetainedCurveSnapshotLoader &load_reference,const PoolingInitializationAudit &audit_initialization) {
  validate_run(run,candidate,load_reference,audit_initialization);const ThreadsIsolation threads;const MeasurementIsolation ambient;const auto &r=run.recipe;torch::set_num_threads(r.card.threads);
  const fs::path output(r.output_directory);require(!output.empty() && fs::create_directories(output),"output must be an exclusively claimed new directory");
  write_text(output/"paired-pooling-card.json",card_json(run,candidate));
  if(r.stress_sweep)write_text(output/"stress-card.json",fixed_readout_stress_card_json(fixed_stress_card(r.card)));
  InputGuard guard;for(const auto &cohort:run.cohorts)register_inputs(cohort,guard);write_text(output/"input-manifest.json",guard.manifest());
  std::vector<Cohort> cohorts;cohorts.reserve(run.cohorts.size());std::set<std::string> universe;
  std::ostringstream validation;validation << "{\"protocol\":" << quote(run.protocol_id) << ",\"policy_version\":\"1.2\",\"stage\":\"development\",\"fixed_budget\":" << run.completed_updates << ",\"cohorts\":[";
  bool first=true;
  for(const auto &retained:run.cohorts) {
    Cohort cohort;cohort.retained=retained;cohort.directory=output/("seed-"+std::to_string(retained.master_seed)+"-lag_sign");require(fs::create_directory(cohort.directory),"paired cohort directory exists");
    cohort.training=load_observations(retained.training_observations);cohort.validation=load_observations(retained.validation_observations);
    validate_split(cohort.training,r,r.card.train_pairs,universe);validate_split(cohort.validation,r,r.card.validation_pairs,universe);
    save_split(cohort.directory/"controlled-training.pt",cohort.training);save_split(cohort.directory/"controlled-validation.pt",cohort.validation);
    write_text(cohort.directory/"development-manifest.json","{\"testing_generated\":false,\"retained_training\":"+split_manifest(cohort.training,"training")+",\"retained_validation\":"+split_manifest(cohort.validation,"validation")+'}');
    cohort.metadata={legal_clone(cohort.training.observed),r.card.shape,retained.master_seed,cohort.training.source_ids,r.card.channel_ids,r.card.feature_units,
        run.training_protocol_namespace+"/lag_sign",r.card.sampling_interval,(r.card.shape.history_length-1)*r.card.sampling_interval};
    {
      const MeasurementIsolation measure;cohort.raw=FrozenRaw::load(retained.raw_scaler,r);
      const auto train_raw=cohort.raw.extract(cohort.training.observed),val_raw=cohort.raw.extract(cohort.validation.observed);
      add_loaded(cohort,r,"raw","Raw data — no encoder",train_raw.values.size(1),train_raw,val_raw,retained.raw_fits,retained.raw_validation_predictions);
      add_loaded(cohort,r,"pca_only","PCA only — no encoder",r.export_width,train_raw,val_raw,retained.pca_fits,retained.pca_validation_predictions,true);
      add_loaded(cohort,r,"mask_metadata","Mask metadata",r.card.shape.channel_count*r.card.shape.history_length*r.card.shape.input_width,mask_surface(cohort.training.observed),mask_surface(cohort.validation.observed),retained.mask_fits,retained.mask_validation_predictions);
      cohort.reference=load_reference(retained.reference_checkpoint,legal_fit(cohort.metadata));cohort.reference_initial=load_reference(retained.reference_initial_checkpoint,legal_fit(cohort.metadata));
      require(bool(cohort.reference.reconstruct) && bool(cohort.reference_initial.reconstruct),"retained reference snapshot missing decoder");
      const auto train_reference=load_features(retained.reference_training_features,cohort.training,r.export_width),val_reference=load_features(retained.reference_validation_features,cohort.validation,r.export_width);
      const auto train_initial=load_features(retained.reference_initial_training_features,cohort.training,r.export_width),val_initial=load_features(retained.reference_initial_validation_features,cohort.validation,r.export_width);
      same_export(global_surface(cohort.reference,cohort.training.observed,r),train_reference,"restored reference TRAIN export differs from original");
      same_export(global_surface(cohort.reference,cohort.validation.observed,r),val_reference,"restored reference VALIDATION export differs from original");
      same_export(global_surface(cohort.reference_initial,cohort.training.observed,r),train_initial,"restored reference initialization TRAIN export differs");
      same_export(global_surface(cohort.reference_initial,cohort.validation.observed,r),val_initial,"restored reference initialization VALIDATION export differs");
      add_loaded(cohort,r,"reference",run.reference_tag,r.export_width,train_reference,val_reference,retained.reference_fits,retained.reference_validation_predictions);
      add_loaded(cohort,r,"reference_initial",run.reference_tag+" — untrained weights",r.export_width,train_initial,val_initial,retained.reference_initial_fits,retained.reference_initial_validation_predictions);
      write_text(cohort.directory/"retained-fit-audit.json","{\"transform_fitting_constructors\":0,\"readout_fitting_constructors\":0,\"original_validation_prediction_parity\":true,\"native_export_parity\":true}");
      write_text(cohort.directory/"reference-training-reconstruction.json",reconstruction(cohort.directory/"reference-training-reconstruction.pt",cohort.training,cohort.reference,r));
      write_text(cohort.directory/"reference-validation-reconstruction.json",reconstruction(cohort.directory/"reference-validation-reconstruction.pt",cohort.validation,cohort.reference,r));
      reconstruction(cohort.directory/"reference_initial-validation-reconstruction.pt",cohort.validation,cohort.reference_initial,r);
    }
    const auto trainer=candidate.factory(legal_fit(cohort.metadata));require(bool(trainer.train_to) && bool(trainer.save_checkpoint) && bool(trainer.snapshot),"candidate trainer incomplete");
    write_text(cohort.directory/"candidate-trainer-audit.json",fields(trainer.audit_fields));CurveProgress previous;
    for(const auto budget:{int64_t(0),run.completed_updates}) {
      const auto directory=cohort.directory/("candidate-milestone-"+std::to_string(budget));require(fs::create_directory(directory),"candidate point directory exists");
      const auto progress=trainer.train_to(budget);const MeasurementIsolation measure;check_progress(progress,budget,budget?&previous:nullptr);
      // The retained reference completed this exact prefix without skips. A
      // candidate skip would change the row/mask counter draws before update512.
      if(budget)require(progress.attempted==progress.completed && progress.completed==run.completed_updates,
          "candidate skipped an attempted update; retained training prefix no longer matches");
      if(budget)require(!progress.losses.empty() && progress.losses.back().completed==budget,"candidate cumulative trace omitted fixed endpoint");
      const auto checkpoint=directory/"checkpoint.pt";trainer.save_checkpoint(checkpoint.string());require(fs::is_regular_file(checkpoint),"candidate checkpoint missing");
      auto snapshot=trainer.snapshot(checkpoint.string());require(bool(snapshot.reconstruct),"candidate immutable decoder snapshot missing");
      if(!budget) {
        const auto audit=audit_initialization(checkpoint.string(),retained,legal_fit(cohort.metadata));
        for(const auto &key:{"common_parameters_exact","scaler_exact","training_dataset_exact","counter_streams_exact"})
          require(audit.count(key) && audit.at(key)=="true",std::string("initialization/scaler/stream audit failed: ")+key);
        write_text(cohort.directory/"initialization-audit.json",fields(audit));
      }
      const auto training=global_surface(snapshot,cohort.training.observed,r),val=global_surface(snapshot,cohort.validation.observed,r);
      const auto &reference=cohort.methods[3];require(torch::equal(training.valid,reference.training.valid) && torch::equal(val.valid,reference.validation.valid),"candidate/reference ordinary support differs");
      const auto name=budget?"candidate":"candidate_initial";Method method{name,budget?run.candidate_tag:run.candidate_tag+" — untrained weights",r.export_width,training,val,{},{},{},{}};
      save_surface(directory/"native-training.pt",training,cohort.training);save_surface(directory/"native-validation.pt",val,cohort.validation);
      method.fits=fit_candidate(training,val,cohort.training,cohort.validation,r,directory);for(const auto &rep:r.repetitions)method.fit_paths.push_back((directory/rep.id/"fit.pt").string());retain_witnesses(method);cohort.methods.push_back(std::move(method));
      const auto train_error=reconstruction(cohort.directory/(std::string(name)+"-training-reconstruction.pt"),cohort.training,snapshot,r);
      const auto val_error=reconstruction(cohort.directory/(std::string(name)+"-validation-reconstruction.pt"),cohort.validation,snapshot,r);
      write_text(directory/"point.json","{\"completed_updates\":"+std::to_string(budget)+",\"progress\":"+progress_json(progress)+",\"training_reconstruction\":"+train_error+",\"validation_reconstruction\":"+val_error+'}');
      if(budget){cohort.candidate=std::move(snapshot);cohort.progress=progress;}else cohort.candidate_initial=std::move(snapshot);
      previous=progress;check_fits(cohort);
      std::cout << "paired-pooling master=" << retained.master_seed << " completed=" << budget << " training_seconds=" << progress.training_seconds << '\n' << std::flush;
    }
    verify_snapshots(cohort,r);guard.verify();
    if(!first)validation << ',';
    first=false;validation << validation_json(cohort,r);cohorts.push_back(std::move(cohort));
  }
  write_text(output/"validation-report.json",validation.str()+"]}");guard.verify();
  // This fixed budget is not selected using reused validation or fresh TEST.
  // Close+fsync file and directory make the admission boundary durable.
  const auto training_namespace=run.training_protocol_namespace=="native-curve-v1"?std::string{}:",\"training_protocol_namespace\":"+quote(run.training_protocol_namespace);
  durable_selection(output/"comparison-manifest.json","{\"protocol\":"+quote(run.protocol_id)+training_namespace+",\"policy_version\":\"1.2\",\"stage\":\"development\",\"completed_updates\":"+
      std::to_string(run.completed_updates)+",\"budget_policy\":\"predeclared fixed positive budget, no validation search\",\"all_validation_complete\":true,\"all_retained_witnesses_exact\":true,\"all_testing_after_manifest\":true,\"fresh_test_stream\":"+
      quote(std::to_string(run.fresh_test_stream))+",\"validation_report\":\"validation-report.json\",\"input_manifest\":\"input-manifest.json\",\"source_fingerprint\":"+quote(r.source_fingerprint)+'}');
  std::ostringstream report,stress;report << "{\"protocol\":" << quote(run.protocol_id) << ",\"policy_version\":\"1.2\",\"stage\":\"development\",\"card\":\"paired-pooling-card.json\",\"comparison_manifest\":\"comparison-manifest.json\",\"source_fingerprint\":" << quote(r.source_fingerprint)
      << ",\"fixed_updates\":" << run.completed_updates << ",\"reference_tag\":" << quote(run.reference_tag) << ",\"candidate_tag\":" << quote(run.candidate_tag) << ",\"retained_transform_refits\":0,\"retained_readout_refits\":0,\"testing_runs\":[";
  stress << "{\"protocol\":\"fixed-readout-stress-v1\",\"primary_cases\":[\"intact\",\"random_dropout_030\"],\"stage\":\"development\",\"runs\":[";
  bool first_test=true,first_stress=true;
  for(auto &cohort:cohorts){if(!first_test)report << ',';first_test=false;report << test_cohort(cohort,run,output,universe,stress,first_stress);guard.verify();}
  write_text(output/"report.json",report.str()+"]}");if(r.stress_sweep)write_text(output/"stress-report.json",stress.str()+"]}");
  guard.verify();write_text(output/"input-integrity-after.json","{\"original_inputs_byte_preserved\":true,\"retained_fits_immutable\":true,\"retained_readout_refits\":0,\"retained_transform_refits\":0,\"files\":"+guard.manifest()+'}');
}
} // namespace embedding::evaluation
