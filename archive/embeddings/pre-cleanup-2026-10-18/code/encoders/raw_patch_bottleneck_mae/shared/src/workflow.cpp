// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef RPB_SOURCE_ID
#define RPB_SOURCE_ID "unrecorded"
#endif
#ifndef RPB_GIT_HEAD
#define RPB_GIT_HEAD "unrecorded"
#endif
#ifndef RPB_GIT_DIRTY
#define RPB_GIT_DIRTY "unknown"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {

constexpr const char *encoder_id = kEncoderId;
constexpr const char *data_kind = "rpb_raw_uniform_history_v1";
constexpr const char *scaler_kind = "rpb_frozen_training_scaler_v1";
using training_detail::mixed;
using training_detail::counter_seed;
using training_detail::sampled_indices;
using training_detail::selected;
using embedding::archive::distinct_paths;
using embedding::archive::save_archive;
using embedding::archive::text_tensor;
using embedding::archive::tensor_text;

#define MODEL_INTS(X) X(channel_count) X(history_length) X(input_width) X(patch_length) X(encoder_width) X(export_width) X(num_layers) X(num_heads) X(feedforward_width) X(decoder_hidden_width) X(channel_mixer_layers)
#define MODEL_DOUBLES(X) X(dropout) X(layer_norm_epsilon) X(mask_ratio) X(huber_delta) X(scale_floor) X(sampling_interval)
#define RUN_INTS(X) X(steps) X(batch_size) X(seed) X(threads) X(log_every) X(checkpoint_every) X(attempt_limit)
#define RUN_DOUBLES(X) X(learning_rate) X(weight_decay) X(gradient_clip_norm)

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}

std::string trim(const std::string &text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  return begin == std::string::npos ? std::string{} :
      text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

int64_t integer(const std::string &text) {
  int64_t value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc() && result.ptr == text.data() + text.size(),
          "expected integer: " + text);
  return value;
}

double real(const std::string &text) {
  size_t used{};
  const double value = std::stod(text, &used);
  require(used == text.size() && std::isfinite(value), "expected finite number: " + text);
  return value;
}

std::vector<int64_t> integer_list(const std::string &text) {
  if (text.empty()) return {};
  require(text.back() != ',', "trailing comma in channel_ids");
  std::istringstream stream(text);
  std::vector<int64_t> values;
  std::string item;
  while (std::getline(stream, item, ',')) values.push_back(integer(trim(item)));
  return values;
}

torch::Device device_from(const std::string &text) {
  require(text == "cpu" || text == "cuda", "device must be cpu or cuda");
  return torch::Device(text);
}

void activate(const Settings &settings) {
  validate_settings(settings);
  require(!settings.model.device.is_cuda() || torch::cuda::is_available(),
          "CUDA is unavailable; select cpu or the managed container GPU");
  torch::set_num_threads(static_cast<int>(settings.threads));
  torch::manual_seed(static_cast<uint64_t>(settings.seed));
}

// A content identity, not cryptographic integrity. Source provenance uses the
// separate SHA-256 fingerprint injected when this translation unit is built.
struct Fingerprint {
  uint64_t value{14695981039346656037ULL};
  void bytes(const void *data, size_t count) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < count; ++i) { value ^= p[i]; value *= 1099511628211ULL; }
  }
  void text(const std::string &s) { bytes(s.data(), s.size()); const uint8_t end=0; bytes(&end,1); }
  void tensor(const torch::Tensor &t) {
    auto cpu = t.detach().to(torch::kCPU).contiguous();
    text(std::to_string(static_cast<int>(cpu.scalar_type())));
    text(std::to_string(cpu.dim()));
    for (auto size : cpu.sizes()) text(std::to_string(size));
    bytes(cpu.data_ptr(), static_cast<size_t>(cpu.numel()) * cpu.element_size());
  }
  std::string id(const char *prefix) const {
    std::ostringstream out; out << prefix << '-' << std::hex << std::setw(16)
                                << std::setfill('0') << value; return out.str();
  }
};

std::string default_units(int64_t width) {
  std::string result;
  for (int64_t f=0; f<width; ++f) result += (f ? ",unitless" : "unitless");
  return result;
}

void validate_units(const std::string &units, int64_t width) {
  std::istringstream stream(units); std::string item; int64_t count=0;
  require(!units.empty() && units.back()!=',', "feature_units must declare each feature");
  while (std::getline(stream,item,',')) { require(!trim(item).empty(), "empty feature unit"); ++count; }
  require(count==width, "feature_units count does not match input_width");
}

void validate_raw(const Input &input, const Config &config) {
  validate_config(config);
  require(input.data.defined() && input.data.dim()==4 && input.data.size(0)>0 &&
      (input.data.scalar_type()==torch::kFloat64 || input.data.scalar_type()==torch::kFloat32),
      "raw data must be nonempty float32/float64 [B,C,H,F]");
  require(input.data.size(1)==config.channel_count && input.data.size(2)==config.history_length &&
      input.data.size(3)==config.input_width, "raw data C/H/F differs from config");
  require(input.observed.defined() && input.observed.scalar_type()==torch::kBool &&
      input.observed.sizes()==input.data.sizes(), "observed must be bool matching raw data");
  require(torch::isfinite(input.data).logical_or(input.observed.logical_not()).all().item<bool>(),
      "observed raw values contain NaN/Inf");
  require(input.channel_ids.defined() && input.channel_ids.scalar_type()==torch::kInt64 &&
      ((input.channel_ids.dim()==1 && input.channel_ids.size(0)==config.channel_count) ||
       (input.channel_ids.dim()==2 && input.channel_ids.size(0)==input.data.size(0) &&
        input.channel_ids.size(1)==config.channel_count)), "channel_ids must be int64 [C] or [B,C]");
  auto ids=input.channel_ids.to(torch::kCPU).contiguous();
  if (ids.dim()==1) ids=ids.unsqueeze(0);
  const auto known=resolved_channel_ids(config);
  const std::set<int64_t> expected(known.begin(),known.end());
  for (int64_t b=0; b<ids.size(0); ++b) {
    std::set<int64_t> seen;
    for (int64_t c=0; c<ids.size(1); ++c) seen.insert(ids[b][c].item<int64_t>());
    require(seen==expected, "unknown or duplicate semantic channel IDs");
  }
  require(input.endpoints.defined() && input.endpoints.scalar_type()==torch::kFloat64 &&
      input.endpoints.dim()==1 && input.endpoints.size(0)==input.data.size(0) &&
      torch::isfinite(input.endpoints).all().item<bool>(), "endpoints must be finite float64 [B]");
  require(std::isfinite(input.sampling_interval) && input.sampling_interval>0 &&
      std::abs(input.sampling_interval-config.sampling_interval)<=
          1e-12*std::max(1.0,std::abs(config.sampling_interval)),
      "sampling_interval differs from the declared uniform schema");
}

std::string schema_identity(const Input &input, const std::string &units) {
  Fingerprint hash; hash.text(data_kind); hash.text(units);
  for (int64_t axis=1; axis<4; ++axis) hash.text(std::to_string(input.data.size(axis)));
  hash.text(std::to_string(static_cast<int>(input.data.scalar_type())));
  hash.bytes(&input.sampling_interval,sizeof(input.sampling_interval));
  auto ids=input.channel_ids.to(torch::kCPU).contiguous();
  if (ids.dim()==2) ids=ids[0].contiguous();
  std::vector<int64_t> sorted(ids.data_ptr<int64_t>(),ids.data_ptr<int64_t>()+ids.numel());
  std::sort(sorted.begin(),sorted.end());
  for (auto id : sorted) hash.text(std::to_string(id));
  return hash.id("rpb-schema-fnv1a-v1");
}

std::string dataset_identity(const Dataset &dataset) {
  Fingerprint hash; hash.text(schema_identity(dataset.input,dataset.feature_units));
  hash.tensor(dataset.input.data); hash.tensor(dataset.input.observed);
  hash.tensor(dataset.input.channel_ids); hash.tensor(dataset.input.endpoints);
  return hash.id("rpb-dataset-fnv1a-v1");
}

void write_text(torch::serialize::OutputArchive &archive,const char *key,const std::string &text) {
  archive.write(key,text_tensor(text),true);
}

std::string read_text(torch::serialize::InputArchive &archive,const char *key) {
  torch::Tensor value; archive.read(key,value,true); return tensor_text(value);
}

int64_t read_count(torch::serialize::InputArchive &archive,const char *key) {
  torch::Tensor value; archive.read(key,value,true);
  require(value.scalar_type()==torch::kInt64 && value.numel()==1 && value.item<int64_t>()>=0,
          std::string("invalid counter: ")+key); return value.item<int64_t>();
}

void check_envelope(torch::serialize::InputArchive &archive,const char *kind) {
  torch::Tensor id;
  require(archive.try_read("encoder_id",id,true),
          "archive lacks RPB-MAE encoder_id; baseline and untagged archives cannot be loaded");
  require(tensor_text(id)==encoder_id,"archive encoder_id mismatch");
  require(read_count(archive,"format_version")==1,"unsupported RPB-MAE format_version");
  require(read_text(archive,"artifact_kind")==kind,"archive artifact_kind mismatch");
}

void envelope(torch::serialize::OutputArchive &archive,const char *kind) {
  write_text(archive,"encoder_id",encoder_id);
  archive.write("format_version",torch::tensor(int64_t{1}),true);
  write_text(archive,"artifact_kind",kind);
}

using Arguments=std::map<std::string,std::string>;
Arguments arguments(int argc,char **argv,const std::set<std::string> &allowed) {
  Arguments result;
  for (int i=2;i<argc;i+=2) {
    const std::string key(argv[i]); require(allowed.count(key),"unknown option: "+key);
    require(i+1<argc,"missing value for "+key);
    require(result.emplace(key,argv[i+1]).second,"duplicate option: "+key);
  }
  return result;
}
std::string optional(const Arguments &args,const std::string &key,const std::string &fallback={}) {
  const auto it=args.find(key); return it==args.end()?fallback:it->second;
}
std::string required(const Arguments &args,const std::string &key) {
  const auto value=optional(args,key); require(!value.empty(),"required option: "+key); return value;
}
void protect_inputs(const Arguments &args,const std::string &output,const std::set<std::string> &keys) {
  for (const auto &key:keys) {
    const auto input=optional(args,key);distinct_paths(input,output);
    if(!input.empty() && std::filesystem::exists(input) && std::filesystem::exists(output))
      require(!std::filesystem::equivalent(input,output),"input and output paths alias the same file");
  }
}
void usage() {
  std::cout << R"(Usage:
  embedding_raw_patch_bottleneck_mae synthetic --output raw.pt [--config smoke.conf]
      [--samples 32] [--seed 101] [--units unitless,unitless,unitless]
  embedding_raw_patch_bottleneck_mae prepare --input raw.pt --output scaler.pt [--config smoke.conf]
  embedding_raw_patch_bottleneck_mae train --checkpoint model.pt [--input raw.pt] [--config smoke.conf]
      [--scaler scaler.pt] [--steps 8] [--attempt-limit 1000] [--batch-size 4]
      [--device cpu|cuda] [--seed 101] [--checkpoint-every 0]
  embedding_raw_patch_bottleneck_mae train --resume model.pt --checkpoint continued.pt
      [--input raw.pt] [--steps 8] [--attempt-limit 1000] [--device cpu|cuda]
  embedding_raw_patch_bottleneck_mae embed --checkpoint model.pt --input raw.pt --output embeddings.pt
      [--batch-size 4] [--device cpu|cuda]
  embedding_evaluate --help

--steps requests additional completed updates; skipped batches consume attempted
batches and the separate attempt limit. Periodic saves count attempted batches.
Resume requires the same raw dataset and training batch size. Without --input,
train uses a deterministic 32-example float64 fixture. Raw archives require this
encoder's versioned precision, semantic-ID, endpoint and uniform-interval schema.
Preparation fits a frozen training-only scaler; do not fit it on validation/test.
)";
}

} // namespace

Settings default_settings() { return {}; }

Dataset describe_dataset(const Input &input,const Config &config,const std::string &feature_units) {
  validate_raw(input,config);
  Dataset dataset{input,feature_units.empty()?default_units(config.input_width):feature_units,{}, {}};
  validate_units(dataset.feature_units,config.input_width);
  dataset.schema_id=schema_identity(input,dataset.feature_units);
  dataset.dataset_id=dataset_identity(dataset);
  return dataset;
}

std::string workflow_source_fingerprint() { return RPB_SOURCE_ID; }

void validate_settings(const Settings &settings) {
  validate_config(settings.model);
  require(settings.model.dtype==torch::kFloat32,"workflow models require float32");
  require(settings.steps>0 && settings.batch_size>0 && settings.threads>0 &&
      settings.threads<=std::numeric_limits<int>::max() && settings.log_every>0 &&
      settings.attempt_limit>0,"steps/batch_size/threads/log_every/attempt_limit must be positive");
  require(settings.seed>=0 && settings.checkpoint_every>=0,"seed/checkpoint_every must be nonnegative");
  require(std::isfinite(settings.learning_rate) && settings.learning_rate>0 &&
      std::isfinite(settings.weight_decay) && settings.weight_decay>=0 &&
      std::isfinite(settings.gradient_clip_norm) && settings.gradient_clip_norm>=0,
      "learning_rate must be positive; weight_decay/gradient_clip_norm finite nonnegative");
}

Settings parse_settings(const std::string &text) {
  auto settings=default_settings(); std::istringstream stream(text);
  std::string line; std::set<std::string> seen; int64_t number=0;
  while (std::getline(stream,line)) {
    ++number; line=trim(line.substr(0,line.find('#'))); if (line.empty()) continue;
    try {
      const auto separator=line.find('='); require(separator!=std::string::npos,"expected key=value");
      const auto key=trim(line.substr(0,separator)),value=trim(line.substr(separator+1));
      require(seen.insert(key).second,"duplicate key: "+key);
#define PARSE_MODEL_INT(name) if(key==#name){settings.model.name=integer(value);continue;}
      MODEL_INTS(PARSE_MODEL_INT)
#undef PARSE_MODEL_INT
      if(key=="global_bottleneck_mode"){settings.model.global_bottleneck_mode=integer(value);continue;}
      if(key=="channel_mixer_placement"){settings.model.channel_mixer_placement=integer(value);continue;}
      if(key=="global_pool_input_source"){settings.model.global_pool_input_source=integer(value);continue;}
      if(key=="temporal_difference_input"){settings.model.temporal_difference_input=integer(value);continue;}
#define PARSE_MODEL_DOUBLE(name) if(key==#name){settings.model.name=real(value);continue;}
      MODEL_DOUBLES(PARSE_MODEL_DOUBLE)
#undef PARSE_MODEL_DOUBLE
#define PARSE_RUN_INT(name) if(key==#name){settings.name=integer(value);continue;}
      RUN_INTS(PARSE_RUN_INT)
#undef PARSE_RUN_INT
#define PARSE_RUN_DOUBLE(name) if(key==#name){settings.name=real(value);continue;}
      RUN_DOUBLES(PARSE_RUN_DOUBLE)
#undef PARSE_RUN_DOUBLE
      if(key=="channel_ids") settings.model.channel_ids=integer_list(value);
      else if(key=="device") settings.model.device=device_from(value);
      else throw std::runtime_error("unknown key: "+key);
    } catch(const std::exception &error) {
      throw std::runtime_error("config line "+std::to_string(number)+": "+error.what());
    }
  }
  validate_settings(settings); return settings;
}

Settings read_settings(const std::string &path) {
  std::ifstream file(path); require(file.good(),"cannot open config: "+path);
  std::ostringstream out; out<<file.rdbuf(); require(!file.bad(),"config read failed: "+path);
  return parse_settings(out.str());
}

std::string settings_text(const Settings &settings) {
  validate_settings(settings); std::ostringstream out; out<<std::setprecision(17);
#define WRITE_MODEL(name) out<<#name "="<<settings.model.name<<'\n';
  MODEL_INTS(WRITE_MODEL)
  MODEL_DOUBLES(WRITE_MODEL)
#undef WRITE_MODEL
  // A missing mode means legacy behavior. Keep archived mode-0 settings text
  // and its content/configuration identity byte-for-byte unchanged.
  if(settings.model.global_bottleneck_mode>0)
    out<<"global_bottleneck_mode="<<settings.model.global_bottleneck_mode<<'\n';
  // Placement zero is the historical path; preserve its canonical text/IDs.
  if(settings.model.channel_mixer_placement>0)
    out<<"channel_mixer_placement="<<settings.model.channel_mixer_placement<<'\n';
  if(settings.model.global_pool_input_source>0)
    out<<"global_pool_input_source="<<settings.model.global_pool_input_source<<'\n';
  if(settings.model.temporal_difference_input>0)
    out<<"temporal_difference_input="<<settings.model.temporal_difference_input<<'\n';
#define WRITE_RUN(name) out<<#name "="<<settings.name<<'\n';
  RUN_INTS(WRITE_RUN)
  RUN_DOUBLES(WRITE_RUN)
#undef WRITE_RUN
  out<<"channel_ids=";
  for(size_t i=0;i<settings.model.channel_ids.size();++i) out<<(i?",":"")<<settings.model.channel_ids[i];
  out<<"\ndevice="<<(settings.model.device.is_cuda()?"cuda":"cpu")<<'\n';
  return out.str();
}

Dataset synthetic_dataset(const Config &config,int64_t samples,int64_t seed) {
  validate_config(config); require(samples>0 && seed>=0,"samples must be positive and seed nonnegative");
  Dataset result;
  auto &input=result.input;
  input.data=torch::empty({samples,config.channel_count,config.history_length,config.input_width},torch::kFloat64);
  input.observed=torch::ones(input.data.sizes(),torch::kBool);
  input.channel_ids=torch::tensor(resolved_channel_ids(config),torch::kInt64);
  input.endpoints=(torch::arange(samples,torch::kFloat64)*config.history_length+config.history_length-1)*config.sampling_interval;
  input.sampling_interval=config.sampling_interval;
  auto data=input.data.accessor<double,4>(); auto observed=input.observed.accessor<bool,4>();
  constexpr double pi=3.14159265358979323846;
  for(int64_t b=0;b<samples;++b) for(int64_t c=0;c<config.channel_count;++c)
    for(int64_t h=0;h<config.history_length;++h) for(int64_t f=0;f<config.input_width;++f) {
      const double time=static_cast<double>(h)/std::max<int64_t>(1,config.history_length-1);
      const double phase=0.13*(b+c)+0.001*seed;
      const double frequency=2.0+(b%4)+0.5*f;
      data[b][c][h][f]=(1.0+0.1*c+0.03*f)*std::sin(2*pi*frequency*time+phase)+
          0.15*std::cos(2*pi*(1.0+f)*time-phase)+0.02*b+0.1*f*time;
      const auto cell=static_cast<uint64_t>(((b*config.channel_count+c)*config.history_length+h)*config.input_width+f);
      if(mixed(static_cast<uint64_t>(seed)+cell)%29==0) observed[b][c][h][f]=false;
    }
  result.feature_units=default_units(config.input_width);
  result.schema_id=schema_identity(input,result.feature_units);
  result.dataset_id=dataset_identity(result); return result;
}

void save_dataset(const std::string &path,const Dataset &dataset) {
  const auto &input=dataset.input;
  require(input.data.defined() && input.data.dim()==4,"raw archive requires BCHF data");
  Config config; config.channel_count=input.data.size(1);config.history_length=input.data.size(2);
  config.input_width=input.data.size(3); config.patch_length=1;
  auto ids=input.channel_ids.to(torch::kCPU).contiguous(); if(ids.dim()==2) ids=ids[0].contiguous();
  require(ids.scalar_type()==torch::kInt64 && ids.dim()==1,"invalid channel IDs");
  config.channel_ids=std::vector<int64_t>(ids.data_ptr<int64_t>(),ids.data_ptr<int64_t>()+ids.numel());
  config.sampling_interval=input.sampling_interval;
  validate_raw(input,config); validate_units(dataset.feature_units,config.input_width);
  torch::serialize::OutputArchive archive; envelope(archive,data_kind);
  archive.write("data",input.data.detach().to(torch::kCPU),true);
  archive.write("observed",input.observed.detach().to(torch::kCPU),true);
  archive.write("channel_ids",input.channel_ids.to(torch::kCPU),true);
  archive.write("endpoints",input.endpoints.to(torch::kCPU),true);
  archive.write("sampling_interval",torch::tensor(input.sampling_interval,torch::kFloat64),true);
  write_text(archive,"feature_units",dataset.feature_units);
  write_text(archive,"schema_id",schema_identity(input,dataset.feature_units));
  write_text(archive,"dataset_id",dataset_identity(dataset)); save_archive(path,archive);
}

Dataset load_dataset(const std::string &path,const Config &config) {
  torch::serialize::InputArchive archive; archive.load_from(path,torch::kCPU); check_envelope(archive,data_kind);
  Dataset result; auto &input=result.input;
  archive.read("data",input.data,true); archive.read("observed",input.observed,true);
  archive.read("channel_ids",input.channel_ids,true); archive.read("endpoints",input.endpoints,true);
  torch::Tensor interval; archive.read("sampling_interval",interval,true);
  require(interval.scalar_type()==torch::kFloat64 && interval.numel()==1,"invalid sampling_interval");
  input.sampling_interval=interval.item<double>(); result.feature_units=read_text(archive,"feature_units");
  validate_raw(input,config); validate_units(result.feature_units,config.input_width);
  result.schema_id=read_text(archive,"schema_id"); result.dataset_id=read_text(archive,"dataset_id");
  require(result.schema_id==schema_identity(input,result.feature_units),"raw archive schema identity mismatch");
  require(result.dataset_id==dataset_identity(result),"raw archive dataset identity mismatch"); return result;
}

void save_scaler(const std::string &path,const FrozenScaler &scaler,const Config &config,const std::string &schema_id,const std::string &fit_dataset_id) {
  scaler.validate(config); require(!schema_id.empty(),"scaler schema_id is empty");
  torch::serialize::OutputArchive archive,state; envelope(archive,scaler_kind);
  write_text(archive,"schema_id",schema_id); write_text(archive,"preprocessing_id",scaler.identity());
  write_text(archive,"fit_dataset_id",fit_dataset_id.empty()?"unrecorded":fit_dataset_id);
  scaler.save(state); archive.write("scaler",state); save_archive(path,archive);
}

FrozenScaler load_scaler(const std::string &path,const Config &config,const std::string &schema_id) {
  torch::serialize::InputArchive archive,state;archive.load_from(path,torch::kCPU);check_envelope(archive,scaler_kind);
  require(read_text(archive,"schema_id")==schema_id,"scaler input schema mismatch");
  archive.read("scaler",state); auto scaler=FrozenScaler::load(state);scaler.validate(config);
  require(scaler.identity()==read_text(archive,"preprocessing_id"),"scaler identity mismatch");return scaler;
}

void save_checkpoint(const std::string &path,const Checkpoint &checkpoint,torch::optim::AdamW &optimizer) {
  validate_settings(checkpoint.settings);checkpoint.scaler.validate(checkpoint.settings.model);
  require(checkpoint.model && checkpoint.attempted_steps>=checkpoint.completed_steps &&
      checkpoint.completed_steps>=0,"invalid checkpoint model/counters");
  require(!checkpoint.schema_id.empty() && !checkpoint.dataset_id.empty(),"missing checkpoint dataset provenance");
  require(checkpoint.training_policy_id.empty() || (checkpoint.training_policy_id.size()<=128 &&
      checkpoint.training_policy_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")==std::string::npos),
      "invalid checkpoint training policy identifier");
  torch::serialize::OutputArchive archive,weights,state,scaler;
  envelope(archive,"rpb_training_checkpoint_v1");
  write_text(archive,"output_semantics",output_semantics(checkpoint.settings.model));
  if(checkpoint.settings.model.temporal_difference_input>0) {
    archive.write("temporal_difference_input",torch::tensor(checkpoint.settings.model.temporal_difference_input,torch::kInt64),true);
    write_text(archive,"temporal_difference_input_semantics",visible_difference_input_semantics(checkpoint.settings.model));
  }
  if(checkpoint.settings.model.channel_mixer_placement>0) {
    archive.write("channel_mixer_placement",torch::tensor(checkpoint.settings.model.channel_mixer_placement),true);
    write_text(archive,"architecture_id",architecture_id(checkpoint.settings.model));
  }
  if(checkpoint.settings.model.global_pool_input_source>0) {
    archive.write("global_pool_input_source",torch::tensor(checkpoint.settings.model.global_pool_input_source,torch::kInt64),true);
    write_text(archive,"global_pool_input_semantics",global_pool_input_semantics(checkpoint.settings.model));
  }
  if(checkpoint.settings.model.global_bottleneck_mode>0) {
    archive.write("global_bottleneck_mode",torch::tensor(checkpoint.settings.model.global_bottleneck_mode),true);
    write_text(archive,"reconstruction_export_semantics",reconstruction_output_semantics(checkpoint.settings.model));
  }
  const auto text=settings_text(checkpoint.settings);write_text(archive,"settings",text);
  Fingerprint config_hash;config_hash.text(text);write_text(archive,"configuration_id",config_hash.id("rpb-config-fnv1a-v1"));
  write_text(archive,"schema_id",checkpoint.schema_id);write_text(archive,"dataset_id",checkpoint.dataset_id);
  write_text(archive,"scaler_fit_dataset_id",checkpoint.scaler_fit_dataset_id.empty()?"unrecorded":checkpoint.scaler_fit_dataset_id);
  write_text(archive,"preprocessing_id",checkpoint.scaler.identity());
  write_text(archive,"source_fingerprint_algorithm","sha256-source-manifest-v1");
  write_text(archive,"source_fingerprint",RPB_SOURCE_ID);write_text(archive,"git_head",RPB_GIT_HEAD);
  write_text(archive,"git_dirty",RPB_GIT_DIRTY);
  // Omit the optional field on ordinary checkpoints, preserving old schemas.
  if(!checkpoint.training_policy_id.empty())write_text(archive,"training_policy_id",checkpoint.training_policy_id);
  archive.write("attempted_steps",torch::tensor(checkpoint.attempted_steps),true);
  archive.write("completed_steps",torch::tensor(checkpoint.completed_steps),true);
  write_text(archive,"rng_policy","splitmix64-counter-rows-masks-torch-attempt-v1");
  checkpoint.model->save(weights); optimizer.save(state); checkpoint.scaler.save(scaler);
  archive.write("model",weights);archive.write("optimizer",state);archive.write("scaler",scaler);
  save_archive(path,archive);
}

Checkpoint load_checkpoint(const std::string &path,const torch::Device &device) {
  torch::serialize::InputArchive archive,weights,scaler;archive.load_from(path,device);
  check_envelope(archive,"rpb_training_checkpoint_v1");
  require(read_text(archive,"rng_policy")=="splitmix64-counter-rows-masks-torch-attempt-v1","checkpoint RNG policy mismatch");
  const auto text=read_text(archive,"settings");Fingerprint hash;hash.text(text);
  require(read_text(archive,"configuration_id")==hash.id("rpb-config-fnv1a-v1"),"checkpoint config identity mismatch");
  Checkpoint checkpoint;checkpoint.settings=parse_settings(text);checkpoint.settings.model.device=device;
  torch::Tensor difference_tag,difference_semantics_tag;
  const bool has_difference=archive.try_read("temporal_difference_input",difference_tag,true);
  const bool has_difference_semantics=archive.try_read("temporal_difference_input_semantics",difference_semantics_tag,true);
  require(checkpoint.settings.model.temporal_difference_input==0 ? (!has_difference && !has_difference_semantics) :
      (has_difference && difference_tag.scalar_type()==torch::kInt64 && difference_tag.dim()==0 &&
       difference_tag.item<int64_t>()==checkpoint.settings.model.temporal_difference_input &&
       has_difference_semantics && tensor_text(difference_semantics_tag)==visible_difference_input_semantics(checkpoint.settings.model)),
      "checkpoint temporal difference input/semantics metadata mismatch");
  torch::Tensor pool_source_tag,pool_semantics_tag;
  const bool has_pool_source=archive.try_read("global_pool_input_source",pool_source_tag,true);
  const bool has_pool_semantics=archive.try_read("global_pool_input_semantics",pool_semantics_tag,true);
  require(checkpoint.settings.model.global_pool_input_source==0 ? (!has_pool_source && !has_pool_semantics) :
      (has_pool_source && pool_source_tag.scalar_type()==torch::kInt64 && pool_source_tag.dim()==0 &&
       pool_source_tag.item<int64_t>()==checkpoint.settings.model.global_pool_input_source &&
       has_pool_semantics && tensor_text(pool_semantics_tag)==global_pool_input_semantics(checkpoint.settings.model)),
      "checkpoint global pool input source/semantics metadata mismatch");
  torch::Tensor placement_tag,architecture_tag;
  const bool has_placement=archive.try_read("channel_mixer_placement",placement_tag,true);
  const bool has_architecture=archive.try_read("architecture_id",architecture_tag,true);
  // Validate new architecture metadata before constructing or loading a model.
  require(checkpoint.settings.model.channel_mixer_placement==0 ? (!has_placement && !has_architecture) :
      (has_placement && placement_tag.scalar_type()==torch::kInt64 && placement_tag.dim()==0 &&
       placement_tag.item<int64_t>()==checkpoint.settings.model.channel_mixer_placement &&
       has_architecture && tensor_text(architecture_tag)==architecture_id(checkpoint.settings.model)),
      "checkpoint channel mixer placement/architecture metadata mismatch");
  require(read_text(archive,"output_semantics")==output_semantics(checkpoint.settings.model),
      "checkpoint output semantics mismatch");
  torch::Tensor global_mode_tag,reconstruction_tag;
  const bool has_global_mode=archive.try_read("global_bottleneck_mode",global_mode_tag,true);
  const bool has_reconstruction=archive.try_read("reconstruction_export_semantics",reconstruction_tag,true);
  require((!has_global_mode || (global_mode_tag.scalar_type()==torch::kInt64 &&
      global_mode_tag.numel()==1 && global_mode_tag.item<int64_t>()==checkpoint.settings.model.global_bottleneck_mode)) &&
      (!has_reconstruction || tensor_text(reconstruction_tag)==reconstruction_output_semantics(checkpoint.settings.model)) &&
      (checkpoint.settings.model.global_bottleneck_mode==0 || (has_global_mode && has_reconstruction)),
      "checkpoint global bottleneck metadata mismatch");
  checkpoint.schema_id=read_text(archive,"schema_id");checkpoint.dataset_id=read_text(archive,"dataset_id");
  checkpoint.scaler_fit_dataset_id=read_text(archive,"scaler_fit_dataset_id");
  checkpoint.source_fingerprint=read_text(archive,"source_fingerprint");
  checkpoint.git_head=read_text(archive,"git_head");checkpoint.git_dirty=read_text(archive,"git_dirty");
  torch::Tensor policy_tag;
  if(archive.try_read("training_policy_id",policy_tag,true)) {
    checkpoint.training_policy_id=tensor_text(policy_tag);
    require(!checkpoint.training_policy_id.empty() && checkpoint.training_policy_id.size()<=128 &&
        checkpoint.training_policy_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")==std::string::npos,
        "invalid checkpoint training policy identifier");
  }
  require(!checkpoint.schema_id.empty() && !checkpoint.dataset_id.empty(),"checkpoint lacks dataset provenance");
  checkpoint.attempted_steps=read_count(archive,"attempted_steps");checkpoint.completed_steps=read_count(archive,"completed_steps");
  require(checkpoint.completed_steps<=checkpoint.attempted_steps,"checkpoint counters are inconsistent");
  archive.read("scaler",scaler);checkpoint.scaler=FrozenScaler::load(scaler);checkpoint.scaler.validate(checkpoint.settings.model);
  require(checkpoint.scaler.identity()==read_text(archive,"preprocessing_id"),"checkpoint preprocessing identity mismatch");
  // Only after the encoder, format, config, schema and preprocessing checks.
  checkpoint.model=Model(checkpoint.settings.model);archive.read("model",weights);checkpoint.model->load(weights);
  return checkpoint;
}

void load_optimizer(const std::string &path,torch::optim::AdamW &optimizer,const torch::Device &device) {
  torch::serialize::InputArchive archive,state;archive.load_from(path,device);
  check_envelope(archive,"rpb_training_checkpoint_v1");archive.read("optimizer",state);optimizer.load(state);
}

int run_cli(int argc,char **argv) {
  if(argc<2 || std::string(argv[1])=="--help" || std::string(argv[1])=="help") {usage();return argc<2?1:0;}
  const std::string command(argv[1]);
  if(command=="evaluate")
    throw std::runtime_error("evaluation is provided by the separate embedding_evaluate executable; build the evaluation target and use embedding_evaluate --help");
  if(argc==3 && std::string(argv[2])=="--help") {usage();return 0;}
  if(command=="synthetic") {
    const auto args=arguments(argc,argv,{"--output","--config","--samples","--seed","--units"});
    const auto output=required(args,"--output");protect_inputs(args,output,{"--config"});
    auto settings=args.count("--config")?read_settings(args.at("--config")):default_settings();
    if(args.count("--seed")) settings.seed=integer(args.at("--seed"));
    validate_settings(settings);
    auto dataset=synthetic_dataset(settings.model,integer(optional(args,"--samples","32")),settings.seed);
    if(args.count("--units"))dataset.feature_units=args.at("--units");
    save_dataset(output,dataset);std::cout<<"saved precision-preserving raw dataset to "<<output<<'\n';return 0;
  }
  if(command=="prepare" || command=="scaler-fit") {
    const auto args=arguments(argc,argv,{"--input","--output","--config"});
    const auto output=required(args,"--output");protect_inputs(args,output,{"--input","--config"});
    auto settings=args.count("--config")?read_settings(args.at("--config")):default_settings();
    auto dataset=load_dataset(required(args,"--input"),settings.model);
    auto scaler=fit_scaler(dataset.input,settings.model);save_scaler(output,scaler,settings.model,dataset.schema_id,dataset.dataset_id);
    std::cout<<"saved frozen training scaler "<<scaler.identity()<<" to "<<output<<'\n';return 0;
  }
  if(command=="train") {
    const auto args=arguments(argc,argv,{"--checkpoint","--config","--input","--resume","--scaler","--steps","--attempt-limit","--batch-size","--device","--seed","--checkpoint-every"});
    const auto output=required(args,"--checkpoint"),input=optional(args,"--input"),resume=optional(args,"--resume");
    protect_inputs(args,output,{"--input","--config","--scaler"});
    require(resume.empty() || (!args.count("--config") && !args.count("--seed") && !args.count("--scaler")),
            "resume restores config, seed and scaler; omit --config/--seed/--scaler");
    Checkpoint checkpoint;
    if(!resume.empty()) {
      checkpoint=load_checkpoint(resume,device_from(optional(args,"--device","cpu")));
      require(checkpoint.settings.model.temporal_difference_input==0,
          "ordinary train/resume rejects visible differences; use its matching training adapter");
      require(checkpoint.settings.model.channel_mixer_placement==0,
          "ordinary train/resume rejects early mixer placement; use its matching training adapter");
      require(checkpoint.training_policy_id.empty(),
          "ordinary train cannot resume this saved training policy; use its matching training adapter: "+checkpoint.training_policy_id);
    }
    else checkpoint.settings=args.count("--config")?read_settings(args.at("--config")):default_settings();
    auto &settings=checkpoint.settings;
    require(settings.model.temporal_difference_input==0,
        "ordinary train/resume rejects visible differences; use its matching training adapter");
    require(settings.model.channel_mixer_placement==0,
        "ordinary train/resume rejects early mixer placement; use its matching training adapter");
    if(args.count("--device"))settings.model.device=device_from(args.at("--device"));
    if(args.count("--steps"))settings.steps=integer(args.at("--steps"));
    if(args.count("--attempt-limit"))settings.attempt_limit=integer(args.at("--attempt-limit"));
    if(args.count("--batch-size")) {
      const auto size=integer(args.at("--batch-size"));
      require(resume.empty() || size==settings.batch_size,"resume requires the saved training batch size");settings.batch_size=size;
    }
    if(args.count("--seed"))settings.seed=integer(args.at("--seed"));
    if(args.count("--checkpoint-every"))settings.checkpoint_every=integer(args.at("--checkpoint-every"));
    activate(settings);
    auto dataset=input.empty()?synthetic_dataset(settings.model,32,settings.seed):load_dataset(input,settings.model);
    if(!resume.empty()) require(checkpoint.dataset_id==dataset.dataset_id && checkpoint.schema_id==dataset.schema_id,
        "resume dataset/schema identity mismatch; use the original training archive");
    else {
      checkpoint.schema_id=dataset.schema_id;checkpoint.dataset_id=dataset.dataset_id;
      checkpoint.scaler=args.count("--scaler")?load_scaler(args.at("--scaler"),settings.model,dataset.schema_id):
          fit_scaler(dataset.input,settings.model);
      checkpoint.scaler_fit_dataset_id=dataset.dataset_id;
      if(args.count("--scaler")) {
        torch::serialize::InputArchive scaler_asset;scaler_asset.load_from(args.at("--scaler"),torch::kCPU);
        checkpoint.scaler_fit_dataset_id=read_text(scaler_asset,"fit_dataset_id");
      }
      checkpoint.model=Model(settings.model);
    }
    torch::optim::AdamW optimizer(checkpoint.model->parameters(),torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
    if(!resume.empty())load_optimizer(resume,optimizer,settings.model.device);
    checkpoint.model->train();
    require(checkpoint.completed_steps<=std::numeric_limits<int64_t>::max()-settings.steps &&
        checkpoint.attempted_steps<=std::numeric_limits<int64_t>::max()-settings.attempt_limit,"training counters would overflow");
    const auto end=checkpoint.completed_steps+settings.steps;
    const auto attempt_end=checkpoint.attempted_steps+settings.attempt_limit;
    while(checkpoint.completed_steps<end && checkpoint.attempted_steps<attempt_end) {
      const auto attempt=checkpoint.attempted_steps;
      const auto indices=sampled_indices(dataset.input.data.size(0),settings.batch_size,settings.seed,attempt);
      auto batch=checkpoint.scaler.transform(selected(dataset.input,indices),settings.model);
      const auto mask=make_training_mask(batch.observed,settings.model,counter_seed(settings.seed,attempt,0x6d61736bULL));
      torch::manual_seed(counter_seed(settings.seed,attempt,0x746f726368ULL));
      ++checkpoint.attempted_steps;
      const auto eligible=mask.eligible_channels.any().item<bool>();
      double loss=0,gradient_norm=0;int64_t target_cells=0;
      if(eligible) {
        optimizer.zero_grad();auto result=checkpoint.model->forward(batch,mask.hidden);
        require(result.eligible_example_count>0 && torch::isfinite(result.loss).all().item<bool>(),"eligible batch has invalid loss/support");
        result.loss.backward();
        gradient_norm=torch::nn::utils::clip_grad_norm_(checkpoint.model->parameters(),
            settings.gradient_clip_norm>0?settings.gradient_clip_norm:std::numeric_limits<double>::infinity(),2.0,true);
        optimizer.step();++checkpoint.completed_steps;loss=result.loss.item<double>();target_cells=result.target_cell_count;
      }
      if(checkpoint.attempted_steps%settings.log_every==0 || checkpoint.completed_steps==end)
        std::cout<<"attempt="<<checkpoint.attempted_steps<<" completed="<<checkpoint.completed_steps
                 <<" skipped="<<(!eligible)<<" loss="<<loss<<" target_cells="<<target_cells
                 <<" gradient_norm="<<gradient_norm<<'\n';
      if(settings.checkpoint_every>0 && checkpoint.attempted_steps%settings.checkpoint_every==0)
        save_checkpoint(output,checkpoint,optimizer);
    }
    save_checkpoint(output,checkpoint,optimizer);
    require(checkpoint.completed_steps==end,"attempt limit exhausted; recovery checkpoint saved with attempted/completed counters");
    std::cout<<"saved checkpoint attempts="<<checkpoint.attempted_steps<<" completed="<<checkpoint.completed_steps<<" to "<<output<<'\n';return 0;
  }
  if(command=="embed") {
    const auto args=arguments(argc,argv,{"--input","--output","--checkpoint","--device","--batch-size"});
    const auto output=required(args,"--output");protect_inputs(args,output,{"--input","--checkpoint"});
    auto checkpoint=load_checkpoint(required(args,"--checkpoint"),device_from(optional(args,"--device","cpu")));
    const auto size=integer(optional(args,"--batch-size",std::to_string(checkpoint.settings.batch_size)));
    require(size>0,"embedding batch size must be positive");activate(checkpoint.settings);
    const auto dataset=load_dataset(required(args,"--input"),checkpoint.settings.model);
    require(dataset.schema_id==checkpoint.schema_id,"embedding input schema differs from checkpoint");
    torch::NoGradGuard no_grad;checkpoint.model->eval();
    std::vector<torch::Tensor> local,global,contextual,contextual_global,channel_valid,sample_valid,observations,patches,ids;
    for(int64_t start=0;start<dataset.input.data.size(0);start+=size) {
      auto indices=torch::arange(start,std::min(start+size,dataset.input.data.size(0)),torch::kInt64);
      auto normalized=checkpoint.scaler.transform(selected(dataset.input,indices),checkpoint.settings.model);
      auto result=checkpoint.model->encode(normalized);
      require(torch::isfinite(result.z_local).all().item<bool>() && torch::isfinite(result.z_global).all().item<bool>(),"nonfinite exported representation");
      local.push_back(result.z_local.to(torch::kCPU));global.push_back(result.z_global.to(torch::kCPU));
      if(checkpoint.settings.model.channel_mixer_layers>0) {
        require(torch::isfinite(result.z_contextual).all().item<bool>() &&
            torch::isfinite(result.z_contextual_global).all().item<bool>(),"nonfinite contextual export");
        contextual.push_back(result.z_contextual.to(torch::kCPU));
        contextual_global.push_back(result.z_contextual_global.to(torch::kCPU));
      }
      channel_valid.push_back(result.channel_valid_mask.to(torch::kCPU));sample_valid.push_back(result.sample_valid_mask.to(torch::kCPU));
      observations.push_back(result.visible_observation_counts.to(torch::kCPU));patches.push_back(result.visible_patch_counts.to(torch::kCPU));
      ids.push_back(result.channel_ids.to(torch::kCPU));
    }
    torch::serialize::OutputArchive archive;envelope(archive,"rpb_embedding_export_v1");
    write_text(archive,"output_semantics",output_semantics(checkpoint.settings.model));write_text(archive,"schema_id",dataset.schema_id);
    if(checkpoint.settings.model.temporal_difference_input>0) {
      archive.write("temporal_difference_input",torch::tensor(checkpoint.settings.model.temporal_difference_input,torch::kInt64),true);
      write_text(archive,"temporal_difference_input_semantics",visible_difference_input_semantics(checkpoint.settings.model));
    }
    if(checkpoint.settings.model.channel_mixer_placement>0) {
      archive.write("channel_mixer_placement",torch::tensor(checkpoint.settings.model.channel_mixer_placement),true);
      write_text(archive,"architecture_id",architecture_id(checkpoint.settings.model));
    }
    if(checkpoint.settings.model.global_pool_input_source>0) {
      archive.write("global_pool_input_source",torch::tensor(checkpoint.settings.model.global_pool_input_source,torch::kInt64),true);
      write_text(archive,"global_pool_input_semantics",global_pool_input_semantics(checkpoint.settings.model));
    }
    if(checkpoint.settings.model.global_bottleneck_mode>0) {
      archive.write("global_bottleneck_mode",torch::tensor(checkpoint.settings.model.global_bottleneck_mode),true);
      write_text(archive,"reconstruction_export_semantics",reconstruction_output_semantics(checkpoint.settings.model));
    }
    write_text(archive,"dataset_id",dataset.dataset_id);write_text(archive,"preprocessing_id",checkpoint.scaler.identity());
    write_text(archive,"scaler_fit_dataset_id",checkpoint.scaler_fit_dataset_id);
    write_text(archive,"checkpoint_source_fingerprint",checkpoint.source_fingerprint);
    write_text(archive,"inference_source_fingerprint",RPB_SOURCE_ID);
    if(!checkpoint.training_policy_id.empty())write_text(archive,"training_policy_id",checkpoint.training_policy_id);
    archive.write("z_local",torch::cat(local),true);archive.write("z_global",torch::cat(global),true);
    if(!contextual.empty()) {
      archive.write("z_contextual",torch::cat(contextual),true);
      archive.write("z_contextual_global",torch::cat(contextual_global),true);
    }
    archive.write("channel_valid_mask",torch::cat(channel_valid),true);archive.write("sample_valid_mask",torch::cat(sample_valid),true);
    archive.write("visible_observation_counts",torch::cat(observations),true);archive.write("visible_patch_counts",torch::cat(patches),true);
    archive.write("channel_ids",torch::cat(ids),true);archive.write("endpoints",dataset.input.endpoints,true);
    archive.write("sampling_interval",torch::tensor(dataset.input.sampling_interval,torch::kFloat64),true);
    save_archive(output,archive);std::cout<<"saved local/global embeddings to "<<output<<'\n';return 0;
  }
  throw std::runtime_error("unknown command: "+command+"; use --help");
}

#undef MODEL_INTS
#undef MODEL_DOUBLES
#undef RUN_INTS
#undef RUN_DOUBLES
} // namespace embedding::encoders::raw_patch_bottleneck_mae
