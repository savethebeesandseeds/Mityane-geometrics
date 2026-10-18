// SPDX-License-Identifier: MIT
#include "embedding/shared/archive_readout.h"
#include <ATen/Context.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[archive readout] " + message);
}
bool safe_name(const std::string &value) {
  return !value.empty() && value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) {
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
struct Outputs {
  fs::path root;
  std::vector<fs::path> files;
  void text(const fs::path &relative,const std::string &value) {
    const auto path=root/relative;require(!fs::exists(path), "refusing existing output: "+path.string());
    std::ofstream out(path,std::ios::binary);require(bool(out),"cannot create output: "+path.string());
    out << value;out.close();require(bool(out),"cannot save output: "+path.string());files.push_back(relative);
  }
  void archive(const fs::path &relative,torch::serialize::OutputArchive &value) {
    const auto path=root/relative;require(!fs::exists(path), "refusing existing output: "+path.string());
    embedding::archive::save_archive(path.string(),value);files.push_back(relative);
  }
  std::string manifest() const {
    std::ostringstream out;out << "{\"checksum_algorithm\":\"sha256-file-bytes\",\"files\":[";
    for (size_t i=0;i<files.size();++i) {
      if (i) out << ',';
      const auto value=bytes(root/files[i]);
      out << "{\"path\":" << quote(files[i].generic_string()) << ",\"bytes\":" << value.size() << ",\"sha256\":" << quote(sha256(value)) << '}';
    }
    return out.str()+"]}";
  }
};
struct RuntimeIsolation {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  int old_threads{at::get_num_threads()};
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i=0;i<at::getNumGPUs();++i) generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator:generators) states.push_back(generator.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try { for (size_t i=0;i<generators.size();++i) generators[i].set_state(states[i]);at::set_num_threads(old_threads); }
    catch (...) { std::terminate(); }
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

struct Split {
  Batch observed;
  FeatureSurface native;
  torch::Tensor labels;
  std::vector<std::string> source_ids;
};
Split load_split(const ArchiveReadoutInput &input,const ArchiveReadoutRun &run,bool training,
                 const ArchiveReadoutValidationView *view=nullptr) {
  torch::serialize::InputArchive observations,features;
  observations.load_from(view?view->observations:(training?input.training_observations:input.validation_observations),torch::kCPU);
  features.load_from(view?view->features:(training?input.training_features:input.validation_features),torch::kCPU);
  Split out;torch::Tensor sources,provenance;
  observations.read(input.observations_key,out.observed.data,true);
  observations.read(input.observation_mask_key,out.observed.feature_mask,true);
  observations.read(input.labels_key,out.labels,true);observations.read(input.source_ids_key,sources,true);
  features.read(input.feature_values_key,out.native.values,true);features.read(input.feature_valid_key,out.native.valid,true);
  features.read(input.feature_provenance_key,provenance,true);
  out.native.provenance=archive::tensor_text(provenance);
  out.source_ids=SourceParser(archive::tensor_text(sources)).parse();
  const auto &data=out.observed.data;const auto &mask=out.observed.feature_mask;
  require(data.defined() && data.device().is_cpu() && data.scalar_type()==torch::kFloat64 && data.dim()==4 && data.size(0)>0 &&
      data.size(1)==run.shape.channel_count && data.size(2)==run.shape.history_length && data.size(3)==run.shape.input_width,
      "observations require declared float64 CPU [B,C,H,F] geometry");
  require(mask.defined() && mask.device().is_cpu() && mask.scalar_type()==torch::kBool && mask.sizes()==data.sizes() &&
      torch::isfinite(data.masked_select(mask)).all().item<bool>(), "mask shape/type or finite observed values invalid");
  require(out.labels.defined() && out.labels.device().is_cpu() && out.labels.scalar_type()==torch::kInt64 &&
      out.labels.sizes()==torch::IntArrayRef({data.size(0)}) && out.labels.ge(0).logical_and(out.labels.le(1)).all().item<bool>() &&
      out.source_ids.size()==size_t(data.size(0)), "binary label/source rows do not match observations");
  validate_features(out.native);
  require((out.native.values.scalar_type()==torch::kFloat32 || out.native.values.scalar_type()==torch::kFloat64) &&
      out.native.values.sizes()==torch::IntArrayRef({data.size(0),run.compact_width}), "native features must retain declared rows and compact width without projection");
  const auto observed_valid=mask.flatten(1).any(1);
  require(out.native.valid.logical_and(observed_valid.logical_not()).any().item<bool>()==false,
      "native support includes an all-missing observation row");
  const auto &expected_provenance=view?view->expected_feature_provenance:input.expected_feature_provenance;
  require(!out.native.provenance.empty() && (expected_provenance.empty() || expected_provenance==out.native.provenance),
      "native feature provenance empty or differs from pinned provenance");
  std::map<std::string,std::vector<int64_t>> groups;
  for (int64_t row=0;row<data.size(0);++row) groups[out.source_ids.at(row)].push_back(row);
  for (const auto &[id,rows]:groups) require(!id.empty() && rows.size()==2 &&
      out.labels[rows[0]].item<int64_t>()!=out.labels[rows[1]].item<int64_t>() && torch::equal(mask[rows[0]],mask[rows[1]]),
      "controlled archives require complete opposite-label source pairs with identical masks");
  out.observed={data.detach().clone(),mask.clone()};
  out.native.values=out.native.values.detach().clone();out.native.valid=out.native.valid.clone();out.labels=out.labels.clone();
  return out;
}
void validate_view(const Split &view,const Split &ordinary) {
  require(view.source_ids==ordinary.source_ids && torch::equal(view.labels,ordinary.labels),
      "validation view labels/source IDs/order differ from ordinary VALIDATION");
  require(view.observed.data.sizes()==ordinary.observed.data.sizes() &&
      view.native.values.scalar_type()==ordinary.native.values.scalar_type(),
      "validation view geometry/native dtype differs from ordinary VALIDATION");
  const auto &mask=view.observed.feature_mask;
  require(!mask.logical_and(ordinary.observed.feature_mask.logical_not()).any().item<bool>(),
      "validation view introduces observation support");
  require(torch::equal(view.observed.data.masked_select(mask),ordinary.observed.data.masked_select(mask)),
      "validation view changes retained observed values");
  require(view.observed.data.masked_select(mask.logical_not()).eq(0).all().item<bool>(),
      "validation view hidden observation storage must be zero");
}
std::string population(const Split &split,const torch::Tensor &valid) {
  std::map<std::string,std::pair<int64_t,int64_t>> groups;
  int64_t classes[2]{0,0},complete=0,selected_groups=0;
  for (int64_t row=0;row<valid.size(0);++row) {
    auto &group=groups[split.source_ids.at(row)];++group.first;
    if (valid[row].item<bool>()) {++group.second;++classes[split.labels[row].item<int64_t>()];}
  }
  for (const auto &[id,count]:groups) { (void)id;if(count.second)++selected_groups;if(count.second==count.first)++complete; }
  std::ostringstream out;out << std::setprecision(17) << "{\"total_rows\":" << valid.size(0) << ",\"valid_rows\":" << classes[0]+classes[1]
      << ",\"class_valid_rows\":[" << classes[0] << ',' << classes[1] << "],\"total_source_groups\":" << groups.size()
      << ",\"valid_source_groups\":" << selected_groups << ",\"complete_source_pairs\":" << complete
      << ",\"coverage\":" << double(classes[0]+classes[1])/valid.size(0) << '}';
  return out.str();
}
std::string score_json(const Score &value) {
  std::ostringstream out;out << std::setprecision(17) << "{\"total\":" << value.total << ",\"valid\":" << value.valid
      << ",\"correct\":" << value.correct << ",\"abstained\":" << value.total-value.valid
      << ",\"full_population_correctness\":" << double(value.correct)/value.total << ",\"coverage\":" << value.coverage << ",\"accuracy\":";
  if (value.supported) out << value.accuracy;else out << "null";
  return out.str()+'}';
}
std::string interval_json(const GroupedInterval &value,int64_t replicates) {
  std::ostringstream out;out << std::setprecision(17) << "{\"source_groups\":" << value.source_groups << ",\"replicates\":" << replicates
      << ",\"confidence\":0.95,\"estimate\":";
  if(value.source_groups)out << value.estimate;else out << "null";
  out << ",\"lower\":";if(value.supported)out << value.lower;else out << "null";
  out << ",\"upper\":";if(value.supported)out << value.upper;else out << "null";
  return out.str()+'}';
}
std::string view_json(const ArchiveReadoutValidationView &view) {
  return "{\"id\":"+quote(view.id)+",\"observations\":"+quote(view.observations)+",\"features\":"+quote(view.features)+
      ",\"observations_sha256\":"+quote(view.observations_sha256)+",\"features_sha256\":"+quote(view.features_sha256)+
      ",\"expected_feature_provenance\":"+quote(view.expected_feature_provenance)+",\"corruption_provenance\":"+quote(view.corruption_provenance)+'}';
}
std::string input_json(const ArchiveReadoutInput &input) {
  auto value="{\"id\":"+quote(input.id)+",\"tag\":"+quote(input.tag)+",\"task\":"+quote(input.task)+
      ",\"master_seed\":"+quote(std::to_string(input.master_seed))+",\"checkpoint_steps\":"+std::to_string(input.checkpoint_steps)+
      ",\"producer_source_fingerprint\":"+quote(input.producer_source_fingerprint)+",\"cohort_provenance\":"+quote(input.cohort_provenance)+
      ",\"training_observations\":"+quote(input.training_observations)+",\"validation_observations\":"+quote(input.validation_observations)+
      ",\"training_features\":"+quote(input.training_features)+",\"validation_features\":"+quote(input.validation_features)+
      ",\"observations_key\":"+quote(input.observations_key)+",\"observation_mask_key\":"+quote(input.observation_mask_key)+
      ",\"labels_key\":"+quote(input.labels_key)+",\"source_ids_key\":"+quote(input.source_ids_key)+
      ",\"feature_values_key\":"+quote(input.feature_values_key)+",\"feature_valid_key\":"+quote(input.feature_valid_key)+
      ",\"feature_provenance_key\":"+quote(input.feature_provenance_key)+
      ",\"training_observations_sha256\":"+quote(input.training_observations_sha256)+",\"validation_observations_sha256\":"+quote(input.validation_observations_sha256)+
      ",\"training_features_sha256\":"+quote(input.training_features_sha256)+",\"validation_features_sha256\":"+quote(input.validation_features_sha256)+
      ",\"expected_feature_provenance\":"+quote(input.expected_feature_provenance);
  if(!input.validation_views.empty()) {
    value+=",\"validation_views\":[";
    for(size_t i=0;i<input.validation_views.size();++i) {if(i)value+=',';value+=view_json(input.validation_views[i]);}
    value+=']';
  }
  return value+'}';
}
std::string card_json(const ArchiveReadoutRun &run) {
  std::ostringstream out;out << std::setprecision(17) << "{\"version\":1,\"protocol\":\"archive-readout-v1\",\"stage\":\"development\",\"policy_version\":\"1.2\","
      << "\"validation_only\":true,\"test_access\":false,\"encoder_training\":false,\"post_encoder_pca\":false,"
      << "\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"shape\":[" << run.shape.channel_count << ',' << run.shape.history_length << ',' << run.shape.input_width << ']'
      << ",\"compact_width\":" << run.compact_width << ",\"threads\":" << run.threads << ",\"repetitions\":[";
  for(size_t i=0;i<run.repetitions.size();++i) {if(i)out << ',';out << "{\"id\":" << quote(run.repetitions[i].id)
      << ",\"probe_seed\":" << quote(std::to_string(run.repetitions[i].probe_seed)) << '}';}
  out << "],\"inputs\":[";for(size_t i=0;i<run.inputs.size();++i){if(i)out << ',';out << input_json(run.inputs[i]);}
  if(std::any_of(run.inputs.begin(),run.inputs.end(),[](const auto &input){return !input.validation_views.empty();}))
    out << "],\"validation_view_policy\":{\"roles\":\"declared VALIDATION only; exact ordinary row/source/label order\","
        "\"corruption\":\"support deletion only; retained raw values exact; hidden storage zero\","
        "\"fit_policy\":\"ordinary TRAIN maps and each head fitted once; all validation views reuse those same tensors; no view fit\","
        "\"effects\":\"view minus ordinary on common valid rows; full-population correctness counts abstentions as failure\","
        "\"bootstrap_seed\":\"stream_seed(rep.probe_seed,FNV1a64(archive-readout-validation-view-v1/view_id)); named method/probe streams\"},\"recipe\":{";
  else out << "],\"recipe\":{";
  out << "\"observation_scaler\":\"existing ObservationScaler; float64 observed TRAIN per-channel/feature population mean/std across rows/history; scale floor1e-8\","
      << "\"raw\":\"masked standardized observations flattened in declared C/H/F order, followed by original visibility flags; no clean values\","
      << "\"pca_only\":\"raw TRAIN outer FeatureNormalizer -> centered train-fit PCA to compact_width; no encoder\","
      << "\"native\":\"exact archived compact_width export -> TRAIN outer FeatureNormalizer; no projection\","
      << "\"probe_normalization\":\"each probe own valid-TRAIN FeatureNormalizer; all fitted assets retained\","
      << "\"support\":\"raw/PCA: at least one observed coordinate; native: archived valid subset of observed rows; each method fits its valid TRAIN rows; paired validation uses intersections; all-missing abstains\","
      << "\"row_association\":\"inherited original producer row order pinned by declared archive hashes and caller-audited cohort lineage; feature archive has no embedded row IDs\","
      << "\"actual_probe_seed\":\"stream_seed(repetition.probe_seed, input_width), shared across equal-width methods and cohorts; no best-seed selection\","
      << "\"ridge\":{\"penalty\":" << run.ridge_penalty << ",\"classes\":2,\"parameters\":\"2*input_width+2\"},"
      << "\"tiny_secondary\":{\"hidden\":" << run.tiny_hidden << ",\"activation\":\"tanh\",\"updates\":" << run.tiny_steps
      << ",\"optimizer\":\"Adam\",\"learning_rate\":" << run.tiny_learning_rate << ",\"parameters\":\"hidden*(input_width+3)+2\"},"
      << "\"comparisons\":[\"native minus pca_only\",\"native minus raw\",\"pca_only minus raw\"],"
      << "\"uncertainty\":\"paired validation source-group percentile bootstrap on common valid rows; conditional on archived encoder and fitted readouts; no across-retraining interval\","
      << "\"confidence\":0.95,\"bootstrap_replicates\":" << run.bootstrap_replicates << ",\"acceptance\":\"development diagnosis only; none\"}}";
  return out.str();
}
void save_features(Outputs &out,const fs::path &path,const FeatureSurface &surface,const Split &split) {
  torch::serialize::OutputArchive value;
  value.write("features",surface.values,true);value.write("valid",surface.valid,true);
  value.write("provenance",archive::text_tensor(surface.provenance),true);
  value.write("labels_scoring_only",split.labels,true);value.write("source_ids_json",archive::text_tensor(strings(split.source_ids)),true);
  out.archive(path,value);
}
void save_normalizer(Outputs &out,const fs::path &path,const FeatureNormalizer &normalizer) {
  torch::serialize::OutputArchive value;value.write("feature_mean",normalizer.mean,true);value.write("feature_scale",normalizer.scale,true);
  value.write("fitted_rows",torch::tensor(normalizer.fitted_rows),true);out.archive(path,value);
}
struct Method {
  std::string id,label,reason;
  int64_t width{0};
  FeatureSurface training,validation;
  std::unique_ptr<FeatureNormalizer> outer;
  torch::Tensor training_support,validation_support;
  bool supported() const {return reason.empty();}
};
struct Readout {
  RidgeProbe ridge;
  TinyProbe tiny;
  torch::Tensor training_ridge,training_tiny,validation_ridge,validation_tiny;
  uint64_t seed;
  static torch::Tensor ridge_predictions(const RidgeProbe &probe,const FeatureSurface &surface) {
    torch::NoGradGuard guard;
    const auto logits=probe.normalizer.transform(surface).values.matmul(probe.weights)+probe.intercept;
    require(torch::isfinite(logits).all().item<bool>(),"ridge prediction arithmetic overflow");
    return logits.argmax(1);
  }
  static torch::Tensor tiny_predictions(const TinyProbe &probe,const FeatureSurface &surface) {
    torch::NoGradGuard guard;
    const auto hidden=probe.normalizer.transform(surface).values.matmul(probe.w1)+probe.b1;
    require(torch::isfinite(hidden).all().item<bool>(),"neural prediction arithmetic overflow");
    const auto logits=torch::tanh(hidden).matmul(probe.w2)+probe.b2;
    require(torch::isfinite(logits).all().item<bool>(),"neural prediction logits overflow");
    return logits.argmax(1);
  }
  Readout(const Method &method,const Split &training,const ArchiveReadoutRun &run,uint64_t actual_seed)
      :ridge(method.training,training.labels,run.ridge_penalty),tiny(method.training,training.labels,actual_seed,run.tiny_steps,run.tiny_hidden,run.tiny_learning_rate),seed(actual_seed) {
    training_ridge=ridge_predictions(ridge,method.training);training_tiny=tiny_predictions(tiny,method.training);
    validation_ridge=ridge_predictions(ridge,method.validation);validation_tiny=tiny_predictions(tiny,method.validation);
    for(const auto &tensor:{ridge.weights,ridge.intercept,tiny.w1,tiny.b1,tiny.w2,tiny.b2})
      require(torch::isfinite(tensor).all().item<bool>(),"fitted head contains nonfinite tensors");
  }
};
void save_readout(Outputs &out,const fs::path &path,const Readout &readout,const Method &method) {
  torch::serialize::OutputArchive value;
  value.write("feature_mean",method.outer->mean,true);value.write("feature_scale",method.outer->scale,true);
  value.write("fitted_rows",torch::tensor(method.outer->fitted_rows),true);
  value.write("ridge_mean",readout.ridge.normalizer.mean,true);value.write("ridge_scale",readout.ridge.normalizer.scale,true);
  value.write("ridge_weights",readout.ridge.weights,true);value.write("ridge_intercept",readout.ridge.intercept,true);
  value.write("tiny_mean",readout.tiny.normalizer.mean,true);value.write("tiny_scale",readout.tiny.normalizer.scale,true);
  value.write("tiny_w1",readout.tiny.w1,true);value.write("tiny_b1",readout.tiny.b1,true);
  value.write("tiny_w2",readout.tiny.w2,true);value.write("tiny_b2",readout.tiny.b2,true);
  value.write("actual_probe_seed_decimal",archive::text_tensor(std::to_string(readout.seed)),true);out.archive(path,value);
}
void save_predictions(Outputs &out,const fs::path &path,const torch::Tensor &ridge,const torch::Tensor &tiny,const FeatureSurface &features,const Split &split) {
  torch::serialize::OutputArchive value;value.write("ridge",ridge,true);value.write("tiny_secondary",tiny,true);
  value.write("valid",features.valid,true);value.write("probe_features",features.values,true);
  value.write("labels_scoring_only",split.labels,true);value.write("source_ids_json",archive::text_tensor(strings(split.source_ids)),true);out.archive(path,value);
}
uint64_t validation_view_seed(uint64_t seed,const std::string &id) {
  uint64_t hash=14695981039346656037ULL;
  for(const unsigned char c:"archive-readout-validation-view-v1/"+id) {hash^=c;hash*=1099511628211ULL;}
  return stream_seed(seed,hash);
}
std::vector<torch::Tensor> readout_tensors(const Readout &readout,const Method &method) {
  return {method.outer->mean,method.outer->scale,readout.ridge.normalizer.mean,readout.ridge.normalizer.scale,
      readout.ridge.weights,readout.ridge.intercept,readout.tiny.normalizer.mean,readout.tiny.normalizer.scale,
      readout.tiny.w1,readout.tiny.b1,readout.tiny.w2,readout.tiny.b2};
}
void write_validation_views(Outputs &outputs,std::ostringstream &report,const ArchiveReadoutInput &input,
    const ArchiveReadoutRun &run,const ArchiveReadoutRepetition &rep,const fs::path &rep_dir,
    const std::vector<Split> &views,const std::vector<Method> &methods,
    const std::array<std::vector<FeatureSurface>,3> &features,
    const std::vector<std::unique_ptr<Readout>> &readouts) {
  if(views.empty())return;
  fs::create_directory(outputs.root/rep_dir/"validation-views");
  report << ",\"validation_views\":[";
  for(size_t view_index=0;view_index<views.size();++view_index) {
    if(view_index)report << ',';
    const auto &view=views[view_index];const auto &declared=input.validation_views[view_index];
    const auto directory=rep_dir/"validation-views"/declared.id;fs::create_directory(outputs.root/directory);
    const auto view_seed=validation_view_seed(rep.probe_seed,declared.id);
    report << "{\"view\":" << view_json(declared) << ",\"bootstrap_seed_decimal\":" << quote(std::to_string(view_seed))
        << ",\"fit_policy\":\"same ordinary TRAIN fit tensors; no validation-view fitting\",\"methods\":[";
    std::array<torch::Tensor,3> valid,ridge,tiny;
    for(size_t m=0;m<methods.size();++m) {
      if(m)report << ',';
      const auto &method=methods[m];
      valid[m]=(m==2?view.native.valid:view.observed.feature_mask.flatten(1).any(1)).clone();
      const auto common=valid[m].logical_and(method.validation_support);
      report << "{\"method\":" << quote(method.id) << ",\"label\":" << quote(method.label) << ",\"size\":" << method.width
          << ",\"validation_population\":" << population(view,valid[m]) << ",\"ordinary_common_population\":" << population(view,common);
      if(!readouts[m]) {
        report << ",\"status\":\"unsupported\",\"reason\":" << quote(method.reason) << ",\"ridge\":null,\"tiny_secondary\":null}";
        continue;
      }
      const auto &readout=*readouts[m];const auto &surface=features[m].at(view_index);
      require(torch::equal(surface.valid,valid[m]),"validation view transform changed support");
      std::vector<torch::Tensor> frozen;
      for(const auto &value:readout_tensors(readout,method))frozen.push_back(value.clone());
      ridge[m]=Readout::ridge_predictions(readout.ridge,surface);tiny[m]=Readout::tiny_predictions(readout.tiny,surface);
      const auto after=readout_tensors(readout,method);
      for(size_t i=0;i<after.size();++i)require(torch::equal(after[i],frozen[i]),"validation view mutated ordinary TRAIN fit tensors");
      const auto predictions=directory/(method.id+"-predictions.pt");
      save_predictions(outputs,predictions,ridge[m],tiny[m],surface,view);
      const auto method_seed=stream_seed(view_seed,m+1);
      auto probe_json=[&](const torch::Tensor &predicted,const torch::Tensor &base,uint64_t stream) {
        const auto seed=stream_seed(method_seed,stream);
        const auto all=torch::ones_like(valid[m]);const auto truth=torch::ones_like(view.labels);
        const auto correct=predicted.eq(view.labels).logical_and(valid[m]).to(torch::kInt64);
        const auto base_correct=base.eq(view.labels).logical_and(method.validation_support).to(torch::kInt64);
        return "{\"validation\":"+score_json(score(predicted,view.labels,valid[m]))+
            ",\"validation_interval\":"+interval_json(grouped_accuracy_interval(predicted,view.labels,valid[m],view.source_ids,seed,run.bootstrap_replicates),run.bootstrap_replicates)+
            ",\"ordinary_validation_on_common\":"+score_json(score(base,view.labels,common))+
            ",\"view_validation_on_common\":"+score_json(score(predicted,view.labels,common))+
            ",\"view_minus_ordinary_common_interval\":"+interval_json(grouped_accuracy_interval(predicted,view.labels,common,view.source_ids,stream_seed(seed,1),run.bootstrap_replicates,base),run.bootstrap_replicates)+
            ",\"view_minus_ordinary_full_population_interval\":"+interval_json(grouped_accuracy_interval(correct,truth,all,view.source_ids,stream_seed(seed,2),run.bootstrap_replicates,base_correct),run.bootstrap_replicates)+'}';
      };
      report << ",\"status\":\"measured\",\"fit_file\":" << quote((rep_dir/(method.id+"-fit.pt")).generic_string())
          << ",\"validation_predictions_file\":" << quote(predictions.generic_string())
          << ",\"ridge\":" << probe_json(ridge[m],readout.validation_ridge,0x7269646765ULL)
          << ",\"tiny_secondary\":" << probe_json(tiny[m],readout.validation_tiny,0x74696e79ULL) << '}';
    }
    report << "],\"paired_comparisons\":[";
    const std::array<std::pair<size_t,size_t>,3> pairs{{{2,1},{2,0},{1,0}}};
    for(size_t p=0;p<pairs.size();++p) {
      if(p)report << ',';
      const auto [left,right]=pairs[p];const auto common=valid[left].logical_and(valid[right]);
      report << "{\"id\":" << quote(methods[left].id+"_minus_"+methods[right].id) << ",\"candidate\":" << quote(methods[left].id)
          << ",\"comparator\":" << quote(methods[right].id) << ",\"common_population\":" << population(view,common);
      if(!readouts[left] || !readouts[right]) {
        report << ",\"status\":\"unsupported\",\"reason\":\"at least one ordinary TRAIN fit unsupported\",\"ridge\":null,\"tiny_secondary\":null}";
        continue;
      }
      const auto pair_seed=stream_seed(view_seed,0x706169720000ULL+p);
      auto paired=[&](const torch::Tensor &candidate,const torch::Tensor &comparator,uint64_t stream) {
        return "{\"candidate\":"+score_json(score(candidate,view.labels,common))+",\"comparator\":"+score_json(score(comparator,view.labels,common))+
            ",\"candidate_minus_comparator_interval\":"+interval_json(grouped_accuracy_interval(candidate,view.labels,common,view.source_ids,stream_seed(pair_seed,stream),run.bootstrap_replicates,comparator),run.bootstrap_replicates)+'}';
      };
      report << ",\"status\":\"measured\",\"ridge\":" << paired(ridge[left],ridge[right],0x7269646765ULL)
          << ",\"tiny_secondary\":" << paired(tiny[left],tiny[right],0x74696e79ULL) << '}';
    }
    report << "]}";
  }
  report << ']';
}
bool both_classes(const torch::Tensor &labels,const torch::Tensor &valid) {
  const auto selected=labels.masked_select(valid);
  return selected.eq(0).any().item<bool>() && selected.eq(1).any().item<bool>();
}
void validate_run(const ArchiveReadoutRun &run) {
  require(!run.output_directory.empty() && !fs::exists(run.output_directory),"new nonempty output directory required");
  require(run.shape.channel_count>0 && run.shape.history_length>0 && run.shape.input_width>0 && run.shape.dtype==torch::kFloat64 && run.shape.device.is_cpu(),"declared geometry must be positive CPU float64");
  const auto maximum=std::numeric_limits<int64_t>::max();
  require(run.shape.channel_count<=maximum/run.shape.history_length && run.shape.channel_count*run.shape.history_length<=maximum/run.shape.input_width/2,
      "raw representation geometry overflows");
  require(!run.inputs.empty() && !run.repetitions.empty() && run.compact_width>0 && run.threads>0 && run.bootstrap_replicates>=100 &&
      std::isfinite(run.ridge_penalty) && run.ridge_penalty>0 && run.tiny_hidden>0 && run.tiny_steps>0 &&
      std::isfinite(run.tiny_learning_rate) && run.tiny_learning_rate>0,"empty/invalid readout recipe");
  const auto raw_width=2*run.shape.channel_count*run.shape.history_length*run.shape.input_width;
  require(raw_width<=(maximum-2)/std::max<int64_t>(2,run.tiny_hidden)-3 && run.compact_width<=(maximum-2)/std::max<int64_t>(2,run.tiny_hidden)-3,
      "head parameter count overflows");
  std::set<std::string> ids;
  for(const auto &input:run.inputs) {
    require(safe_name(input.id) && ids.insert(input.id).second && !input.tag.empty() && safe_name(input.task) && input.checkpoint_steps>=0 &&
        !input.producer_source_fingerprint.empty() && !input.cohort_provenance.empty(),"input requires unique safe ID, tag/task and frozen producer/cohort lineage");
    for(const auto &key:{input.observations_key,input.observation_mask_key,input.labels_key,input.source_ids_key,input.feature_values_key,input.feature_valid_key,input.feature_provenance_key})
      require(!key.empty(),"declared archive key is empty");
    for(const auto &hash:{input.training_observations_sha256,input.validation_observations_sha256,input.training_features_sha256,input.validation_features_sha256})
      require(hash.empty() || (hash.size()==64 && hash.find_first_not_of("0123456789abcdef")==std::string::npos),"expected SHA-256 must be empty or64 lowercase hex digits");
    std::set<std::string> views;
    for(const auto &view:input.validation_views) {
      require(safe_name(view.id) && views.insert(view.id).second && !view.corruption_provenance.empty(),
          "validation view requires a unique safe ID and frozen corruption provenance");
      for(const auto &hash:{view.observations_sha256,view.features_sha256})
        require(hash.empty() || (hash.size()==64 && hash.find_first_not_of("0123456789abcdef")==std::string::npos),
            "validation view expected SHA-256 must be empty or64 lowercase hex digits");
    }
  }
  ids.clear();for(const auto &rep:run.repetitions)require(safe_name(rep.id) && ids.insert(rep.id).second,"repetition requires unique safe ID");
}
} // namespace

void run_archive_readout(const ArchiveReadoutRun &run) {
  validate_run(run);InputGuard guard;
  std::set<std::string> training_paths,validation_paths;
  for(const auto &input:run.inputs) {
    guard.add(input.training_observations,input.training_observations_sha256);guard.add(input.validation_observations,input.validation_observations_sha256);
    guard.add(input.training_features,input.training_features_sha256);guard.add(input.validation_features,input.validation_features_sha256);
    for(const auto &path:{input.training_observations,input.training_features})training_paths.insert(fs::canonical(path).string());
    for(const auto &path:{input.validation_observations,input.validation_features})validation_paths.insert(fs::canonical(path).string());
    for(const auto &view:input.validation_views) {
      guard.add(view.observations,view.observations_sha256);guard.add(view.features,view.features_sha256);
      for(const auto &path:{view.observations,view.features})validation_paths.insert(fs::canonical(path).string());
    }
  }
  for(const auto &path:training_paths)require(!validation_paths.count(path),"archive appears in both TRAIN and VALIDATION roles");
  Outputs outputs{fs::absolute(run.output_directory),{}};
  require(fs::create_directories(outputs.root),"output directory was claimed by another run");
  outputs.text("archive-readout-card.json",card_json(run));outputs.text("input-manifest.json",guard.manifest());
  RuntimeIsolation runtime;at::set_num_threads(run.threads);
  try {
    std::set<std::string> train_sources,validation_sources;
    std::vector<std::pair<Split,Split>> splits;
    std::vector<std::vector<Split>> validation_views;
    // Validate all roles across the complete declared matrix before any fitting.
    for(const auto &input:run.inputs) {
      auto training=load_split(input,run,true),validation=load_split(input,run,false);
      require(training.native.values.scalar_type()==validation.native.values.scalar_type() && training.native.provenance==validation.native.provenance,
          "native dtype/provenance differs between TRAIN and VALIDATION");
      train_sources.insert(training.source_ids.begin(),training.source_ids.end());validation_sources.insert(validation.source_ids.begin(),validation.source_ids.end());
      std::vector<Split> views;
      for(const auto &view:input.validation_views) {
        views.push_back(load_split(input,run,false,&view));validate_view(views.back(),validation);
      }
      validation_views.push_back(std::move(views));
      splits.emplace_back(std::move(training),std::move(validation));
    }
    for(const auto &id:train_sources)require(!validation_sources.count(id),"source group overlaps TRAIN and VALIDATION roles");
    std::ostringstream report;report << "{\"version\":1,\"protocol\":\"archive-readout-v1\",\"stage\":\"development\",\"policy_version\":\"1.2\",\"validation_only\":true,"
        << "\"test_access\":false,\"encoder_training\":false,\"post_encoder_pca\":false,\"card_file\":\"archive-readout-card.json\","
        << "\"input_manifest_file\":\"input-manifest.json\",\"output_manifest_file\":\"output-manifest.json\",\"source_fingerprint\":" << quote(run.source_fingerprint)
        << ",\"row_association\":\"inherited frozen producer row order; exact file hashes and caller-audited cohort lineage; native row identity not independently embedded\",\"inputs\":[";
    for(size_t index=0;index<run.inputs.size();++index) {
      if(index)report << ',';
      const auto &input=run.inputs[index];const auto &[training,validation]=splits[index];
      const auto &views=validation_views[index];
      std::array<std::vector<FeatureSurface>,3> view_features;
      const fs::path directory=input.id;fs::create_directory(outputs.root/directory);
      if(!views.empty()) {
        fs::create_directory(outputs.root/directory/"validation-views");
        for(size_t i=0;i<views.size();++i) {
          const auto view_dir=directory/"validation-views"/input.validation_views[i].id;
          fs::create_directory(outputs.root/view_dir);
          save_features(outputs,view_dir/"native-features.pt",views[i].native,views[i]);
        }
      }
      std::vector<Method> methods;
      methods.push_back({"raw","Raw data — no encoder","",2*run.shape.channel_count*run.shape.history_length*run.shape.input_width,{},{},nullptr,{},{}});
      methods.push_back({"pca_only","PCA only — no encoder","",run.compact_width,{},{},nullptr,{},{}});
      methods.push_back({"native",input.tag,"",run.compact_width,{},{},nullptr,{},{}});
      const auto raw_training_valid=training.observed.feature_mask.flatten(1).any(1);
      const auto raw_validation_valid=validation.observed.feature_mask.flatten(1).any(1);
      for(size_t i=0;i<methods.size();++i) {
        methods[i].training_support=(i==2?training.native.valid:raw_training_valid).clone();
        methods[i].validation_support=(i==2?validation.native.valid:raw_validation_valid).clone();
        if(methods[i].training_support.sum().item<int64_t>()<2 || !both_classes(training.labels,methods[i].training_support))
          methods[i].reason="fewer than two valid TRAIN rows or a training class is absent";
      }
      save_features(outputs,directory/"native-training.pt",training.native,training);
      save_features(outputs,directory/"native-validation.pt",validation.native,validation);
      std::optional<int64_t> raw_pca_rank;
      if(methods[2].supported()) {
        auto &native=methods[2];native.outer=std::make_unique<FeatureNormalizer>(training.native);
        native.training=native.outer->transform(training.native);native.validation=native.outer->transform(validation.native);
        for(const auto &view:views)view_features[2].push_back(native.outer->transform(view.native));
        save_normalizer(outputs,directory/"native-normalizer.pt",*native.outer);
      }
      if(methods[0].supported()) {
        if(!training.observed.feature_mask.sum(std::vector<int64_t>{0,2}).gt(0).all().item<bool>()) {
          methods[0].reason=methods[1].reason="a channel/feature has no observed TRAIN values for ObservationScaler";
        } else {
          ObservationScaler scaler(training.observed);
          torch::serialize::OutputArchive scaler_asset;scaler_asset.write("mean",scaler.mean,true);scaler_asset.write("scale",scaler.scale,true);
          scaler_asset.write("counts",scaler.counts,true);outputs.archive(directory/"raw-scaler.pt",scaler_asset);
          const auto prepared_training=scaler.transform(training.observed),prepared_validation=scaler.transform(validation.observed);
          auto raw_surface=[](const Batch &prepared,const Split &split) {
            return FeatureSurface{torch::cat({prepared.data.flatten(1),prepared.feature_mask.to(torch::kFloat64).flatten(1)},1),
                split.observed.feature_mask.flatten(1).any(1),"TRAIN-only ObservationScaler per-channel/feature float64; masked values plus original flags; no encoder"};
          };
          const auto raw_training=raw_surface(prepared_training,training),raw_validation=raw_surface(prepared_validation,validation);
          save_features(outputs,directory/"raw-training.pt",raw_training,training);save_features(outputs,directory/"raw-validation.pt",raw_validation,validation);
          auto &raw=methods[0];raw.outer=std::make_unique<FeatureNormalizer>(raw_training);
          raw.training=raw.outer->transform(raw_training);raw.validation=raw.outer->transform(raw_validation);
          for(size_t i=0;i<views.size();++i) {
            const auto raw_view=raw_surface(scaler.transform(views[i].observed),views[i]);
            save_features(outputs,directory/"validation-views"/input.validation_views[i].id/"raw-features.pt",raw_view,views[i]);
            view_features[0].push_back(raw.outer->transform(raw_view));
          }
          save_normalizer(outputs,directory/"raw-normalizer.pt",*raw.outer);
          auto &pca_method=methods[1];
          // Reuse precisely the raw TRAIN normalizer; PCA never sees native exports.
          pca_method.outer=std::make_unique<FeatureNormalizer>(*raw.outer);
          try {
            TrainPca pca(raw.training,run.compact_width);raw_pca_rank=pca.numerical_rank;
            pca_method.training=pca.transform(raw.training);pca_method.validation=pca.transform(raw.validation);
            for(size_t i=0;i<views.size();++i) {
              view_features[1].push_back(pca.transform(view_features[0][i]));
              save_features(outputs,directory/"validation-views"/input.validation_views[i].id/"pca-features.pt",view_features[1].back(),views[i]);
            }
            validate_features(pca_method.training);validate_features(pca_method.validation);
            torch::serialize::OutputArchive pca_asset;pca_asset.write("pca_mean",pca.mean,true);pca_asset.write("pca_components",pca.components,true);
            pca_asset.write("pca_singular_values",pca.singular_values,true);pca_asset.write("pca_numerical_rank",torch::tensor(pca.numerical_rank),true);
            pca_asset.write("fitted_rows",torch::tensor(pca.fitted_rows),true);outputs.archive(directory/"pca-only.pt",pca_asset);
            save_features(outputs,directory/"pca-training.pt",pca_method.training,training);save_features(outputs,directory/"pca-validation.pt",pca_method.validation,validation);
          } catch(const std::runtime_error &error) {
            const std::string message=error.what();
            require(message.find("PCA dimensions exceed")!=std::string::npos,message);
            pca_method.reason=message;
          }
        }
      }
      report << "{\"input\":" << input_json(input) << ",\"native_provenance\":" << quote(training.native.provenance)
          << ",\"raw_training_population\":" << population(training,raw_training_valid) << ",\"raw_validation_population\":" << population(validation,raw_validation_valid)
          << ",\"native_training_population\":" << population(training,training.native.valid) << ",\"native_validation_population\":" << population(validation,validation.native.valid)
          << ",\"raw_pca_numerical_rank\":";
      if(raw_pca_rank)report << *raw_pca_rank;else report << "null";
      report << ",\"repetitions\":[";
      for(size_t rep_index=0;rep_index<run.repetitions.size();++rep_index) {
        if(rep_index)report << ',';
        const auto &rep=run.repetitions[rep_index];const auto rep_dir=directory/rep.id;fs::create_directory(outputs.root/rep_dir);
        report << "{\"id\":" << quote(rep.id) << ",\"methods\":[";
        std::vector<std::unique_ptr<Readout>> readouts;
        for(size_t method_index=0;method_index<methods.size();++method_index) {
          if(method_index)report << ',';
          const auto &method=methods[method_index];const auto seed=stream_seed(rep.probe_seed,uint64_t(method.width));
          report << "{\"method\":" << quote(method.id) << ",\"label\":" << quote(method.label) << ",\"size\":" << method.width
              << ",\"actual_probe_seed\":" << quote(std::to_string(seed)) << ",\"ridge_parameters\":" << 2*method.width+2
              << ",\"neural_parameters\":" << run.tiny_hidden*(method.width+3)+2
              << ",\"training_population\":" << population(training,method.training_support)
              << ",\"validation_population\":" << population(validation,method.validation_support);
          if(!method.supported()) { report << ",\"status\":\"unsupported\",\"reason\":" << quote(method.reason) << ",\"ridge\":null,\"tiny_secondary\":null}";readouts.push_back(nullptr);continue; }
          auto readout=std::make_unique<Readout>(method,training,run,seed);
          const auto fit=rep_dir/(method.id+"-fit.pt"),train_predictions=rep_dir/(method.id+"-training-predictions.pt"),val_predictions=rep_dir/(method.id+"-validation-predictions.pt");
          save_readout(outputs,fit,*readout,method);
          save_predictions(outputs,train_predictions,readout->training_ridge,readout->training_tiny,method.training,training);
          save_predictions(outputs,val_predictions,readout->validation_ridge,readout->validation_tiny,method.validation,validation);
          auto probe_json=[&](const torch::Tensor &train,const torch::Tensor &valid,uint64_t stream) {
            return "{\"training\":"+score_json(score(train,training.labels,method.training.valid))+",\"validation\":"+score_json(score(valid,validation.labels,method.validation.valid))+
                ",\"validation_interval\":"+interval_json(grouped_accuracy_interval(valid,validation.labels,method.validation.valid,validation.source_ids,stream_seed(seed,stream),run.bootstrap_replicates),run.bootstrap_replicates)+'}';
          };
          report << ",\"status\":\"measured\",\"fit_file\":" << quote(fit.generic_string()) << ",\"training_predictions_file\":" << quote(train_predictions.generic_string())
              << ",\"validation_predictions_file\":" << quote(val_predictions.generic_string())
              << ",\"ridge\":" << probe_json(readout->training_ridge,readout->validation_ridge,0x7269646765ULL)
              << ",\"tiny_secondary\":" << probe_json(readout->training_tiny,readout->validation_tiny,0x74696e79ULL) << '}';
          readouts.push_back(std::move(readout));
        }
        report << "],\"paired_comparisons\":[";
        const std::array<std::pair<size_t,size_t>,3> pairs{{{2,1},{2,0},{1,0}}};
        for(size_t pair_index=0;pair_index<pairs.size();++pair_index) {
          if(pair_index)report << ',';
          const auto [left,right]=pairs[pair_index];
          const auto common=methods[left].validation_support.logical_and(methods[right].validation_support);
          report << "{\"id\":" << quote(methods[left].id+"_minus_"+methods[right].id) << ",\"candidate\":" << quote(methods[left].id)
              << ",\"comparator\":" << quote(methods[right].id) << ",\"common_population\":" << population(validation,common);
          if(!readouts[left] || !readouts[right]) {report << ",\"status\":\"unsupported\",\"reason\":\"at least one declared fit unsupported\",\"ridge\":null,\"tiny_secondary\":null}";continue;}
          const auto seed=stream_seed(rep.probe_seed,0x706169720000ULL+pair_index);
          auto paired=[&](const torch::Tensor &candidate,const torch::Tensor &comparator,uint64_t stream) {
            return "{\"candidate\":"+score_json(score(candidate,validation.labels,common))+",\"comparator\":"+score_json(score(comparator,validation.labels,common))+
                ",\"candidate_minus_comparator_interval\":"+interval_json(grouped_accuracy_interval(candidate,validation.labels,common,validation.source_ids,stream_seed(seed,stream),run.bootstrap_replicates,comparator),run.bootstrap_replicates)+'}';
          };
          report << ",\"status\":\"measured\",\"ridge\":" << paired(readouts[left]->validation_ridge,readouts[right]->validation_ridge,0x7269646765ULL)
              << ",\"tiny_secondary\":" << paired(readouts[left]->validation_tiny,readouts[right]->validation_tiny,0x74696e79ULL) << '}';
        }
        report << ']';
        write_validation_views(outputs,report,input,run,rep,rep_dir,views,methods,view_features,readouts);
        report << '}';
      }
      report << "]}";
    }
    guard.verify();report << "],\"original_inputs_byte_preserved\":true}";outputs.text("report.json",report.str());
    outputs.text("output-manifest.json",outputs.manifest());guard.verify();
  } catch(...) {guard.verify();throw;}
}

} // namespace embedding::evaluation
