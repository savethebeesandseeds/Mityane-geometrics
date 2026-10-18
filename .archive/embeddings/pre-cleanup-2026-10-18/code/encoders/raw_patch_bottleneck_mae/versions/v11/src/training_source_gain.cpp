// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include <cmath>
#include <iomanip>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace embedding::encoders::raw_patch_bottleneck_mae::training_source_gain {
namespace {
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[rpb source gain] " + message);
}
std::string source_text(const std::vector<std::string> &ids) {
  std::string result;
  for (const auto &id : ids) result += std::to_string(id.size()) + ':' + id;
  return result;
}
std::string json_strings(const std::vector<std::string> &ids) {
  std::ostringstream out; out << '[';
  for (size_t i=0;i<ids.size();++i) {
    if (i) out << ','; out << '"';
    for (unsigned char ch:ids[i]) {
      if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
      else if (ch < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(ch) << std::dec;
      else out << static_cast<char>(ch);
    }
    out << '"';
  }
  out << ']'; return out.str();
}
void validate_manifest(const Manifest &m) {
  require(m.recipe == TrainingSourceGainRecipe::unit_control_v1 || m.recipe == TrainingSourceGainRecipe::source_log2_v1,
      "enabled closed manifest recipe required");
  require(m.actual_training_seed <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),"gain seed must fit signed int64");
  const auto groups=static_cast<int64_t>(m.source_ids.size());
  require(groups >= 2,"at least two source groups required");
  for (int64_t i=0;i<groups;++i) require(!m.source_ids[i].empty() && (i==0 || m.source_ids[i-1]<m.source_ids[i]),"strict lexical unique source identities required");
  const auto tensor=[&](const torch::Tensor &value,torch::ScalarType dtype,int64_t size) {
    require(value.defined() && value.device().is_cpu() && value.scalar_type()==dtype && value.dim()==1 &&
        value.size(0)==size && value.is_contiguous(),"closed contiguous CPU gain manifest array required");
  };
  tensor(m.source_ranks,torch::kInt64,groups); tensor(m.row_source_ranks,torch::kInt64,groups*2);
  const auto draws=m.recipe==TrainingSourceGainRecipe::source_log2_v1?groups:0;
  tensor(m.source_top53,torch::kInt64,draws); tensor(m.source_uniforms,torch::kFloat64,draws);
  tensor(m.source_gains,torch::kFloat64,groups); tensor(m.row_gains,torch::kFloat64,groups*2);
  require(torch::equal(m.source_ranks,torch::arange(groups,torch::kInt64)) &&
      (m.row_source_ranks>=0).all().item<bool>() && (m.row_source_ranks<groups).all().item<bool>() &&
      torch::isfinite(m.source_gains).all().item<bool>() && (m.source_gains>=.5).all().item<bool>() && (m.source_gains<=2).all().item<bool>() &&
      torch::isfinite(m.row_gains).all().item<bool>() && torch::equal(m.row_gains,m.source_gains.index_select(0,m.row_source_ranks)) &&
      (m.source_top53>=0).all().item<bool>() && (m.source_top53<9007199254740992LL).all().item<bool>() &&
      torch::isfinite(m.source_uniforms).all().item<bool>() && (m.source_uniforms>=0).all().item<bool>() && (m.source_uniforms<1).all().item<bool>(),
      "gain ranks, finite bounds or row association differ");
  std::vector<int64_t> counts(static_cast<size_t>(groups)); const auto rows=m.row_source_ranks.accessor<int64_t,1>();
  for (int64_t i=0;i<groups*2;++i) ++counts[static_cast<size_t>(rows[i])];
  for (auto count:counts) require(count==2,"exactly two rows per source required");
  require(draws || torch::equal(m.source_gains,torch::ones_like(m.source_gains)),"unit control gains must be exact ones");
}
std::string identity(const Manifest &m) {
  // Validate all array storage before any raw host-byte access.
  validate_manifest(m);
  uint64_t hash = 14695981039346656037ULL;
  const auto add = [&hash](const void *bytes, size_t n) {
    const auto *data = static_cast<const unsigned char *>(bytes);
    for (size_t i=0;i<n;++i) { hash ^= data[i]; hash *= 1099511628211ULL; }
  };
  const auto prefix = std::string(recipe_name(m.recipe)) + ':' + std::to_string(m.actual_training_seed) + ':' + source_text(m.source_ids);
  add(prefix.data(), prefix.size());
  // Closed CPU contiguous native Long/Double arrays, in this declared order.
  for (const auto &value : {m.source_ranks,m.row_source_ranks,m.source_top53,m.source_uniforms,m.source_gains,m.row_gains}) {
    const auto shape = value.sizes().vec(); const auto length = static_cast<int64_t>(value.numel());
    add(&length,sizeof(length)); for (auto d:shape) add(&d,sizeof(d));
    if (length) add(value.const_data_ptr(),value.nbytes());
  }
  std::ostringstream out; out << "rpb-source-gain-manifest-fnv1a-v1-" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
void text(torch::serialize::OutputArchive &a,const char *key,const std::string &value) {
  a.write(key,embedding::archive::text_tensor(value),true);
}
std::string text(torch::serialize::InputArchive &a,const char *key) {
  torch::Tensor value; a.read(key,value,true); return embedding::archive::tensor_text(value);
}
} // namespace
void validate_options(TrainingSourceGainOptions options) {
  require(options.recipe == TrainingSourceGainRecipe::disabled || options.recipe == TrainingSourceGainRecipe::unit_control_v1 ||
      options.recipe == TrainingSourceGainRecipe::source_log2_v1,"explicit closed source-gain recipe required");
}
bool enabled(TrainingSourceGainOptions options) { validate_options(options); return options.recipe != TrainingSourceGainRecipe::disabled; }
bool gained(TrainingSourceGainOptions options) { validate_options(options); return options.recipe == TrainingSourceGainRecipe::source_log2_v1; }
const char *recipe_name(TrainingSourceGainRecipe recipe) {
  validate_options({recipe});
  return recipe == TrainingSourceGainRecipe::disabled ? "disabled" : recipe == TrainingSourceGainRecipe::unit_control_v1 ? "unit_control_v1" : "source_log2_v1";
}
Manifest make_manifest(const std::vector<std::string> &ids,uint64_t seed,TrainingSourceGainOptions options) {
  require(enabled(options),"disabled recipe has no gain manifest");
  require(seed <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),"gain seed must fit signed int64");
  std::map<std::string,int64_t> counts;
  for (const auto &id:ids) { require(!id.empty(),"empty source identity"); ++counts[id]; }
  require(counts.size() >= 2,"at least two distinct source groups required");
  Manifest m; m.recipe=options.recipe; m.actual_training_seed=seed;
  std::map<std::string,int64_t> ranks; std::vector<int64_t> row_ranks,top;
  std::vector<double> uniforms,gains;
  for (const auto &[id,count]:counts) {
    require(count == 2,"exactly two rows per source required without label access");
    const auto rank=static_cast<int64_t>(m.source_ids.size()); ranks.emplace(id,rank); m.source_ids.push_back(id);
    if (gained(options)) {
      const auto bits=training_detail::counter_seed(seed,rank,stream)>>11;
      const double u=static_cast<double>(bits)*0x1p-53;
      const double g=std::exp((2.0*u-1.0)*ln2);
      require(std::isfinite(g) && g >= .5 && g <= 2,"finite bounded source gain required");
      top.push_back(static_cast<int64_t>(bits)); uniforms.push_back(u); gains.push_back(g);
    } else gains.push_back(1);
  }
  for (const auto &id:ids) row_ranks.push_back(ranks.at(id));
  m.source_ranks=torch::arange(static_cast<int64_t>(counts.size()),torch::kInt64);
  m.row_source_ranks=torch::tensor(row_ranks,torch::kInt64).contiguous();
  m.source_top53=torch::tensor(top,torch::kInt64).contiguous();
  m.source_uniforms=torch::tensor(uniforms,torch::kFloat64).contiguous();
  m.source_gains=torch::tensor(gains,torch::kFloat64).contiguous();
  m.row_gains=m.source_gains.index_select(0,m.row_source_ranks).contiguous(); m.identity=identity(m); return m;
}
Input apply(const Input &original,const torch::Tensor &indices,const Manifest &m) {
  require(m.recipe == TrainingSourceGainRecipe::source_log2_v1 && m.identity == identity(m),"valid candidate manifest required");
  require(original.data.device().is_cpu() && original.data.scalar_type() == torch::kFloat64 && original.data.dim() == 4 &&
      original.observed.device().is_cpu() && original.observed.scalar_type() == torch::kBool && original.observed.sizes() == original.data.sizes() &&
      indices.device().is_cpu() && indices.scalar_type() == torch::kInt64 && indices.dim() == 1 && indices.size(0) == original.data.size(0) &&
      (indices >= 0).all().item<bool>() && (indices < m.row_gains.size(0)).all().item<bool>(),"CPU float64 selected rows and valid row indices required");
  auto out=original;
  const auto gains=m.row_gains.index_select(0,indices).reshape({-1,1,1,1});
  // Erase absent storage before multiplication, so even hidden NaN is harmless.
  out.data=torch::where(original.observed,original.data,torch::zeros_like(original.data))*gains;
  require(torch::isfinite(out.data).all().item<bool>(),"gained observed values must be finite");
  return out;
}
void write_manifest(torch::serialize::OutputArchive &a,const Manifest &m) {
  require(m.identity == identity(m),"gain manifest mutated");
  text(a,"source_ids_json",json_strings(m.source_ids));
  text(a,"gain_manifest_id",m.identity); text(a,"source_gain_recipe",recipe_name(m.recipe));
  a.write("actual_training_seed_value",torch::tensor(static_cast<int64_t>(m.actual_training_seed)),true);
  for (const auto &[name,value]:std::map<std::string,torch::Tensor>{{"source_ranks",m.source_ranks},{"row_source_ranks",m.row_source_ranks},
      {"source_top53",m.source_top53},{"source_uniforms",m.source_uniforms},{"source_gains",m.source_gains},{"row_gains",m.row_gains}}) a.write(name,value,true);
}
void verify_manifest(torch::serialize::InputArchive &a,const Manifest &m) {
  require(text(a,"source_ids_json") == json_strings(m.source_ids) && text(a,"gain_manifest_id") == m.identity &&
      text(a,"source_gain_recipe") == recipe_name(m.recipe),"gain source/recipe/manifest identity differs");
  torch::Tensor seed; a.read("actual_training_seed_value",seed,true);
  require(seed.device().is_cpu() && seed.scalar_type() == torch::kInt64 && seed.dim() == 0 &&
      seed.item<int64_t>() == static_cast<int64_t>(m.actual_training_seed),"typed gain seed differs");
  for (const auto &[name,value]:std::map<std::string,torch::Tensor>{{"source_ranks",m.source_ranks},{"row_source_ranks",m.row_source_ranks},
      {"source_top53",m.source_top53},{"source_uniforms",m.source_uniforms},{"source_gains",m.source_gains},{"row_gains",m.row_gains}}) {
    torch::Tensor saved; a.read(name,saved,true);
    require(saved.device().is_cpu() && saved.scalar_type() == value.scalar_type() && saved.sizes() == value.sizes() && torch::equal(saved,value),"typed gain manifest differs: "+name);
  }
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae::training_source_gain
