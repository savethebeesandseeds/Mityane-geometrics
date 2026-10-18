// SPDX-License-Identifier: MIT
#include "configurable_capacity.h"
#include "embedding/shared/data.h"
#include "embedding/shared/feature_harness.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <linux/fs.h>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

namespace embedding::evaluation::configurable_capacity {
namespace {
constexpr double tau=6.283185307179586476925286766559;
void require(bool condition,const char *why) {
  if(!condition)throw std::runtime_error(std::string("[configurable-capacity] ")+why);
}
// Observation archives are immutable evidence. The historical shared publisher
// intentionally replaces files; keep this protocol's exclusive publication
// local to the benchmark, with no dependency on an encoder implementation.
class StagedObservation {
public:
  explicit StagedObservation(const std::string &destination):destination_(destination) {
    const auto target=std::filesystem::path(destination);
    require(!destination.empty()&&!target.filename().empty(),"observation filename required");
    const auto parent=target.parent_path().empty()?std::filesystem::path("."):target.parent_path();
    std::filesystem::create_directories(parent);
    auto pattern=(parent/(target.filename().string()+".tmp-XXXXXX")).string();
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back('\0');
    descriptor_=::mkstemp(name.data());
    require(descriptor_>=0,"cannot reserve observation staging file");staged_=name.data();
  }
  StagedObservation(const StagedObservation &)=delete;
  StagedObservation &operator=(const StagedObservation &)=delete;
  ~StagedObservation() {
    if(descriptor_>=0)::close(descriptor_);
    if(!staged_.empty())::unlink(staged_.c_str());
  }
  void write(torch::serialize::OutputArchive &archive) {
    archive.save_to([this](const void *data,size_t size) {
      const auto *bytes=static_cast<const char *>(data);size_t offset=0;
      while(offset<size) {
        const auto count=::write(descriptor_,bytes+offset,std::min(size-offset,size_t(1024*1024)));
        if(count<0&&errno==EINTR)continue;
        require(count>0,"observation staging write failed");offset+=static_cast<size_t>(count);
      }
      return size;
    });
    require(::fsync(descriptor_)==0,"cannot sync staged observation");
    const auto descriptor=descriptor_;descriptor_=-1;
    require(::close(descriptor)==0,"cannot close staged observation");
    if(::syscall(SYS_renameat2,AT_FDCWD,staged_.c_str(),AT_FDCWD,destination_.c_str(),RENAME_NOREPLACE)==0) {
      staged_.clear();return;
    }
    const auto error=errno;
    if(error==ENOSYS||error==EINVAL||error==EOPNOTSUPP) {
      // The complete staging inode and destination share a filesystem. Link
      // refuses existing files and dangling symlinks on Windows Docker binds.
      require(::link(staged_.c_str(),destination_.c_str())==0,"observation destination exists or publication failed");
      return;
    }
    throw std::runtime_error(std::string("[configurable-capacity] immutable observation publication failed: ")+std::strerror(error));
  }
private:
  std::string destination_,staged_;
  int descriptor_=-1;
};
double uniform53(std::mt19937_64 &rng) {
  return static_cast<double>(rng()>>11)*(1.0/9007199254740992.0);
}
uint64_t bounded(std::mt19937_64 &rng,uint64_t limit) {
  const auto threshold=(uint64_t(0)-limit)%limit;
  uint64_t draw;do{draw=rng();}while(draw<threshold);return draw%limit;
}
double gaussian(std::mt19937_64 &rng) {
  const double u1=1-uniform53(rng),u2=uniform53(rng);
  return std::sqrt(-2*std::log(u1))*std::cos(tau*u2);
}
std::mt19937_64 role_rng(uint64_t namespaced,uint64_t role,uint64_t source) {
  return std::mt19937_64(stream_seed(stream_seed(namespaced,role),source+1));
}
Split assemble(const std::vector<torch::Tensor> &values,const std::vector<torch::Tensor> &masks,
    const std::vector<int64_t> &labels,const std::vector<int64_t> &ids) {
  auto mask=torch::stack(masks);
  auto full=torch::stack(values);
  return {torch::where(mask,full,torch::zeros_like(full)).contiguous(),mask.contiguous(),
      torch::tensor(labels,torch::kInt64),torch::tensor(ids,torch::kInt64)};
}
Development generate(std::size_t training_pairs,std::size_t validation_pairs,uint64_t master,
    const Dimensions &dimensions,double noise_sd) {
  validate_dimensions(dimensions);
  constexpr std::size_t source_limit=std::size_t(1)<<30;
  require(master<(uint64_t(1)<<32)&&training_pairs>0&&validation_pairs>0&&
      training_pairs<source_limit&&validation_pairs<source_limit&&
      training_pairs<source_limit-validation_pairs,"bounded injective seed/source domain required before allocation");
  const auto total=training_pairs+validation_pairs;
  const auto namespaced=stream_seed(master,kTaskStream);
  auto split_rng=std::mt19937_64(stream_seed(namespaced,kSplitStream));
  std::vector<uint64_t> assignment(total);std::iota(assignment.begin(),assignment.end(),uint64_t(0));
  for(uint64_t i=total-1;i>0;--i)std::swap(assignment[i],assignment[bounded(split_rng,i+1)]);
  std::vector<bool> in_training(total,false);
  for(std::size_t i=0;i<training_pairs;++i)in_training[assignment[i]]=true;
  std::vector<torch::Tensor> values[2],masks[2];std::vector<int64_t> labels[2],ids[2];
  const int64_t history=dimensions.history_length;
  for(uint64_t source=0;source<total;++source) {
    auto signal=role_rng(namespaced,kSignalStream,source),distractor=role_rng(namespaced,kDistractorStream,source);
    auto affine=role_rng(namespaced,kAffineStream,source),noise=role_rng(namespaced,kNoiseStream,source);
    auto natural=role_rng(namespaced,kMaskStream,source),gap_rng=role_rng(namespaced,kGapStream,source);
    auto order=role_rng(namespaced,kOrderStream,source);
    // Integer grid is declared before quality: no source-specific periods exit.
    const double ps=32+bounded(signal,17),pf=8+bounded(signal,7);
    const double phi_s=tau*uniform53(signal),phi_f=tau*uniform53(signal);
    const double lag_s=1.5+uniform53(signal),lag_f=.3+.4*uniform53(signal);
    const double fast_sign=uniform53(signal)>=.5?1:-1;
    const double slow_amplitude=.8+.4*uniform53(signal),fast_amplitude=.8+.4*uniform53(signal);
    const double ps2=32+bounded(distractor,17),pf2=8+bounded(distractor,7);
    const double phase_s2=tau*uniform53(distractor),phase_f2=tau*uniform53(distractor);
    std::array<double,3> nuisance_p{},nuisance_q{},nuisance_phi{},nuisance_psi{},nuisance_a{};
    for(int c=0;c<3;++c) {
      nuisance_p[c]=18+8*uniform53(distractor);nuisance_q[c]=70+20*uniform53(distractor);
      nuisance_phi[c]=tau*uniform53(distractor);nuisance_psi[c]=tau*uniform53(distractor);
      nuisance_a[c]=.6+.4*uniform53(distractor);
    }
    std::array<std::array<double,9>,3> gains{},offsets{};std::array<int64_t,3> gap{};
    for(int c=0;c<3;++c)for(int f=0;f<9;++f) {
      gains[c][f]=.5+uniform53(affine);offsets[c][f]=-.75+1.5*uniform53(affine);
    }
    for(int c=0;c<3;++c)gap[c]=static_cast<int64_t>((history-4)*uniform53(gap_rng));
    auto a=torch::zeros({3,history,9},torch::kFloat64),b=torch::zeros_like(a),mask=torch::zeros_like(a,torch::kBool);
    auto av=a.accessor<double,3>(),bv=b.accessor<double,3>();auto mv=mask.accessor<bool,3>();
    // Time-first draw order preserves noise/missingness prefixes as H grows.
    for(int64_t t=0;t<history;++t)for(int c=0;c<3;++c)for(int f=0;f<9;++f) {
      const double epsilon=noise_sd*gaussian(noise);
      mv[c][t][f]=uniform53(natural)>=.10&&!(t>=gap[c]&&t<gap[c]+5);
      const int group=f/3;
      const double nuisance=nuisance_a[c]*(std::sin(tau*t/nuisance_p[c]+nuisance_phi[c])+
          .6*std::sin(tau*t/nuisance_q[c]+nuisance_psi[c]));
      for(int label=0;label<2;++label) {
        double slow,fast;
        if(c==2) {
          slow=slow_amplitude*std::sin(tau*t/ps2+phase_s2);
          fast=fast_amplitude*std::sin(tau*t/pf2+phase_f2);
        } else {
          const double slow_shift=c==1?(label?lag_s:-lag_s):0;
          const double fast_shift=c==1?fast_sign*lag_f:0;
          slow=slow_amplitude*std::sin(tau*(t+slow_shift)/ps+phi_s);
          fast=fast_amplitude*std::sin(tau*(t+fast_shift)/pf+phi_f);
        }
        const double mixture=group==0?slow+.25*fast:group==1?fast+.25*slow:nuisance;
        (label?bv:av)[c][t][f]=offsets[c][f]+gains[c][f]*(1+.05*(f%3))*mixture+epsilon;
      }
    }
    const int split=in_training[source]?0:1;const bool swap=uniform53(order)>=.5;
    const auto id=static_cast<int64_t>((master<<31)|(source<<1));
    for(int variant=0;variant<2;++variant) {
      const int label=swap?1-variant:variant;
      values[split].push_back(label?b:a);masks[split].push_back(mask.clone());
      labels[split].push_back(label);ids[split].push_back(id);
    }
  }
  return {assemble(values[0],masks[0],labels[0],ids[0]),assemble(values[1],masks[1],labels[1],ids[1])};
}
} // namespace

void validate_dimensions(const Dimensions &d) {
  require(d.channel_count==3&&d.input_width==9&&d.history_length>=64&&d.history_length<=4096,
      "benchmark requires semantic C3/F9 and history in[64,4096]");
}
void validate(const Split &source) {
  const auto &x=source.observed,&mask=source.mask,&labels=source.scoring_labels,&ids=source.source_ids;
  require(x.defined()&&x.device().is_cpu()&&x.scalar_type()==torch::kFloat64&&x.is_contiguous()&&
      x.dim()==4&&x.size(0)>0&&x.size(0)%2==0,"paired contiguous CPU F64 rank4 observations required");
  validate_dimensions({x.size(1),x.size(2),x.size(3)});
  require(mask.defined()&&mask.device().is_cpu()&&mask.scalar_type()==torch::kBool&&
      mask.is_contiguous()&&mask.sizes()==x.sizes(),"CPU Bool observation mask required");
  require(labels.defined()&&ids.defined()&&labels.device().is_cpu()&&ids.device().is_cpu()&&
      labels.scalar_type()==torch::kInt64&&ids.scalar_type()==torch::kInt64&&labels.is_contiguous()&&ids.is_contiguous()&&
      labels.sizes()==torch::IntArrayRef({x.size(0)})&&ids.sizes()==labels.sizes(),"CPU Long labels and IDs required");
  require(torch::isfinite(x).all().item<bool>()&&x.masked_select(mask.logical_not()).eq(0).all().item<bool>(),
      "finite observations with exactly zero hidden storage required");
  std::map<int64_t,std::vector<int64_t>> groups;
  auto y=labels.accessor<int64_t,1>(),s=ids.accessor<int64_t,1>();
  for(int64_t row=0;row<x.size(0);++row) {
    require(s[row]>=0&&(s[row]&1)==0&&(y[row]==0||y[row]==1),
        "nonnegative packed source IDs with zero variant bit and binary labels required");
    groups[s[row]].push_back(row);
  }
  for(const auto &[id,pair]:groups) {
    (void)id;require(pair.size()==2&&y[pair[0]]!=y[pair[1]]&&torch::equal(mask[pair[0]],mask[pair[1]]),
        "opposite-label whole source pairs with shared masks required");
  }
}
Development make_development(std::size_t training_pairs,std::size_t validation_pairs,uint64_t master,Dimensions d) {
  return generate(training_pairs,validation_pairs,master,d,.005);
}
Split delete_observations(const Split &source,double probability,uint64_t seed) {
  require(std::isfinite(probability)&&probability>=0&&probability<=1,"finite deletion probability in[0,1] required");
  validate(source);
  std::map<int64_t,torch::Tensor> erased;std::vector<torch::Tensor> rows;
  auto ids=source.source_ids.accessor<int64_t,1>();
  for(int64_t row=0;row<source.observed.size(0);++row) {
    const auto id=ids[row];
    if(!erased.count(id)) {
      std::mt19937_64 rng(stream_seed(stream_seed(seed,kDeletionStream),static_cast<uint64_t>(id)));
      auto e=torch::zeros(source.observed.sizes().slice(1),torch::kBool);auto m=e.accessor<bool,3>();
      for(int64_t t=0;t<e.size(1);++t)for(int64_t c=0;c<e.size(0);++c)for(int64_t f=0;f<e.size(2);++f)
        m[c][t][f]=uniform53(rng)<probability;
      erased.emplace(id,e);
    }
    rows.push_back(erased.at(id));
  }
  auto mask=source.mask.logical_and(torch::stack(rows).logical_not());
  return {torch::where(mask,source.observed,torch::zeros_like(source.observed)),mask,
      source.scoring_labels.clone(),source.source_ids.clone()};
}
std::string source_ids_json(const Split &source) {
  validate(source);std::ostringstream out;out<<'[';
  auto ids=source.source_ids.accessor<int64_t,1>();
  for(int64_t row=0;row<source.observed.size(0);++row) {
    if(row)out<<',';
    out<<'"'<<kDatasetId<<'/'<<ids[row]<<'"';
  }
  out<<']';return out.str();
}
void publish_archive(const std::string &path,torch::serialize::OutputArchive &archive) {
  require(!std::filesystem::exists(path)&&!std::filesystem::is_symlink(path),"evidence archive already exists");
  StagedObservation(path).write(archive);
}
void save_split_archive(const std::string &path,const Split &source,const std::string &role,
    uint64_t master,double additional_deletion) {
  validate(source);
  require(!std::filesystem::exists(path)&&!std::filesystem::is_symlink(path),"observation archive already exists");
  require(role=="TRAIN"||role=="VALIDATION"||role=="VALIDATION_DELETED","explicit data role required");
  require(master<(uint64_t(1)<<32)&&std::isfinite(additional_deletion)&&additional_deletion>=0&&additional_deletion<=1,
      "archive seed/deletion metadata invalid");
  require(source.source_ids.bitwise_right_shift(31).eq(static_cast<int64_t>(master)).all().item<bool>(),
      "archive master disagrees with packed source IDs");
  torch::serialize::OutputArchive out;
  out.write("observed",source.observed,true);out.write("feature_mask",source.mask,true);
  out.write("labels_scoring_only",source.scoring_labels,true);out.write("source_ids",source.source_ids,true);
  out.write("source_ids_json",archive::text_tensor(source_ids_json(source)),true);
  publish_archive(path,out);
}
Split load_split_archive(const std::string &path) {
  torch::serialize::InputArchive in;in.load_from(path,torch::kCPU);Split s;
  const auto keys=in.keys();
  require(std::set<std::string>(keys.begin(),keys.end())==std::set<std::string>{
      "observed","feature_mask","labels_scoring_only","source_ids","source_ids_json"},
      "exact five-key legal observation schema required");
  in.read("observed",s.observed,true);in.read("feature_mask",s.mask,true);
  in.read("labels_scoring_only",s.scoring_labels,true);in.read("source_ids",s.source_ids,true);
  validate(s);
  torch::Tensor ids;in.read("source_ids_json",ids,true);
  require(archive::tensor_text(ids)==source_ids_json(s),"archive source-ID metadata mismatch");
  return s;
}
namespace engineering {
Development make_noiseless_development(std::size_t training_pairs,std::size_t validation_pairs,uint64_t master,Dimensions d) {
  return generate(training_pairs,validation_pairs,master,d,0);
}
}
} // namespace embedding::evaluation::configurable_capacity
