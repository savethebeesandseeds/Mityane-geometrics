// Copyright (c) 2026 Waajacu. All rights reserved.
// See ../../../../../../../../LICENSE for the copyright and permission notice.
#pragma once

#include "cuwacunu/latent_market/risk_targets.h"
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <concepts>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <system_error>
#include <tuple>
#include <type_traits>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cuwacunu::latent_market::v1 {

struct LedgerReadLimits {
  std::uint64_t maximum_string_bytes=1ull<<20;
  std::uint64_t maximum_sequence_elements=1ull<<26;
  std::uint64_t maximum_total_elements=1ull<<28;
  std::uint64_t maximum_total_bytes=1ull<<30;
  std::uint64_t maximum_allocated_bytes=1ull<<30;
};

namespace ledger_detail {

// Ordered fields are the physical v1 format, independent of host padding,
// endianness, size_t width, compiler ABI and locale. New schemas get new magic.
#define CUW_LM_FIELDS(Type, ...) \
  template<class Archive,class T> requires std::same_as<std::remove_cv_t<T>,Type> \
  void fields(Archive& a,T& x) { a(__VA_ARGS__); }

CUW_LM_FIELDS(RoleInterval,x.role,x.begin_ms,x.end_ms)
CUW_LM_FIELDS(TimelineIdentity,x.artifact_id,x.dataset_id,x.encoder_family,x.encoder_version,
  x.checkpoint_hash,x.scaler_hash,x.encoder_config_hash,x.source_hash,x.cache_hash,x.export_layout,
  x.availability_policy,x.source_closure_hash,x.executable_hash,x.encoder_fit_cutoff_ms,x.channel_ids,x.feature_names)
CUW_LM_FIELDS(WindowSupport,x.timestamps_ms,x.channels,x.features,x.observed)
CUW_LM_FIELDS(EmbeddingRow,x.window_id,x.endpoint_ms,x.available_ms,x.role,x.sample_valid,x.support,x.values)
CUW_LM_FIELDS(EmbeddingTimeline,x.identity,x.interval_ms,x.window_ticks,x.embedding_width,x.rows)
CUW_LM_FIELDS(SupportPolicy,x.require_complete_window,x.minimum_observed_fraction,x.minimum_channel_fraction,
  x.minimum_feature_fraction,x.maximum_missing_run_ticks,x.maximum_endpoint_staleness_ticks)
CUW_LM_FIELDS(SupportSummary,x.observed_values,x.expected_values,x.channel_observed,x.feature_observed,
  x.maximum_missing_run_ticks,x.endpoint_staleness_ticks,x.regular_timestamps,x.complete,x.eligible)
CUW_LM_FIELDS(PairSpec,x.history_steps,x.spacing_ticks,x.horizons_ticks,x.future_label_ticks,x.input_support,x.target_support)
CUW_LM_FIELDS(WindowAssociation,x.endpoint_ms,x.window_id,x.row_index,x.reasons,x.support)
CUW_LM_FIELDS(OverlapSummary,x.geometric_fraction,x.shared_observed_values,x.origin_observed_values,
  x.target_observed_values,x.mask_aware_shared_fraction)
CUW_LM_FIELDS(HorizonAssociation,x.horizon_ticks,x.target,x.reasons,x.eligible,x.overlap)
CUW_LM_FIELDS(DynamicsPair,x.pair_id,x.endpoint_ms,x.decision_ms,x.role,x.dependence_group,
  x.history,x.input_reasons,x.raw_begin_ms,x.raw_end_ms,x.futures)
CUW_LM_FIELDS(DynamicsPairs,x.timeline_id,x.spec,x.roles,x.rows)
CUW_LM_FIELDS(CloseSupport,x.endpoint_ms,x.available_ms,x.present,x.observed,x.positive_finite)
CUW_LM_FIELDS(RiskTargetRow,x.pair_id,x.endpoint_ms,x.decision_ms,x.role,x.channel_id,x.horizon_ticks,
  x.eligible,x.reasons,x.dispersion_1m,x.downside,x.mean_return,x.eligible_returns,x.expected_returns,
  x.available_ms,x.support,x.return_valid)
CUW_LM_FIELDS(FutureRiskTargets,x.timeline_id,x.source_hash,x.formula_id,x.dispersion_units,
  x.downside_units,x.interval_ms,x.horizons_ticks,x.rows)
#undef CUW_LM_FIELDS

template<class T> struct is_vector : std::false_type {};
template<class T,class A> struct is_vector<std::vector<T,A>> : std::true_type {};
template<class T> struct is_optional : std::false_type {};
template<class T> struct is_optional<std::optional<T>> : std::true_type {};

class Writer {
  std::ostream& stream_;
 public:
  explicit Writer(std::ostream& stream):stream_(stream) {}
  void bytes(const char* data,std::size_t count) {
    stream_.write(data,static_cast<std::streamsize>(count));
    if (!stream_) throw std::runtime_error("[latent-market v1] ledger write failed");
  }
  void u64(std::uint64_t n) {
    char bytes_le[8];
    for (unsigned i=0;i<8;++i) bytes_le[i]=static_cast<char>((n>>(i*8))&255);
    bytes(bytes_le,8);
  }
  template<class... T> void operator()(const T&... values) {(one(values),...);}
  template<class T> void one(const T& value) {
    if constexpr (std::same_as<T,bool>) u64(value?1:0);
    else if constexpr (std::is_enum_v<T>) u64(static_cast<std::uint64_t>(value));
    else if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) u64(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
      else u64(static_cast<std::uint64_t>(value));
    } else if constexpr (std::same_as<T,double>) {
      static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
      u64(std::bit_cast<std::uint64_t>(value));
    } else if constexpr (std::same_as<T,std::string>) {
      u64(value.size());bytes(value.data(),value.size());
    } else if constexpr (is_vector<T>::value) {
      u64(value.size());
      if constexpr (std::same_as<typename T::value_type,std::uint8_t>)
        bytes(reinterpret_cast<const char*>(value.data()),value.size());
      else for (const auto& element:value) one(element);
    } else if constexpr (is_optional<T>::value) {
      one(value.has_value());if (value) one(*value);
    } else fields(*this,value);
  }
};

class Reader {
  std::istream& stream_;
  LedgerReadLimits limits_;
  std::uint64_t bytes_=0,elements_=0,allocated_=0;
  void allocate(std::uint64_t count,std::uint64_t width) {
    temporal_detail::require(width==0 || count<=limits_.maximum_allocated_bytes/width,
                              "ledger allocation exceeds limit");
    const auto n=count*width;
    temporal_detail::require(allocated_<=limits_.maximum_allocated_bytes &&
      n<=limits_.maximum_allocated_bytes-allocated_,"ledger allocation budget exceeded");
    allocated_+=n;
  }
 public:
  Reader(std::istream& stream,LedgerReadLimits limits):stream_(stream),limits_(limits) {}
  void bytes(char* data,std::uint64_t count) {
    temporal_detail::require(count<=limits_.maximum_total_bytes-bytes_ &&
      count<=static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),"ledger byte budget exceeded");
    stream_.read(data,static_cast<std::streamsize>(count));
    if (!stream_) throw std::runtime_error("[latent-market v1] truncated ledger");
    bytes_+=count;
  }
  std::uint64_t u64() {
    unsigned char data[8];bytes(reinterpret_cast<char*>(data),8);
    std::uint64_t n=0;
    for (unsigned i=0;i<8;++i) n|=static_cast<std::uint64_t>(data[i])<<(i*8);
    return n;
  }
  template<class... T> void operator()(T&... values) {(one(values),...);}
  template<class T> void one(T& value) {
    if constexpr (std::same_as<T,bool>) {
      const auto n=u64();temporal_detail::require(n<=1,"noncanonical ledger boolean");value=n!=0;
    } else if constexpr (std::is_enum_v<T>) {
      const auto n=u64();
      temporal_detail::require(n<=static_cast<std::uint64_t>(std::numeric_limits<std::underlying_type_t<T>>::max()),
                                "ledger enum overflow");
      value=static_cast<T>(n);
    } else if constexpr (std::is_integral_v<T>) {
      const auto n=u64();
      if constexpr (std::is_signed_v<T>) {
        const auto signed_n=std::bit_cast<std::int64_t>(n);
        temporal_detail::require(signed_n>=std::numeric_limits<T>::min() && signed_n<=std::numeric_limits<T>::max(),
                                  "ledger signed integer overflow");value=static_cast<T>(signed_n);
      } else {
        temporal_detail::require(n<=std::numeric_limits<T>::max(),"ledger unsigned integer overflow");value=static_cast<T>(n);
      }
    } else if constexpr (std::same_as<T,double>) value=std::bit_cast<double>(u64());
    else if constexpr (std::same_as<T,std::string>) {
      const auto n=u64();
      temporal_detail::require(n<=limits_.maximum_string_bytes,"ledger string exceeds limit");
      allocate(n,1);value.resize(static_cast<std::size_t>(n));bytes(value.data(),n);
    } else if constexpr (is_vector<T>::value) {
      const auto n=u64();
      temporal_detail::require(n<=limits_.maximum_sequence_elements &&
        n<=limits_.maximum_total_elements-elements_,"ledger element budget exceeded");
      elements_+=n;allocate(n,sizeof(typename T::value_type));value.resize(static_cast<std::size_t>(n));
      if constexpr (std::same_as<typename T::value_type,std::uint8_t>)
        bytes(reinterpret_cast<char*>(value.data()),n);
      else for (auto& element:value) one(element);
    } else if constexpr (is_optional<T>::value) {
      bool present=false;one(present);
      if (present) {value.emplace();one(*value);} else value.reset();
    } else fields(*this,value);
  }
  void finish() {
    temporal_detail::require(stream_.peek()==std::char_traits<char>::eof(),"trailing bytes after ledger");
    if (stream_.bad()) throw std::runtime_error("[latent-market v1] ledger stream failed");
  }
};

template<class T> void write(std::ostream& stream,const char* magic,const T& value) {
  Writer writer(stream);writer.bytes(magic,8);writer.u64(1);writer(value);
}
template<class T> T read(std::istream& stream,const char* magic,LedgerReadLimits limits) {
  Reader reader(stream,limits);char found[8];reader.bytes(found,8);
  temporal_detail::require(std::memcmp(found,magic,8)==0 && reader.u64()==1,"wrong ledger kind/version");
  T value;reader(value);reader.finish();return value;
}

// The caller creates a fresh run leaf. Each file then publishes atomically by
// a same-filesystem hard link, which refuses an existing destination even if a
// second process races the initial check. No user file is replaced or removed.
template<class Write> void publish(const std::filesystem::path& destination,Write write) {
#ifdef __linux__
  namespace fs=std::filesystem;
  const auto parent=destination.has_parent_path()?destination.parent_path():fs::path(".");
  temporal_detail::require(fs::is_directory(parent),"publication parent must already exist");
  temporal_detail::require(!destination.filename().empty(),"publication filename is empty");
  static std::atomic<std::uint64_t> serial{0};
  fs::path stage;
  bool created=false;
  for (unsigned attempt=0;attempt<64 && !created;++attempt) {
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    stage=parent/("."+destination.filename().string()+".stage."+std::to_string(::getpid())+"."+
      std::to_string(stamp)+"."+std::to_string(serial.fetch_add(1)));
    std::error_code ec;created=fs::create_directory(stage,ec);
    if (ec) throw std::system_error(ec,"create ledger staging directory");
  }
  temporal_detail::require(created,"cannot reserve ledger staging directory");
  const auto payload=stage/"payload";
  try {
    {
      std::ofstream stream(payload,std::ios::binary|std::ios::out|std::ios::trunc);
      if (!stream) throw std::runtime_error("cannot create staged ledger");
      write(stream);stream.flush();
      if (!stream) throw std::runtime_error("cannot flush staged ledger");
      stream.close();if (!stream) throw std::runtime_error("cannot close staged ledger");
    }
    const int fd=::open(payload.c_str(),O_RDONLY|O_NOFOLLOW);
    if (fd<0) throw std::system_error(errno,std::generic_category(),"open staged ledger");
    const int sync_result=::fsync(fd);const int sync_errno=errno;::close(fd);
    if (sync_result!=0) throw std::system_error(sync_errno,std::generic_category(),"sync staged ledger");
    if (::link(payload.c_str(),destination.c_str())!=0)
      throw std::system_error(errno,std::generic_category(),"exclusive ledger publication");
    fs::remove(payload);fs::remove(stage);
    const int dirfd=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);
    if (dirfd>=0) {::fsync(dirfd);::close(dirfd);}
  } catch (...) {
    std::error_code ec;fs::remove(payload,ec);fs::remove(stage,ec);throw;
  }
#else
  (void)destination;(void)write;
  throw std::runtime_error("immutable ledger publication requires the managed Linux container");
#endif
}
} // namespace ledger_detail

inline void write_timeline(std::ostream& stream,const EmbeddingTimeline& timeline) {
  validate_timeline(timeline);ledger_detail::write(stream,"CUWLM.E1",timeline);
}
inline EmbeddingTimeline read_timeline(std::istream& stream,LedgerReadLimits limits={}) {
  auto out=ledger_detail::read<EmbeddingTimeline>(stream,"CUWLM.E1",limits);validate_timeline(out);return out;
}
inline void write_pairs(std::ostream& stream,const DynamicsPairs& pairs) {
  temporal_detail::require(!pairs.timeline_id.empty(),"empty pair timeline identity");
  validate_pair_spec(pairs.spec);validate_roles(pairs.roles);
  ledger_detail::write(stream,"CUWLM.P1",pairs);
}
inline DynamicsPairs read_pairs(std::istream& stream,LedgerReadLimits limits={}) {
  auto out=ledger_detail::read<DynamicsPairs>(stream,"CUWLM.P1",limits);
  temporal_detail::require(!out.timeline_id.empty(),"empty pair timeline identity");
  validate_pair_spec(out.spec);validate_roles(out.roles);return out;
}
inline DynamicsPairs read_pairs(std::istream& stream,const EmbeddingTimeline& timeline,LedgerReadLimits limits={}) {
  auto out=read_pairs(stream,limits);validate_pairs(timeline,out);return out;
}

inline void validate_risk_targets(const FutureRiskTargets& targets) {
  using namespace temporal_detail;
  require(!targets.timeline_id.empty() && !targets.source_hash.empty() && !targets.formula_id.empty() &&
          !targets.dispersion_units.empty() && !targets.downside_units.empty() && targets.interval_ms>0,
          "invalid risk artifact identity");
  std::set<std::int64_t> horizons;
  for (auto h:targets.horizons_ticks) require(h>0 && horizons.insert(h).second,"invalid risk horizon panel");
  require(!horizons.empty(),"empty risk horizon panel");
  std::set<std::tuple<std::string,std::int64_t,std::int64_t>> ids;
  for (const auto& row:targets.rows) {
    require(!row.pair_id.empty() && ids.emplace(row.pair_id,row.channel_id,row.horizon_ticks).second &&
      horizons.contains(row.horizon_ticks) && valid_role(row.role) && row.role!=Role::unassigned &&
      row.endpoint_ms>=0 && row.decision_ms>=0,"invalid/duplicate risk row association");
    require(row.expected_returns==static_cast<std::uint64_t>(row.horizon_ticks) &&
      row.eligible_returns<=row.expected_returns && row.support.size()==row.expected_returns+1 &&
      row.return_valid.size()==row.expected_returns,"risk count/support shape mismatch");
    std::uint64_t counted=0;
    bool all_available=true;
    std::int64_t latest_available=0;
    for (std::size_t i=0;i<row.support.size();++i) {
      const auto& s=row.support[i];
      require(s.endpoint_ms==shift_timestamp(row.endpoint_ms,static_cast<std::int64_t>(i),targets.interval_ms) &&
        (!s.positive_finite || s.observed) && (!s.observed || s.present) &&
        s.present==s.available_ms.has_value() && (!s.available_ms || *s.available_ms>=s.endpoint_ms),
        "invalid risk close support");
      if (!s.available_ms) all_available=false;
      else latest_available=std::max(latest_available,*s.available_ms);
      if (i) {
        require(row.return_valid[i-1]<=1 && static_cast<bool>(row.return_valid[i-1])==
          (s.positive_finite && row.support[i-1].positive_finite),"risk adjacent-return mask mismatch");
        counted+=row.return_valid[i-1];
      }
    }
    require(counted==row.eligible_returns && row.available_ms.has_value()==all_available &&
      (!all_available || row.available_ms.value()==latest_available),"risk availability/count mismatch");
    require(std::isfinite(row.dispersion_1m) && row.dispersion_1m>=0 && std::isfinite(row.downside) &&
      row.downside>=0 && std::isfinite(row.mean_return),"nonfinite risk label");
    require(row.eligible==(row.reasons==risk_no_issue && counted==row.expected_returns),"risk eligibility mismatch");
    if (row.eligible) require(row.available_ms && row.support[0].available_ms &&
      *row.support[0].available_ms<=row.decision_ms,"eligible risk origin was unavailable");
  }
}

inline void write_risk_targets(std::ostream& stream,const FutureRiskTargets& targets) {
  validate_risk_targets(targets);ledger_detail::write(stream,"CUWLM.R1",targets);
}
inline FutureRiskTargets read_risk_targets(std::istream& stream,LedgerReadLimits limits={}) {
  auto out=ledger_detail::read<FutureRiskTargets>(stream,"CUWLM.R1",limits);validate_risk_targets(out);return out;
}
inline void publish_timeline(const std::filesystem::path& path,const EmbeddingTimeline& timeline) {
  ledger_detail::publish(path,[&](auto& stream){write_timeline(stream,timeline);});
}
inline void publish_pairs(const std::filesystem::path& path,const DynamicsPairs& pairs) {
  ledger_detail::publish(path,[&](auto& stream){write_pairs(stream,pairs);});
}
inline void publish_risk_targets(const std::filesystem::path& path,const FutureRiskTargets& targets) {
  ledger_detail::publish(path,[&](auto& stream){write_risk_targets(stream,targets);});
}
inline EmbeddingTimeline load_timeline(const std::filesystem::path& path,LedgerReadLimits limits={}) {
  std::ifstream stream(path,std::ios::binary);if (!stream) throw std::runtime_error("cannot open timeline ledger");
  return read_timeline(stream,limits);
}
inline DynamicsPairs load_pairs(const std::filesystem::path& path,LedgerReadLimits limits={}) {
  std::ifstream stream(path,std::ios::binary);if (!stream) throw std::runtime_error("cannot open pair ledger");
  return read_pairs(stream,limits);
}
inline DynamicsPairs load_pairs(const std::filesystem::path& path,const EmbeddingTimeline& timeline,LedgerReadLimits limits={}) {
  auto out=load_pairs(path,limits);validate_pairs(timeline,out);return out;
}
inline FutureRiskTargets load_risk_targets(const std::filesystem::path& path,LedgerReadLimits limits={}) {
  std::ifstream stream(path,std::ios::binary);if (!stream) throw std::runtime_error("cannot open risk ledger");
  return read_risk_targets(stream,limits);
}

} // namespace cuwacunu::latent_market::v1
