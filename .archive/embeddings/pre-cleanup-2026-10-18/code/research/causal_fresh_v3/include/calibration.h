// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Fresh TRAIN-only sufficient statistics. This file executes no encoder.
#pragma once
#include "market_data.h"
#include <torch/cuda.h>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace causal_fresh_v3::calibration {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v3;
namespace impl=cp::implementation;
namespace md=causal_fresh_v3::market;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
struct Result {cp::FrozenScaler scaler;cp::BankStandardizer bank;double seconds{0};int64_t loader_calls{0},raw_cells{0};std::string source_identity;};
inline Result fit(md::Dataset &data,const cp::Config &config,const std::vector<int64_t> &rows,
    const fs::path &directory,const std::string &source_identity){
  torch::NoGradGuard guard;const auto started=Clock::now();const auto &c=config.level;
  md::check(rows.size()==4096 && !fs::exists(directory) && fs::create_directory(directory),"exclusive fresh TRAIN calibration");
  const auto before_calls=data.calls,before_cells=data.record_cells;
  auto counts=torch::zeros({3,9},torch::kInt64),sum=torch::zeros({3,9},torch::kFloat64),sumsq=torch::zeros_like(sum);
  const auto width=impl::consecutive_change_bank_width(c);
  auto bank_counts=torch::zeros({width},torch::kInt64),bank_sum=torch::zeros({width},torch::kFloat64),bank_sumsq=torch::zeros_like(bank_sum);
  for(size_t offset=0;offset<rows.size();offset+=8){
    const auto selected=md::slice(rows,offset,8);
    const auto raw=data.load(selected,md::train_begin,md::train_end);
    const auto window=cp::boundary::retain(raw,torch::tensor(selected,torch::kFloat64),config);
    const auto observed=window.raw.observed;
    const auto rounded=torch::where(observed,window.raw.data,torch::zeros_like(window.raw.data)).to(torch::kFloat32).to(torch::kFloat64);
    md::check(rounded.device().is_cpu()&&torch::isfinite(rounded).all().item<bool>(),"fresh parent rounding before scaler fitting");
    counts+=observed.sum(std::vector<int64_t>{0,2});sum+=rounded.sum(std::vector<int64_t>{0,2});sumsq+=rounded.square().sum(std::vector<int64_t>{0,2});
    const auto targets=impl::relative_change_targets(window.raw,c);
    const auto valid=targets.valid.cpu();const auto safe=torch::where(targets.valid,targets.values,torch::zeros_like(targets.values)).to(torch::kCPU,torch::kFloat64);
    bank_counts+=valid.sum(0);bank_sum+=safe.sum(0);bank_sumsq+=safe.square().sum(0);
    if(offset==0){
      torch::serialize::OutputArchive witness;
      witness.write("row_ids",torch::tensor(selected,torch::kInt64),true);witness.write("raw_F64",window.raw.data,true);witness.write("observed",observed,true);
      witness.write("timestamps",window.times,true);witness.write("parent_rounded_F32_then_F64",rounded,true);
      witness.write("relative_bank_F32",targets.values.cpu(),true);witness.write("relative_bank_valid",valid,true);
      witness.save_to((directory/"first-TRAIN-batch-witness.pt").string());
    }
  }
  md::check(counts.gt(0).all().item<bool>()&&torch::isfinite(sum).all().item<bool>()&&torch::isfinite(sumsq).all().item<bool>(),"finite fresh input sufficient statistics");
  const auto denominator=counts.to(torch::kFloat64),mean=sum/denominator;
  const auto variance=(sumsq/denominator-mean.square()).clamp_min(0),deviation=variance.sqrt();
  cp::FrozenScaler scaler{mean,deviation.clamp_min(c.scale_floor),counts,torch::tensor({0,1,2},torch::kInt64),c.scale_floor,deviation.lt(c.scale_floor)};
  scaler.validate(impl::configurable_spectral_input_config(c));
  cp::BankStandardizer bank(c,source_identity,scaler.identity());
  bank.counts=bank_counts;bank.sum=bank_sum;bank.sumsq=bank_sumsq;bank.freeze();bank.validate(c,nullptr,&bank.input_scaler_identity);
  const auto text=[](const std::string &value){return impl::configurable_checkpoint_detail::text_tensor(value);};
  torch::serialize::OutputArchive input_archive,input_payload;
  input_archive.write("artifact_kind",text("cpb_v3_fresh_TRAIN_input_scaler_v1"),true);input_archive.write("source_identity",text(source_identity),true);
  input_archive.write("fit_recipe",text("4096-saved-TRAIN-anchor-recent128;observed-raw-F64-to-F32-to-CPU-F64;window-weighted-count-sum-sumsq;population-variance-clamp0;scale-floor1e-6"),true);
  input_archive.write("scaler_identity",text(scaler.identity()),true);input_archive.write("fit_row_ids",torch::tensor(rows,torch::kInt64),true);
  input_archive.write("sum",sum,true);input_archive.write("sumsq",sumsq,true);scaler.save(input_payload);input_archive.write("scaler",input_payload);
  input_archive.save_to((directory/"input-scaler.pt").string());
  torch::serialize::OutputArchive bank_archive,bank_payload;
  bank_archive.write("artifact_kind",text("cpb_v3_fresh_TRAIN_relative_bank_standardizer_v1"),true);bank_archive.write("source_identity",text(source_identity),true);
  bank_archive.write("fit_row_ids",torch::tensor(rows,torch::kInt64),true);bank.save(bank_payload);bank_archive.write("bank",bank_payload);
  bank_archive.save_to((directory/"bank-standardizer.pt").string());
  torch::cuda::synchronize();Result out{scaler,bank,std::chrono::duration<double>(Clock::now()-started).count(),data.calls-before_calls,data.record_cells-before_cells,source_identity};
  std::ostringstream report;report<<std::setprecision(17)<<"{\"kind\":\"cpb_v3_fresh_TRAIN_calibration_v1\",\"status\":\"passed\",\"TRAIN_anchors\":4096,\"history\":128,\"input_scaler_fits\":1,\"bank_normalizer_fits\":1,\"validation_rows_used\":0,\"test_rows_used\":0,\"encoder_calls\":0,\"old_normalization_artifacts_opened\":0,\"input_scaler_identity\":\""<<scaler.identity()<<"\",\"bank_identity\":\""<<bank.identity()<<"\",\"loader_calls\":"<<out.loader_calls<<",\"raw_record_cells\":"<<out.raw_cells<<",\"seconds_including_loader_witness_IO\":"<<out.seconds<<",\"artifacts\":[";
  bool first=true;for(const auto &name:{"input-scaler.pt","bank-standardizer.pt","first-TRAIN-batch-witness.pt"}){const auto path=directory/name;if(!first)report<<',';first=false;report<<"{\"file\":\""<<name<<"\",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":\""<<cp::boundary::sha256_file(path)<<"\"}";}
  report<<"]}";std::ofstream stream(directory/"summary.json");md::check(bool(stream),"write fresh calibration receipt");stream<<report.str()<<'\n';stream.close();md::check(bool(stream),"save fresh calibration receipt");
  return out;
}
} // namespace causal_fresh_v3::calibration
