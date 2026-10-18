// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Bounded source-layout witness. Saved inference only; no fits or updates.
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/relative_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_checkpoint.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
namespace r = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
void check(bool ok, const char *why) { TORCH_CHECK(ok, "[RPB layout] ", why); }
struct Witness {
  std::ostream &out;
  template<class T> void number(T v) { out.write(reinterpret_cast<const char *>(&v), sizeof v); }
  void text(const std::string &s) { number(uint64_t(s.size())); out.write(s.data(), s.size()); }
  void tensor(const torch::Tensor &v) {
    number(uint8_t(v.defined())); if (!v.defined()) return;
    text(c10::toString(v.scalar_type())); text(v.device().str());
    number(uint8_t(v.requires_grad())); number(uint8_t(v.grad().defined()));
    number(uint64_t(v.dim())); for (auto d : v.sizes()) number(d);
    const auto cpu = v.detach().to(torch::kCPU).contiguous();
    number(uint64_t(cpu.nbytes()));
    if (cpu.nbytes()) out.write(static_cast<const char *>(cpu.const_data_ptr()), cpu.nbytes());
  }
  void rng() {
    text("CPU RNG"); tensor(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state());
    text("CUDA RNG"); tensor(at::globalContext().defaultGenerator(torch::Device(torch::kCUDA, 0)).get_state());
  }
  template<class Model> void state(Model model) {
    const auto modules = model->named_modules(); number(uint64_t(modules.size()));
    for (const auto &m : modules) { text(m.key()); number(uint8_t(m.value()->is_training())); }
    const auto parameters = model->named_parameters(), buffers = model->named_buffers();
    number(uint64_t(parameters.size())); for (const auto &v : parameters) { text(v.key()); tensor(v.value()); }
    number(uint64_t(buffers.size())); for (const auto &v : buffers) { text(v.key()); tensor(v.value()); }
  }
  void config(const r::Config &c) {
    tensor(torch::tensor(r::configurable_checkpoint_detail::integers(c), torch::kInt64));
    tensor(torch::tensor(r::configurable_checkpoint_detail::numbers(c), torch::kFloat64));
    tensor(torch::tensor(r::resolved_channel_ids(c), torch::kInt64));
    text(c.device.str()); text(c10::toString(c.dtype));
  }
  void scaler(const r::FrozenScaler &s) {
    text(s.identity()); number(s.scale_floor);
    for (const auto &t : {s.mean, s.scale, s.count, s.channel_ids, s.floor_applied}) tensor(t);
  }
  template<class Normalizer> void normalizer(const Normalizer &n) {
    text(n.identity()); text(n.source_identity); text(n.input_scaler_identity);
    for (const auto &t : {n.counts,n.sum,n.sumsq,n.mean,n.scale,n.support}) tensor(t);
  }
};
template<class Model> std::string state_bytes(Model model) {
  std::ostringstream s(std::ios::binary); Witness{s}.state(model); return s.str();
}
const r::Config &level_config(const r::Config &c) { return c; }
const r::Config &level_config(const r::SeparatedChangeConfig &c) { return c.level; }
r::Input artificial(const r::Config &c, const r::FrozenScaler &scaler, int view) {
  const auto n = 2*c.channel_count*c.history_length*c.input_width;
  const auto phase = torch::arange(n, torch::kFloat64).reshape({2,c.channel_count,c.history_length,c.input_width});
  auto data = scaler.mean.reshape({1,c.channel_count,1,c.input_width}) +
      scaler.scale.reshape({1,c.channel_count,1,c.input_width})*(phase*.031).sin();
  auto observed = torch::ones_like(data, torch::kBool);
  if (view == 1 || view == 2) {
    observed[0][0].narrow(0, 9, 5).fill_(false);
    observed[1][c.channel_count-1].narrow(0, c.history_length/2, 3).fill_(false);
    observed[0][0][4][0] = false;
  }
  if (view == 2) data = torch::where(observed, data, torch::full_like(data, std::numeric_limits<double>::quiet_NaN()));
  if (view == 3) observed.fill_(false);
  return {data,observed,torch::tensor(r::resolved_channel_ids(c),torch::kInt64),
      torch::full({2},double(c.history_length-1)*c.sampling_interval,torch::kFloat64),c.sampling_interval};
}
template<class Model> r::EncodeOutput encode(Model m, const r::Input &raw,
    const r::FrozenScaler &scaler, const r::Config &c) {
  return m->encode(r::transform_configurable_spectral_input(scaler,raw,c));
}
r::EncodeOutput encode(r::RelativeChangeModel m, const r::Input &raw,const r::FrozenScaler &s,const r::Config &) { return m->encode(raw,s); }
r::EncodeOutput encode(r::ParentModulatedDecoderModel m,const r::Input &raw,const r::FrozenScaler &s,const r::Config &) { return m->encode(raw,s); }
r::EncodeOutput encode(r::LocationScaleDecoderModel m,const r::Input &raw,const r::FrozenScaler &s,const r::Config &) { return m->encode(raw,s); }
template<class Model> void extra(Witness &, Model , const torch::Tensor &, const torch::Tensor &) {}
void extra(Witness &w, r::LocationScaleDecoderModel m,const torch::Tensor &z,const torch::Tensor &ids) {
  const auto trace=m->decode_with_trace(z,ids); w.text("V30 variance"); w.tensor(trace.variance);
}
template<class Loaded, class Fresh> void exercise(Witness &w,const std::string &label,Loaded &loaded,Fresh fresh) {
  w.text(label); const auto &c=level_config(loaded.config);
  w.config(c); w.number(loaded.training_updates); w.scaler(loaded.scaler); w.rng();
  torch::manual_seed(731199); auto initialized=fresh(); w.text("fresh constructor"); w.state(initialized); w.rng();
  const auto before=state_bytes(loaded.model); w.text("saved registered state"); w.text(before);
  std::vector<torch::Tensor> natives;
  for (int view=0;view<4;++view) {
    auto raw=artificial(c,loaded.scaler,view);
    const auto output=encode(loaded.model,raw,loaded.scaler,c);
    const auto z=output.z_contextual_global;
    check(z.defined() && z.is_cuda() && torch::isfinite(z).all().template item<bool>(),"finite actual CUDA native");
    check(!z.requires_grad(),"no inference graph");
    const auto decoded=loaded.model->decode(z,raw.channel_ids);
    check(torch::isfinite(decoded).all().template item<bool>(),"finite decoder");
    w.number(view); w.tensor(z); w.tensor(decoded);
    for (const auto &v : {output.sample_valid_mask,output.channel_valid_mask,output.visible_observation_counts,
        output.visible_patch_counts,output.channel_ids}) w.tensor(v);
    extra(w,loaded.model,z,raw.channel_ids); natives.push_back(z.detach().clone());
    if (view==3) check(z.eq(0).all().template item<bool>() && !output.sample_valid_mask.any().template item<bool>(),"allmissing exact invalid zero");
  }
  check(torch::equal(natives[1],natives[2]),"masked NaNs preserve native output exactly");
  check(state_bytes(loaded.model)==before,"registered state frozen across inference");
  w.rng(); std::cout << "PASS " << label << ": saved typed load, constructor/state/RNG, four inference views\n";
}
template<class Loader, class Action> void load_case(Witness &w,const std::string &label,
    const fs::path &root,const char *relative,Loader loader,Action action) {
  torch::manual_seed(731198); const auto path=root/relative;
  check(fs::is_regular_file(path),"saved checkpoint exists");
  w.text(relative); auto loaded=loader(path.string()); action(label,loaded);
}
} // namespace

int main(int argc,char **argv) try {
  check(argc==3,"usage: layout_compatibility INTEGRATION_ROOT NEW_WITNESS_PATH");
  check(torch::cuda::is_available(),"actual CUDA required");
  torch::set_num_threads(1); torch::NoGradGuard guard;
  const fs::path root=argv[1],destination=argv[2];
  check(!fs::exists(destination),"witness destination must be new");
  fs::create_directories(destination.parent_path());
  std::ofstream output(destination,std::ios::binary);check(bool(output),"open new witness"); Witness w{output};
  w.text("rpb-source-layout-compatibility-v1;zero-updates;zero-fits;no-heads;batch2;four-views");
  torch::manual_seed(731199);r::ConfigurableSpectralModel v20(r::default_configurable_spectral_config());
  v20->eval();w.text("RPB-v20 fresh constructor");w.config(v20->config());w.state(v20);w.rng();
#define CASE(LABEL,PATH,LOAD,MODEL) load_case(w,LABEL,root,PATH,[](const std::string &p){return r::LOAD(p);},[&](const std::string &name,auto &s){exercise(w,name,s,[&]{return r::MODEL(s.config);});})
  CASE("RPB-v22","output/runs/market-capacity-study/run-base-gryiAV/checkpoints/inference-2048.pt",load_content_ordered_relation_checkpoint,ContentOrderedRelationModel);
  CASE("RPB-v23","output/runs/market-consecutive-changes/run-hAxXvh/checkpoints/inference-2048.pt",load_consecutive_change_checkpoint,ConsecutiveChangeModel);
  CASE("RPB-v24 native40","output/runs/market-separate-changes/run-Lr58TR/checkpoints/inference-2048.pt",load_separated_change_checkpoint,SeparatedChangeModel);
  CASE("RPB-v24 native36","output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-2048.pt",load_separated_change_checkpoint,SeparatedChangeModel);
  CASE("RPB-v25","output/runs/market-change-energy/run-r297GQ/checkpoints/inference-2048.pt",load_energy_change_checkpoint,EnergyChangeModel);
#undef CASE
#define NORMALIZED(LABEL,PATH,LOAD,MODEL) load_case(w,LABEL,root,PATH,[](const std::string &p){return r::LOAD(p);},[&](const std::string &name,auto &s){w.normalizer(s.model->normalizer());exercise(w,name,s,[&]{return r::MODEL(s.config,s.model->normalizer());});})
  NORMALIZED("RPB-v26","output/runs/market-conditioned-changes/run-RjTTZZ/checkpoints/inference-2048.pt",load_conditioned_change_checkpoint,ConditionedChangeModel);
  load_case(w,"RPB-v27",root,"output/runs/market-parent-conditional-changes/run-1KSQSS/checkpoints/inference-2048.pt",[](const std::string &p){return r::load_parent_conditional_change_checkpoint(p);},[&](const std::string &name,auto &s){
    w.normalizer(s.model->normalizer());w.text(s.model->conditional_map().identity());
    exercise(w,name,s,[&]{return r::ParentConditionalChangeModel(s.config,s.model->normalizer(),s.model->conditional_map());});});
  NORMALIZED("RPB-v28","output/runs/market-relative-changes/run-vaPC4X/checkpoints/inference-2048.pt",load_relative_change_checkpoint,RelativeChangeModel);
  NORMALIZED("RPB-v29","output/runs/market-parent-modulated-decoder/run-p4XMUY/checkpoints/inference-2048.pt",load_parent_modulated_decoder_checkpoint,ParentModulatedDecoderModel);
  NORMALIZED("RPB-v30","output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt",load_location_scale_decoder_checkpoint,LocationScaleDecoderModel);
#undef NORMALIZED
  torch::cuda::synchronize();output.close();check(bool(output),"complete witness write");
  std::cout << "PASS 10 saved checkpoints; 11 constructors; 40 bounded inference views; zero updates/fits/head calls\n";
  return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
