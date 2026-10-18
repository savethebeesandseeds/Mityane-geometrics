// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Artificial compatibility/resume witnesses only; no quality training.
#include "cuwacunu/networks/predictive_training/v1/artifact.h"
#include <cstring>
#include <iostream>
#include <optional>

namespace pt=cuwacunu::networks::predictive_training::v1;
namespace pa=pt::artifact;
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
namespace {
void check(bool condition,const std::string& message) { TORCH_CHECK(condition,"[predictive artifact test] ",message); }
template<class F> void rejects(F&& action,const std::string& message) {
  bool rejected=false;try { action(); } catch(const std::exception&) { rejected=true; }
  check(rejected,message);
}
bool exact(const torch::Tensor& a,const torch::Tensor& b) {
  if(!a.defined()||!b.defined())return a.defined()==b.defined();
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type()==y.scalar_type()&&x.sizes()==y.sizes()&&
    (!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.nbytes())==0);
}
struct Temporary {
  fs::path path;
  Temporary() {
    auto pattern=(fs::temp_directory_path()/"predictive-artifact-v1-XXXXXX").string();
    std::vector<char> text(pattern.begin(),pattern.end());text.push_back('\0');
    const auto created=::mkdtemp(text.data());check(created,"private fixture directory");path=created;
  }
  ~Temporary() { std::error_code error;fs::remove_all(path,error); }
};
pt::Config config(pt::Arm arm=pt::Arm::full) {
  pt::Config config;config.state.embedding_width=3;config.state.state_width=3;config.state.dtype=torch::kFloat64;
  config.dynamics.state_width=3;config.dynamics.history_length=2;config.dynamics.horizons={1,9};config.dynamics.dtype=torch::kFloat64;
  config.arm=arm;config.lambda_variance=.3;config.lambda_covariance=.2;config.variance_floor=1.4;
  config.ema_decay=.8;config.learning_rate=.003;config.clip_norm=.7;config.weight_decay=.01;config.sampler_seed=903;
  return config;
}
pt::Identity identity() {
  return {av::sha256_bytes("artificial Encoder archive"),"artificial Encoder scaler","TRAIN embedding transform",
    "TRAIN reference scale","source receipt","TRAIN split","pair ledger","sampler recipe","loss/source/binary recipe",
    "State Representation Network history to Target State Representation Network forecast; native Encoder unchanged"};
}
pa::Context context(const fs::path& root) {
  pa::Context context;
  context.encoder=av::make_component({"Encoder","v3",av::sha256_bytes("artificial Encoder source")},
    av::sha256_bytes("artificial Encoder state"),av::sha256_bytes("artificial Encoder mapping"));
  context.encoder_archive_sha256=identity().encoder_checkpoint;context.repository_root=root;
  context.cadence_ms=60000;context.history_spacing_ticks=2;context.channel_ids={0,1,2};
  context.feature_names={"close","volume"};context.embedding_export_layout="native artificial 3";
  context.availability_policy="completed-candle cutoff";return context;
}
torch::Tensor values(int64_t n,const torch::TensorOptions& options,double phase=0.) {
  const auto x=torch::arange(n,options);return (x*.37+phase).sin()+.31*(x*.71-phase).cos()+x*.013;
}
pt::PredictiveBatch batch(const pt::Config& config) {
  const auto options=torch::TensorOptions().device(config.state.device).dtype(config.state.dtype);
  return {values(24,options).reshape({4,2,3}),values(24,options,.8).reshape({4,2,3}),
    torch::ones({4,2},options.dtype(torch::kBool)),{values(21,options,1.7).reshape({7,3}),torch::arange(7,torch::kInt64)}};
}
std::string pin(const fs::path& archive) { return av::sha256_file(av::sidecar_path(archive)); }
void same_trainers(const pt::Trainer& a,const pt::Trainer& b) {
  check(av::module_state_sha256(*a.online)==av::module_state_sha256(*b.online)&&
    av::module_state_sha256(*a.target)==av::module_state_sha256(*b.target)&&
    av::module_state_sha256(*a.dynamics)==av::module_state_sha256(*b.dynamics),"all three exact next-update component tensor digests");
  check(exact(a.reference_scale,b.reference_scale),"exact fixed reference scale");
  const auto x=a.trainable_parameters(),y=b.trainable_parameters();
  check(a.optimizer->state().size()==b.optimizer->state().size(),"same populated optimizer slot count");
  for(size_t j=0;j<x.size();++j) {
    const auto ai=a.optimizer->state().find(x[j].unsafeGetTensorImpl()),bi=b.optimizer->state().find(y[j].unsafeGetTensorImpl());
    check((ai==a.optimizer->state().end())==(bi==b.optimizer->state().end()),"same inactive optimizer slots");
    if(ai==a.optimizer->state().end())continue;
    const auto* as=dynamic_cast<const torch::optim::AdamWParamState*>(ai->second.get());
    const auto* bs=dynamic_cast<const torch::optim::AdamWParamState*>(bi->second.get());
    check(as&&bs&&as->step()==bs->step()&&exact(as->exp_avg(),bs->exp_avg())&&exact(as->exp_avg_sq(),bs->exp_avg_sq()),
      "exact AdamW steps and moments after checked resume");
    check(exact(x[j].grad(),y[j].grad()),"exact next-update gradient bytes");
  }
  const auto& ap=a.progress;const auto& bp=b.progress;
  check(ap.attempted_batches==bp.attempted_batches&&ap.completed_updates==bp.completed_updates&&
    ap.online_updates==bp.online_updates&&ap.dynamics_updates==bp.dynamics_updates&&ap.ema_updates==bp.ema_updates&&
    ap.sampler_draws==bp.sampler_draws,"all attempted/component/EMA/sampler counters match");
}
void exact_resume(const fs::path& directory,const pa::Context& context) {
  for(const auto arm:{pt::Arm::frozen_state,pt::Arm::blocked_prediction,pt::Arm::full}) {
    const auto config=::config(arm);torch::manual_seed(382);pt::Trainer trainer(config,identity());const auto input=batch(config);
    trainer.step(input);trainer.sample_indices(23,5);
    auto skipped=input;skipped.prediction_valid=torch::zeros_like(input.prediction_valid);trainer.step(skipped);
    const auto file=directory/("resume-"+std::to_string(static_cast<int>(arm))+".pt");
    const auto cpu_before=at::detail::getDefaultCPUGenerator().get_state();
    const auto cuda_before=at::cuda::detail::getDefaultCUDAGenerator(0).get_state();
    const auto manifest=pa::save_training(trainer,file,context);
    check(exact(cpu_before,at::detail::getDefaultCPUGenerator().get_state())&&
      exact(cuda_before,at::cuda::detail::getDefaultCUDAGenerator(0).get_state()),"sidecar emission preserves live CPU/CUDA RNG");
    check(!manifest.resume_identity_sha256.empty()&&manifest.components.size()==3,"one typed training sidecar covers the atomic tuple");
    const auto bytes_before=av::sha256_file(file);
    const auto expected_cpu=torch::rand({5},torch::kFloat64);
    const auto expected_cuda=torch::rand({5},torch::TensorOptions().device(config.state.device).dtype(torch::kFloat64));
    const auto expected_rows=trainer.sample_indices(23,9);const auto first=trainer.step(input);
    auto restored=pa::load_training(file,pin(file),config,identity(),context);
    check(exact(expected_cpu,torch::rand({5},torch::kFloat64))&&exact(expected_cuda,
      torch::rand({5},torch::TensorOptions().device(config.state.device).dtype(torch::kFloat64))),"checked resume restores CPU/CUDA RNG exactly");
    check(expected_rows==restored->sample_indices(23,9),"checked resume restores sampler engine and draws exactly");
    const auto second=restored->step(input);
    check(first.total_loss==second.total_loss&&first.prediction_loss==second.prediction_loss&&
      first.variance_loss==second.variance_loss&&first.covariance_loss==second.covariance_loss,"checked resume next-update losses exact in every arm");
    same_trainers(trainer,*restored);check(av::sha256_file(file)==bytes_before,"checked resume preserves archive bytes");
    auto wrong=config;wrong.ema_decay=.7;rejects([&]{pa::load_training(file,pin(file),wrong,identity(),context);},"EMA recipe drift rejected");
    wrong=config;wrong.learning_rate*=2;rejects([&]{pa::load_training(file,pin(file),wrong,identity(),context);},"optimizer recipe drift rejected");
    auto changed_identity=identity();changed_identity.loss_recipe+=" changed binary";
    rejects([&]{pa::load_training(file,pin(file),config,changed_identity,context);},"binary/source/training identity drift rejected");
  }
}
void inference_and_failures(const fs::path& directory,pa::Context context) {
  const auto config=::config();const auto id=identity();torch::manual_seed(1209);pt::Trainer trainer(config,id);
  trainer.step(batch(config));const auto file=directory/"inference.pt";
  const auto manifest=pa::save_inference(trainer,file,context);const auto sidecar_pin=pin(file);
  const auto input=batch(config).history_embeddings;
  const auto cpu_before=at::detail::getDefaultCPUGenerator().get_state(),
    cuda_before=at::cuda::detail::getDefaultCUDAGenerator(0).get_state();
  const auto serving=pa::load_inference(file,sidecar_pin,config,id,context);
  check(exact(cpu_before,at::detail::getDefaultCPUGenerator().get_state())&&
    exact(cuda_before,at::cuda::detail::getDefaultCUDAGenerator(0).get_state()),"inference inspection preserves Torch RNG");
  torch::Tensor expected;{torch::NoGradGuard guard;expected=trainer.dynamics->forward(trainer.online->forward(input));}
  check(exact(serving.forecast(input),expected),"checked forecast matches original numerical modules exactly");
  check(torch::allclose(serving.fold_forecast().forward(input),expected.flatten(1),1e-11,1e-11),"checked forecast folding preserves arithmetic");
  check(serving.state_history(input.select(1,0)).sizes()==torch::IntArrayRef({4,3}),"const State Representation Network accepts 2D exports");
  const auto original=serving.forecast(input);trainer.step(batch(config));
  check(exact(serving.forecast(input),original),"admitted frozen inference snapshot survives later Trainer updates");
  auto folded=serving.fold_forecast();folded.weight.zero_();
  auto reference=serving.reference_scale();reference.zero_();
  check(exact(serving.forecast(input),original)&&serving.reference_scale().gt(0).all().item<bool>(),"returned folding/reference copies cannot mutate admitted snapshot");
  auto changed=config;changed.learning_rate=.5;changed.lambda_variance=4;changed.lambda_covariance=3;
  changed.ema_decay=.2;changed.sampler_seed=173;changed.arm=pt::Arm::frozen_state;changed.state.identity_initialization=false;
  auto new_identity=id;new_identity.source="another data source receipt";new_identity.split="another split";
  new_identity.pair_ledger="another pair ledger";new_identity.sampler_recipe="another sampler";
  new_identity.loss_recipe="another training binary";new_identity.reference_scale="another training normalization receipt";
  const auto independent=pa::load_inference(file,sidecar_pin,changed,new_identity,context);
  check(exact(independent.forecast(input),original),"inference excludes optimizer/loss/EMA/seed/arm/data/reference training-only identity");
  auto repackaged=context;repackaged.encoder_archive_sha256=av::sha256_bytes("another byte packaging of identical Encoder mapping");
  auto repackaged_identity=id;repackaged_identity.encoder_checkpoint=repackaged.encoder_archive_sha256;
  const auto same_parent=pa::load_inference(file,sidecar_pin,config,repackaged_identity,repackaged);
  check(exact(same_parent.forecast(input),original),"identical Encoder ModelRef remains compatible across explicitly selected archive packaging");
  rejects([&]{pa::load_training(file,sidecar_pin,config,id,context);},"inference archive cannot exact-resume");
  rejects([&]{pa::load_inference(file,av::sha256_bytes("wrong pin"),config,id,context);},"wrong pinned sidecar rejected");
  changed=config;changed.dynamics.horizons={1,10};
  rejects([&]{pa::load_inference(file,sidecar_pin,changed,id,context);},"same-width horizon association drift rejected");
  auto semantic=context;std::reverse(semantic.channel_ids.begin(),semantic.channel_ids.end());
  rejects([&]{pa::load_inference(file,sidecar_pin,config,id,semantic);},"semantic channel reorder rejected");
  semantic=context;semantic.history_spacing_ticks=3;
  rejects([&]{pa::load_inference(file,sidecar_pin,config,id,semantic);},"history spacing drift rejected");
  semantic=context;semantic.cadence_ms=30000;
  rejects([&]{pa::load_inference(file,sidecar_pin,config,id,semantic);},"cadence drift rejected");
  auto parent=context;parent.encoder=av::make_component(context.encoder.module_ref,av::sha256_bytes("other Encoder weights"),context.encoder.inference_config_sha256);
  rejects([&]{pa::load_inference(file,sidecar_pin,config,id,parent);},"changed exact Encoder ModelRef rejected");
  auto variant=[&](const std::string& label,av::Manifest altered) {
    const auto path=directory/(label+".pt");fs::copy_file(file,path);av::publish_sidecar(path,altered);return path;
  };
  auto altered=manifest;altered.components[2].parents[0].model_ref=altered.components[1].model_ref;
  altered.components[2].parents[1].model_ref=altered.components[0].model_ref;
  altered.components[2].model_ref=av::derive_model_ref(altered.components[2]);
  auto path=variant("swapped-parent-roles",altered);
  rejects([&]{pa::load_inference(path,pin(path),config,id,context);},"State Representation Network and Target State Representation Network cannot swap roles");
  altered=manifest;altered.components[0].state_sha256=av::sha256_bytes("false state digest");
  altered.components[0].model_ref=av::derive_model_ref(altered.components[0]);path=variant("false-state",altered);
  rejects([&]{pa::load_inference(path,pin(path),config,id,context);},"component state digest independently recomputed");
  altered=manifest;altered.components[2].module_ref.source_sha256=av::sha256_bytes("false module source");
  altered.components[2].model_ref=av::derive_model_ref(altered.components[2]);path=variant("false-module",altered);
  const auto rejection_cpu=at::detail::getDefaultCPUGenerator().get_state(),
    rejection_cuda=at::cuda::detail::getDefaultCUDAGenerator(0).get_state();
  rejects([&]{pa::load_inference(path,pin(path),config,id,context);},"actual supported source closure independently recomputed");
  check(exact(rejection_cpu,at::detail::getDefaultCPUGenerator().get_state())&&
    exact(rejection_cuda,at::cuda::detail::getDefaultCUDAGenerator(0).get_state()),"failed post-inspection ModuleRef admission preserves CPU/CUDA RNG");
  altered=manifest;altered.components[0].inference_config_sha256=av::sha256_bytes("false config");
  altered.components[0].model_ref=av::derive_model_ref(altered.components[0]);path=variant("false-config",altered);
  rejects([&]{pa::load_inference(path,pin(path),config,id,context);},"archive configuration independently recomputed");
  path=directory/"missing-sidecar.pt";fs::copy_file(file,path);
  rejects([&]{pa::load_inference(path,sidecar_pin,config,id,context);},"missing sidecar rejected");
  path=variant("corrupt-raw",manifest);{std::ofstream changed_bytes(path,std::ios::binary|std::ios::app);changed_bytes<<"changed";}
  std::optional<pa::CheckedInference> unavailable;
  rejects([&]{unavailable.emplace(pa::load_inference(path,pin(path),config,id,context));},"corrupt raw archive rejected");
  check(!unavailable.has_value(),"failure exposes no usable checked combination");
  rejects([&]{pa::save_inference(trainer,file,context);},"checked publication refuses replacing an immutable pair");
}
void adoption_and_source_changes(const fs::path& directory,pa::Context context) {
  const auto config=::config();const auto id=identity();torch::manual_seed(191);pt::Trainer trainer(config,id);
  // This fixture created the original archive itself and records its source
  // pins at that point. Real adoption requires historical receipt evidence.
  const auto historical_modules=pa::checked_detail::current_modules(context);
  const auto legacy=directory/"legacy.pt";trainer.save_inference(legacy.string());const auto raw_before=av::sha256_file(legacy);
  const auto adopted=pa::adopt(legacy,config,id,context,"inference",historical_modules);
  check(raw_before==av::sha256_file(legacy)&&adopted.archive_sha256==raw_before,"legacy adoption never rewrites archive bytes");
  const auto loaded=pa::load_inference(legacy,pin(legacy),config,id,context);check(loaded.components().size()==3,"additively adopted legacy inference admitted");
  rejects([&]{pa::adopt(legacy,config,id,context,"inference",historical_modules);},"legacy sidecar cannot be overwritten");
  const auto absent=directory/"unproven-parent.pt";trainer.save_inference(absent.string());auto wrong=context;wrong.encoder_archive_sha256=av::sha256_bytes("unproven Encoder");
  rejects([&]{pa::adopt(absent,config,id,wrong,"inference",historical_modules);},"adoption requires exact independently supplied Encoder archive identity");
  rejects([&]{pa::adopt(absent,config,id,context,"inference",{});},"adoption requires independent historical implementation pins");
  check(!fs::exists(av::sidecar_path(absent)),"failed legacy adoption publishes no sidecar");
  const auto wrong_kind=directory/"wrong-kind.pt";trainer.save_training(wrong_kind.string());
  rejects([&]{pa::adopt(wrong_kind,config,id,context,"inference",historical_modules);},"internal archive kind independently checked");
  // Private copies witness changes to the supported source bytes without ever
  // editing an evaluated implementation in the repository.
  const auto source_root=directory/"source-copy";
  const auto state_relative=fs::path("code/embeddings/state_representation/versions/v1/include/cuwacunu/networks/state_representation/v1/state.h");
  const auto dynamics_relative=fs::path("code/operations/forecasting/latent_dynamics/versions/v1/include/cuwacunu/networks/latent_dynamics/v1/dynamics.h");
  for(const auto& relative:{state_relative,dynamics_relative}) {
    fs::create_directories((source_root/relative).parent_path());fs::copy_file(context.repository_root/relative,source_root/relative);
  }
  context.repository_root=source_root;
  const auto first=directory/"before-source-change.pt";const auto initial=pa::save_inference(trainer,first,context);
  const auto legacy_before_change=directory/"legacy-before-source-change.pt";trainer.save_inference(legacy_before_change.string());
  {std::ofstream output(source_root/dynamics_relative,std::ios::app);output<<"\n// artificial source-change witness\n";}
  rejects([&]{pa::adopt(legacy_before_change,config,id,context,"inference",historical_modules);},
    "legacy archive cannot be relabeled as current after historical Dynamics Model source changes");
  check(!fs::exists(av::sidecar_path(legacy_before_change)),"unmatched historical source evidence publishes no sidecar");
  rejects([&]{pa::load_inference(first,pin(first),config,id,context);},"changed Dynamics Model source bytes reject old ModuleRef");
  const auto second=directory/"after-source-change.pt";const auto changed=pa::save_inference(trainer,second,context);
  check(initial.components[0].model_ref==changed.components[0].model_ref&&initial.components[1].model_ref==changed.components[1].model_ref&&
    initial.components[2].model_ref!=changed.components[2].model_ref,"Dynamics Model source change preserves both representation identities");
  {std::ofstream output(source_root/state_relative,std::ios::app);output<<"\n// artificial State Representation Network source-change witness\n";}
  rejects([&]{pa::load_inference(second,pin(second),config,id,context);},"changed State Representation Network source bytes reject old ModuleRef");
  // New weights under the unchanged implementation keep ModuleRef and change
  // only the affected snapshot plus its downstream identities.
  auto new_context=::context(fs::current_path());
  const auto before=directory/"before-weight-change.pt";const auto old=pa::save_inference(trainer,before,new_context);
  {torch::NoGradGuard guard;trainer.dynamics->affine->bias.add_(.01);}
  const auto after=directory/"after-weight-change.pt";const auto weights=pa::save_inference(trainer,after,new_context);
  check(old.components[0].model_ref==weights.components[0].model_ref&&old.components[1].model_ref==weights.components[1].model_ref&&
    old.components[2].module_ref==weights.components[2].module_ref&&old.components[2].model_ref!=weights.components[2].model_ref,
    "Dynamics Model weight-only snapshot preserves module and representation identities");
  {torch::NoGradGuard guard;trainer.online->affine->bias.add_(.01);}
  const auto upstream=directory/"after-state-weight-change.pt";const auto state_weights=pa::save_inference(trainer,upstream,new_context);
  check(weights.components[0].model_ref!=state_weights.components[0].model_ref&&
    weights.components[1].model_ref==state_weights.components[1].model_ref&&
    weights.components[2].state_sha256==state_weights.components[2].state_sha256&&
    weights.components[2].model_ref!=state_weights.components[2].model_ref,
    "State Representation Network weight change propagates Dynamics Model identity through exact input_history parent");
}
} // namespace
int main() try {
  check(torch::cuda::is_available(),"managed-container actual CUDA required");torch::set_num_threads(1);
  Temporary directory;const auto context=::context(fs::current_path());
  exact_resume(directory.path,context);inference_and_failures(directory.path,context);adoption_and_source_changes(directory.path,context);
  torch::cuda::synchronize();
  std::cout<<"PASS predictive artifact v1: immutable joint sidecars, source/state/config/role/Encoder parent checks, "
    "typed archives, training-independent frozen inference, additive legacy adoption and all-arm exact AdamW/RNG/sampler/EMA next-update resume; artificial engineering only\n";
  return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
