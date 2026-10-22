// SPDX-License-Identifier: MIT
// Artificial engineering checks only: no market observations or historical weights.
#include "embedding/encoders/mtf_jepa_mae_vicreg/family.h"
#include "cuwacunu/artifacts/v1/sidecar.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;
namespace fam = mtf::family;
namespace fs = std::filesystem;
namespace {
int64_t checks = 0, physical_updates = 0, refused_updates = 0;
std::string source_sha256;
fs::path progress_path;
void progress() {
  if(progress_path.empty())return;
  std::ofstream f(progress_path);f<<"{\"status\":\"running\",\"physical_updates\":"<<physical_updates
    <<",\"refused_update_attempts\":"<<refused_updates<<"}\n";
  TORCH_CHECK(bool(f),"progress receipt write failed");
}
std::string quoted(const std::string &s) {
  std::ostringstream out;out<<'"';for(unsigned char c:s) {
    if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";
    else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";else out<<c;
  }out<<'"';return out.str();
}
void check(bool ok, const std::string &message) {
  ++checks; TORCH_CHECK(ok, "[MTF family fixture] ", message);
}
template<class F> void refuses(F f, const std::string &message) {
  bool rejected = false; try { f(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && torch::equal(a, b);
}
bool teacher(const std::string &name) {
  return name.rfind("target_tokenizer.", 0) == 0 || name.rfind("target_encoder.", 0) == 0;
}
std::string online_name(std::string name) { return name.erase(0, std::string("target_").size()); }
embedding::Batch artificial(const mtf::Config &c) {
  auto x = torch::empty({5, c.channel_count, c.history_length, c.input_width}, torch::kFloat64);
  auto m = torch::ones(x.sizes(), torch::kBool);
  auto a = x.accessor<double, 4>(); auto support = m.accessor<bool, 4>();
  for (int64_t b=0;b<5;++b) for (int64_t ch=0;ch<c.channel_count;++ch)
    for (int64_t t=0;t<c.history_length;++t) for (int64_t f=0;f<c.input_width;++f) {
      a[b][ch][t][f] = .1*b+.17*ch+.08*f+.03*t+std::sin(.27*t+.31*f+.13*ch+.22*b);
      support[b][ch][t][f] = (t+f+ch+b)%29 != 0;
    }
  return {x,m};
}
fam::Recipe recipe(mtf::Settings settings, const embedding::Batch &raw, const std::string &sampling) {
  fam::Recipe r; r.settings = std::move(settings); r.settings.steps = 3;
  r.settings.threads = 1; r.settings.batch_size = 4;
  r.source_sha256 = source_sha256;
  r.dataset_sha256 = fam::dataset_identity(fam::prepare(raw, r.settings.model));
  r.input_source_identity = "artificial-analytic-B5-BCHF-v1";
  r.split_identity = "engineering-TRAIN-only-no-heldout";
  r.recipe_identity = "bounded-mtf-family-fixture-v1"; r.sampling = sampling;
  return r;
}
std::map<std::string,torch::Tensor> snapshot(mtf::Model &model) {
  std::map<std::string,torch::Tensor> out;
  for (const auto &p : model->named_parameters()) out.emplace(p.key(), p.value().detach().clone());
  return out;
}
void same_model(mtf::Model &a, mtf::Model &b) {
  const auto ap=a->named_parameters(), bp=b->named_parameters();
  check(ap.size()==bp.size(), "same parameter inventory");
  for(const auto &p:ap) check(bp.contains(p.key()) && exact(p.value(),bp[p.key()]), "exact parameter "+p.key());
  const auto ab=a->named_buffers(), bb=b->named_buffers();
  check(ab.size()==bb.size(), "same buffer inventory");
  for(const auto &p:ab) check(bb.contains(p.key()) && exact(p.value(),bb[p.key()]), "exact buffer "+p.key());
}
void step(fam::Session &s) {
  const auto before=snapshot(s.model());
  const auto result=s.update(); ++physical_updates;progress();
  check(result.completed_updates==s.completed_updates() && std::isfinite(result.loss), "finite counted update");
  bool changed=false, nonzero_gradient=false;
  std::map<std::string,bool> group_gradients{{"tokenizer.",false},{"encoder.",false},{"predictor.",false},
    {"mae_decoder.",false},{"vicreg_stability_head.",false}};
  const auto after=s.model()->named_parameters();
  const double tau=s.recipe().settings.model.target_ema_tau;
  for(const auto &p:after) {
    if(teacher(p.key())) {
      check(!p.value().requires_grad() && !p.value().grad().defined(), "teacher remains frozen and gradient-free");
      auto expected=before.at(p.key()).clone();expected.mul_(tau);expected.add_(after[online_name(p.key())].detach(),1.-tau);
      check(exact(p.value(),expected), "EMA uses exact post-optimizer online state arithmetic");
    } else {
      changed=changed||!exact(before.at(p.key()),p.value());
      if(p.value().grad().defined()) {
        const bool nonzero=p.value().grad().ne(0).any().item<bool>();nonzero_gradient=nonzero_gradient||nonzero;
        for(auto &[prefix,seen]:group_gradients)if(p.key().rfind(prefix,0)==0)seen=seen||nonzero;
      }
    }
  }
  check(changed && nonzero_gradient, "real nonzero gradients and parameter update");
  for(const auto &[prefix,seen]:group_gradients)check(seen,"enabled component receives nonzero gradients: "+prefix);
}
void same_output(const mtf::EncodeOutput &a, const mtf::EncodeOutput &b) {
  check(exact(a.embeddings,b.embeddings),"token export roundtrip");
  check(exact(a.pooled_embedding,b.pooled_embedding),"global export roundtrip");
  check(exact(a.pooled_by_channel,b.pooled_by_channel),"channel export roundtrip");
  check(exact(a.pooled_time,b.pooled_time)&&exact(a.pooled_frequency,b.pooled_frequency),"domain export roundtrip");
  check(exact(a.sample_valid_mask,b.sample_valid_mask)&&exact(a.channel_valid_mask,b.channel_valid_mask),"validity export roundtrip");
}
void preparation_checks(const fam::Recipe &r, const embedding::Batch &raw) {
  auto absent=raw; absent.feature_mask=torch::Tensor();
  refuses([&]{fam::prepare(absent,r.settings.model);},"explicit mask required");
  auto badmask=raw; badmask.feature_mask=raw.feature_mask.to(torch::kFloat32);
  refuses([&]{fam::prepare(badmask,r.settings.model);},"boolean support required");
  auto poison=raw; poison.data=raw.data.clone(); poison.feature_mask=torch::ones_like(raw.feature_mask);
  poison.data.flatten()[0]=std::numeric_limits<double>::quiet_NaN();
  refuses([&]{fam::prepare(poison,r.settings.model);},"observed NaN refused");
  poison.data.flatten()[0]=std::numeric_limits<double>::infinity();
  refuses([&]{fam::prepare(poison,r.settings.model);},"observed infinity refused");
  poison.data.flatten()[0]=std::numeric_limits<double>::max();
  refuses([&]{fam::prepare(poison,r.settings.model);},"observed F64 to F32 overflow refused");
  poison.feature_mask.flatten()[0]=false;
  check(fam::prepare(poison,r.settings.model).data.flatten()[0].item<float>()==0,"unobserved poison zeroed");
  auto empty=raw; empty.feature_mask=torch::zeros_like(raw.feature_mask);
  auto empty_recipe=r; empty_recipe.dataset_sha256=fam::dataset_identity(fam::prepare(empty,r.settings.model));
  refuses([&]{fam::Session::fresh(empty_recipe,empty);},"all-missing training rows refused");
}
bool temporal_checks(fam::Session &s, const embedding::Batch &raw) {
  torch::NoGradGuard guard; s.model()->eval();
  auto p=fam::prepare(raw,s.recipe().settings.model);
  auto suffix=p; suffix.data=p.data.clone();
  suffix.data.narrow(2,p.data.size(2)/2,p.data.size(2)-p.data.size(2)/2).add_(2.0);
  const auto ta=s.model()->tokenize(p.data,p.feature_mask), tb=s.model()->tokenize(suffix.data,suffix.feature_mask);
  const auto frequency=ta.metadata.domain_id.eq(1).view({1,-1,1});
  check(torch::where(frequency,ta.frequency_reconstruction_targets-tb.frequency_reconstruction_targets,
        torch::zeros_like(ta.frequency_reconstruction_targets)).abs().max().item<double>()>1e-6,
        "frequency descriptors change when their window suffix changes; no prefix claim");
  const auto &config=s.recipe().settings.model;
  const bool full_window=std::any_of(config.time_scales.begin(),config.time_scales.end(),
    [&](int64_t scale){return scale>=config.history_length;});
  if(full_window) {
    const auto full_frequency=ta.metadata.domain_id.eq(1).logical_and(ta.metadata.start_index.eq(0))
      .logical_and(ta.metadata.width.eq(config.history_length));
    check(full_frequency.any().item<bool>(),"configured full-history frequency token has start zero and width H");
    check(torch::where(full_frequency.view({1,-1,1}),ta.frequency_reconstruction_targets-tb.frequency_reconstruction_targets,
      torch::zeros_like(ta.frequency_reconstruction_targets)).abs().max().item<double>()>1e-6,
      "full-history frequency descriptors specifically change under latter-half-only perturbation");
  }
  if(p.data.size(1)>1) {
    auto other=p; other.data=p.data.clone(); other.data.select(1,1).add_(3.0);
    const auto a=s.model()->encode(p.data,p.feature_mask), b=s.model()->encode(other.data,other.feature_mask);
    check(!torch::allclose(a.pooled_by_channel.select(1,0),b.pooled_by_channel.select(1,0),1e-6,1e-7),
          "channel zero is contextual through other-channel mixing");
  }
  return full_window;
}
void no_target_check(mtf::Settings settings) {
  auto &c=settings.model; c.channel_count=1;c.history_length=1;c.input_width=2;
  c.time_scales={1};c.scale_strides={1};c.use_frequency_tokens=false;c.strict_jepa_support=true;
  c.mask_ratio_time=.5;c.min_context_ratio=1.;
  auto raw=artificial(c); auto r=recipe(settings,raw,fam::kReplacementSampling);
  auto s=fam::Session::fresh(r,raw); auto prepared=fam::prepare(raw,c);
  auto masks=s.model()->create_masks(s.model()->tokenize(prepared.data,prepared.feature_mask));
  check(!masks.target_mask.any().item<bool>(),"strict singleton legitimately has no target");
  refuses([&]{s.update();},"training session refuses a no-target JEPA/MAE step"); ++refused_updates;
  check(s.completed_updates()==0,"refused no-target step does not advance progress");
}
void forged_payload(const std::string &original,const std::string &destination,fam::Session &payload) {
  torch::serialize::InputArchive in;in.load_from(original,torch::kCPU);
  torch::serialize::OutputArchive out,model,optimizer;
  for(const auto &key:in.keys()) if(key!="model"&&key!="optimizer") {
    torch::Tensor value;in.read(key,value,true);out.write(key,value,true);
  }
  payload.model()->save(model);payload.optimizer().save(optimizer);
  out.write("model",model);out.write("optimizer",optimizer);
  embedding::archive::save_archive(destination,out);
}
void scenario(const fs::path &config_path,const fs::path &out,const std::string &sampling,std::ostream &report,bool comma) {
  auto settings=mtf::read_settings(config_path.string()); const auto raw=artificial(settings.model);
  auto r=recipe(settings,raw,sampling); preparation_checks(r,raw);
  const auto name=config_path.stem().string()+"-"+(sampling==fam::kReplacementSampling?"replacement":"full-partition");
  const auto checkpoint=out/(name+"-training.pt"), inference=out/(name+"-inference.pt");
  auto whole=fam::Session::fresh(r,raw); const auto initial=whole.state_sha256();
  const auto inventory=fam::parameter_inventory(whole.model());
  int64_t all=0,learned=0,frozen=0;std::set<std::string> names;
  for(const auto &record:inventory.records) {
    check(names.insert(record.name).second,"unique named parameter inventory");
    int64_t product=1;for(const auto d:record.shape){check(d>0,"positive parameter shape");product*=d;}
    check(product==record.elements,"parameter shape derives element count");all+=record.elements;
    (record.trainable?learned:frozen)+=record.elements;
  }
  check(all==inventory.total&&learned==inventory.trainable&&frozen==inventory.frozen&&all==learned+frozen,
    "dynamic total/trainable/frozen inventory sums and schemas agree");
  check(learned>0&&frozen>0,"trainable online and frozen teacher inventories present");
  static std::map<std::string,int64_t> configuration_totals;
  configuration_totals[config_path.stem().string()]=all;
  if(configuration_totals.size()>1)check(configuration_totals.at("small")!=configuration_totals.at("alternative"),
    "meaningful configurations have distinct parameter inventories");
  for(int i=0;i<3;++i) step(whole);
  auto split=fam::Session::fresh(r,raw); check(split.state_sha256()==initial,"seeded fresh construction exact");
  step(split); split.save_training(checkpoint.string());
  auto resumed=fam::Session::resume(checkpoint.string(),r,raw); same_model(split.model(),resumed.model());
  check(split.optimizer_sha256()==resumed.optimizer_sha256(),"optimizer save/reload exact");
  step(resumed);step(resumed);
  check(whole.completed_updates()==3&&resumed.completed_updates()==3,"fixed total budget reached");
  same_model(whole.model(),resumed.model());
  check(whole.state_sha256()==resumed.state_sha256(),"uninterrupted/resumed complete model state exact");
  check(whole.optimizer_sha256()==resumed.optimizer_sha256(),"uninterrupted/resumed AdamW state exact");
  refuses([&]{resumed.update();},"fixed total budget cannot be exceeded");++refused_updates;
  resumed.export_inference(inference.string());
  fam::InferenceExpectation expected{r.settings.model,r.source_sha256,resumed.state_sha256()};
  auto loaded=fam::load_inference(inference.string(),expected);
  const auto dimensions=loaded.encode(raw);
  check(dimensions.pooled_embedding.sizes()==torch::IntArrayRef({5,r.settings.model.latent_dim})&&
    dimensions.pooled_by_channel.sizes()==torch::IntArrayRef({5,r.settings.model.channel_count,r.settings.model.latent_dim})&&
    dimensions.embeddings.dim()==3&&dimensions.embeddings.size(0)==5&&dimensions.embeddings.size(2)==r.settings.model.latent_dim,
    "native token/global/channel output dimensions derive from configuration");
  whole.model()->eval(); {torch::NoGradGuard guard;auto p=fam::prepare(raw,r.settings.model);
    same_output(whole.model()->encode(p.data,p.feature_mask),loaded.encode(raw));}
  auto missing=raw;missing.feature_mask=torch::zeros_like(raw.feature_mask);
  const auto empty=loaded.encode(missing);
  check(!empty.sample_valid_mask.any().item<bool>()&&empty.pooled_embedding.eq(0).all().item<bool>(),"missing inference invalid with zero output");
  auto changed=r;changed.recipe_identity+="-changed";
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed recipe refused");
  changed=r;changed.source_sha256[0]=changed.source_sha256[0]=='b'?'a':'b';
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed source refused");
  changed=r;changed.settings.batch_size++;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed batch/config refused");
  changed=r;changed.split_identity+="-changed";
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed split refused");
  changed=r;changed.adam_beta1=.8;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed AdamW beta refused");
  changed=r;changed.adam_epsilon=1e-6;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed AdamW epsilon refused");
  changed=r;changed.adam_amsgrad=true;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed AdamW AMSGrad refused");
  changed=r;changed.settings.steps=4;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed total budget refused");
  changed=r;changed.sampling=r.sampling==fam::kReplacementSampling?fam::kFullPartitionSampling:fam::kReplacementSampling;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed sampling algorithm refused");
  changed=r;changed.input_source_identity+="-changed";
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed input source identity refused");
  changed=r;changed.settings.model.latent_dim+=changed.settings.model.num_heads;
  refuses([&]{fam::Session::resume(checkpoint.string(),changed,raw);},"changed architectural latent dimension refused");
  auto changed_data=raw;changed_data.data=raw.data.clone();changed_data.data.flatten()[1]+=1.;
  refuses([&]{fam::Session::resume(checkpoint.string(),r,changed_data);},"changed data refused");
  auto bad=expected;bad.preparation="fictitious-scaler";
  refuses([&]{fam::load_inference(inference.string(),bad);},"incompatible preparation refused");
  bad=expected;bad.state_sha256=std::string(64,'0');
  refuses([&]{fam::load_inference(inference.string(),bad);},"wrong expected state refused");
  refuses([&]{fam::Session::resume(inference.string(),r,raw);},"inference artifact cannot resume training");
  refuses([&]{fam::load_inference(checkpoint.string(),expected);},"training artifact cannot load as inference");
  torch::set_num_threads(2);
  refuses([&]{resumed.save_training((out/(name+"-thread-drift.pt")).string());},"live runtime thread drift refused");
  torch::set_num_threads(1);
  auto forged=fam::Session::resume(checkpoint.string(),r,raw);
  {torch::NoGradGuard guard;for(auto &p:forged.model()->parameters())if(p.requires_grad()){p.add_(.001);break;}}
  const auto forged_model=(out/(name+"-forged-model.pt")).string();forged_payload(checkpoint.string(),forged_model,forged);
  refuses([&]{fam::Session::resume(forged_model,r,raw);},"finite model payload mutation with stale hash refused");
  auto forged_moment=fam::Session::resume(checkpoint.string(),r,raw);bool moment_mutated=false;
  for(auto &entry:forged_moment.optimizer().state()) {
    auto *state=dynamic_cast<torch::optim::AdamWParamState*>(entry.second.get());
    if(state&&state->exp_avg().defined()){state->exp_avg().add_(.001);moment_mutated=true;break;}
  }
  check(moment_mutated,"actual AdamW moment available for payload mutation");
  const auto forged_optimizer=(out/(name+"-forged-optimizer.pt")).string();forged_payload(checkpoint.string(),forged_optimizer,forged_moment);
  refuses([&]{fam::Session::resume(forged_optimizer,r,raw);},"finite AdamW moment mutation with stale hash refused");
  const bool full_window_checked=temporal_checks(resumed,raw);
  if(comma)report<<',';
  report<<"{\"configuration\":\""<<config_path.stem().string()<<"\",\"sampling\":\""<<sampling
        <<"\",\"configuration_file_sha256\":\""<<cuwacunu::artifacts::v1::sha256_file(config_path.string())
        <<"\",\"recipe_sha256\":\""<<fam::recipe_sha256(r)<<"\",\"dataset_sha256\":\""<<r.dataset_sha256
        <<"\",\"preparation\":\""<<fam::kPreparation<<"\",\"seed\":"<<r.settings.seed<<",\"initial_state_sha256\":\""<<initial
        <<"\",\"model_state_sha256\":\""<<resumed.state_sha256()<<"\",\"optimizer_sha256\":\""<<resumed.optimizer_sha256()
        <<"\",\"parameter_inventory\":{\"total\":"<<inventory.total<<",\"trainable\":"<<inventory.trainable<<",\"frozen\":"<<inventory.frozen
        <<",\"named_tensors\":"<<inventory.records.size()<<"},\"native_output_dimensions\":{\"global\":[5,"<<r.settings.model.latent_dim
        <<"],\"channel\":[5,"<<r.settings.model.channel_count<<','<<r.settings.model.latent_dim<<"],\"tokens\":[5,"<<dimensions.embeddings.size(1)<<','<<r.settings.model.latent_dim
        <<"]},\"gradient_groups\":[\"tokenizer\",\"encoder\",\"predictor\",\"mae_decoder\",\"vicreg_stability_head\"]"
        <<",\"mask_ratios\":{\"time\":"<<r.settings.model.mask_ratio_time<<",\"frequency\":"<<r.settings.model.mask_ratio_frequency
        <<",\"channel\":"<<r.settings.model.mask_ratio_channel<<"},\"full_window_frequency_check\":"<<(full_window_checked?"true":"false")
        <<",\"physical_updates\":6,\"logical_updates\":3,\"exact_resume\":true,\"inference_roundtrip\":true}";
}
}
int main(int argc,char **argv) {
  try {
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2) {TORCH_CHECK(i+1<argc,"missing option value");
      const std::string key(argv[i]);TORCH_CHECK(key=="--config-directory"||key=="--output"||key=="--source-sha256","unknown fixture option");
      TORCH_CHECK(args.emplace(key,argv[i+1]).second,"duplicate fixture option");}
    TORCH_CHECK(args.count("--config-directory")&&args.count("--output"),
      "Usage: mtf_family_test --config-directory DIR --output NEW_DIR [--source-sha256 SHA256]");
    const auto binary_sha256=cuwacunu::artifacts::v1::sha256_file(fs::canonical("/proc/self/exe").string());
    source_sha256=args.count("--source-sha256")?args.at("--source-sha256"):binary_sha256;
    torch::set_num_threads(1); const fs::path configs(args.at("--config-directory")), out(args.at("--output"));
    TORCH_CHECK(!fs::exists(out),"fixture output must be new: ",out.string());fs::create_directories(out);
    progress_path=out/"progress.json";progress();
    std::ostringstream scenarios; bool comma=false;
    for(const auto &config:{"small","alternative"}) for(const auto &sampling:{fam::kReplacementSampling,fam::kFullPartitionSampling}) {
      scenario(configs/(std::string(config)+".conf"),out,sampling,scenarios,comma);comma=true;
    }
    no_target_check(mtf::read_settings((configs/"small.conf").string()));
    const auto malformed=out/"malformed.pt";{std::ofstream f(malformed);f<<"invalid archive";}
    auto settings=mtf::read_settings((configs/"small.conf").string());auto raw=artificial(settings.model);
    refuses([&]{fam::Session::resume(malformed.string(),recipe(settings,raw,fam::kReplacementSampling),raw);},"malformed state refused");
    check(physical_updates==24,"exact physical update accounting");
    std::ostringstream receipt;receipt<<"{\"schema\":\"mtf-family-engineering-proof-v1\",\"status\":\"passed\",\"format_version\":1,\"scope\":\"artificial engineering; no quality result\",\"passed\":true,\"checks\":"<<checks
      <<",\"runtime\":"<<quoted(fam::runtime_identity())<<",\"source_sha256\":"<<quoted(source_sha256)
      <<",\"source_binding_kind\":"<<quoted(args.count("--source-sha256")?"caller-supplied-source-closure":"executable-sha256-fallback")
      <<",\"binary_sha256\":"<<quoted(binary_sha256)<<",\"physical_updates\":"<<physical_updates<<",\"refused_update_attempts\":"<<refused_updates
      <<",\"assertion_coverage\":[\"real-gradients-and-updates\",\"all-five-enabled-online-gradient-groups\",\"dynamic-parameter-and-output-dimensions\",\"teacher-freeze-and-post-optimizer-EMA\",\"two-configurations-two-recipes\",\"nonzero-channel-mask-and-feature-deletion-RNG\",\"exact-model-and-AdamW-resume\",\"inference-roundtrip\",\"input-and-binding-refusals\",\"specific-full-window-frequency-suffix-response\",\"contextual-channel-semantics\",\"strict-no-target-refusal\"]"
      <<",\"scenarios\":["<<scenarios.str()<<"]}";
    std::ofstream f(out/"result.json");f<<receipt.str()<<'\n';f.close();TORCH_CHECK(bool(f),"receipt write failed");
    std::cout<<receipt.str()<<'\n';return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
