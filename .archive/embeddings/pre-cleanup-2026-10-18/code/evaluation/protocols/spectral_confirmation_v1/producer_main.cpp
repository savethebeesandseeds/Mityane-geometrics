// SPDX-License-Identifier: MIT
#include "retained_input.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/spectral_temporal_relation.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/fixed_feature_readouts.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#ifndef SPECTRAL_CONFIRMATION_SOURCE_ID
#define SPECTRAL_CONFIRMATION_SOURCE_ID "unrecorded"
#endif
#ifndef SPECTRAL_CONFIRMATION_CARD_SHA256
#define SPECTRAL_CONFIRMATION_CARD_SHA256 "unrecorded"
#endif
namespace {
using namespace spectral_confirmation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
using Clock=std::chrono::steady_clock;
constexpr const char *protocol="spectral-confirmation-v1";
constexpr const char *sdk_sha="e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb";
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
torch::Tensor cpu(const torch::Tensor&t){return t.detach().to(torch::kCPU).contiguous().clone();}
void text(torch::serialize::OutputArchive&a,const std::string&k,const std::string&v){a.write(k,embedding::archive::text_tensor(v),true);}
std::string text(torch::serialize::InputArchive&a,const std::string&k){return embedding::archive::tensor_text(read(a,k));}
void integer(torch::serialize::OutputArchive&a,const std::string&k,int64_t n){a.write(k,torch::tensor(n,torch::kInt64),true);}
void save_archive(const fs::path&p,torch::serialize::OutputArchive&a){
  const int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);require(fd>=0,"new archive required: "+p.string());::close(fd);
  embedding::archive::save_archive(p.string(),a);
}
bool exact(const torch::Tensor&a,const torch::Tensor&b){const auto x=cpu(a),y=cpu(b);return x.scalar_type()==y.scalar_type()&&x.sizes()==y.sizes()&&(!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0);}
struct RuntimeIsolation{
  int threads{at::get_num_threads()};std::vector<at::Generator> generators;std::vector<torch::Tensor> states;
  RuntimeIsolation(){generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i)generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto&g:generators)states.push_back(g.get_state().clone());}
  ~RuntimeIsolation()noexcept{try{for(size_t i=0;i<states.size();++i)generators[i].set_state(states[i]);at::set_num_threads(threads);}catch(...){std::terminate();}}
};
rpb::Config config(){rpb::Config c;c.device=torch::Device(torch::kCUDA,0);c.channel_mixer_layers=1;c.channel_mixer_placement=1;c.global_bottleneck_mode=2;
  c.global_pool_input_source=0;c.temporal_difference_input=0;c.export_width=32;rpb::validate_config(c);return c;}
rpb::Input input(const embedding::Batch&batch){return {batch.data,batch.feature_mask,torch::tensor({0,1,2},torch::kInt64),torch::full({batch.data.size(0)},31.,torch::kFloat64),1.};}
struct Candidate{
  std::shared_ptr<torch::nn::Module> module;
  std::function<rpb::EncodeOutput(const rpb::Input&)> encode;
  std::function<torch::Tensor(const torch::Tensor&,const torch::Tensor&)> decode;
  std::string id,tag,architecture,layout,training_recipe;int64_t trainable_parameter_count{0},frozen_parameter_count{0};
};
void validate_trainability(Candidate&candidate){int64_t registered=0,trainable=0,frozen=0;bool found=false;
  for(const auto&p:candidate.module->named_parameters()){registered+=p.value().numel();const bool odd=p.key()=="grouped_odd_relation_projection.weight";
    require(p.value().requires_grad()!=odd,"only the odd projection is frozen");
    if(odd){require(p.value().sizes()==torch::IntArrayRef({3,4,36})&&!p.value().grad().defined(),"literal fixed-prior432 geometry and no gradient");found=true;frozen+=p.value().numel();}else trainable+=p.value().numel();}
  require(found&&registered==226877&&trainable==226445&&frozen==432,"registered/trainable/frozen candidate counts");
  candidate.trainable_parameter_count=trainable;candidate.frozen_parameter_count=frozen;
}
Candidate make_candidate(const std::string &id) {
  require(id=="v19","unchanged v19 candidate only"); const auto c=config(); Candidate out; out.id=id;
  auto m=rpb::SpectralTemporalRelationModel(c); out.module=m.ptr();
  out.encode=[m](const rpb::Input&i)mutable{return m->encode(i);};
  out.decode=[m](const torch::Tensor&z,const torch::Tensor&ids)mutable{return m->decode(z,ids);};
  out.tag="RPB-v19.alt-01"; out.architecture=rpb::kSpectralTemporalRelationArchitectureId;
  out.layout=rpb::kSpectralTemporalRelationLayout; out.training_recipe=rpb::kSpectralTemporalRelationTrainingRecipeId;
  int64_t count=0; for(const auto&p:out.module->parameters()) {
    require(p.is_cuda()&&p.scalar_type()==torch::kFloat32,"all parameters CUDA F32"); count+=p.numel();
  }
  require(count==226877,"declared candidate parameter count"); validate_trainability(out); return out;
}
struct Trace{int64_t attempted,completed,target_cells,dynamics_dimensions,requested,actual,restored;double loss,waveform,dynamics,gradient;};
struct Trainer{
  Candidate candidate;rpb::FrozenScaler scaler;rpb::Input raw;
  std::unique_ptr<torch::optim::AdamW> optimizer;std::vector<torch::Tensor> initial;std::vector<Trace> trace;
  DatasetTask task;uint64_t master;int64_t parameter_count{0};double training_seconds{0};std::array<int64_t,3> context_counts{0,0,0};
  std::string source_ids,card_sha256;
  Trainer(const std::string&id,const embedding::Batch&training,uint64_t seed,const std::vector<std::string>&ids,const std::string&card,DatasetTask task_value):task(task_value),master(seed),source_ids(strings(ids)),card_sha256(card){
    raw=input(training);scaler=rpb::fit_scaler(raw,config());
    torch::manual_seed(rpb::training_detail::mixed(seed^0x7270622d696e6974ULL));candidate=make_candidate(id);candidate.module->train();
    optimizer=std::make_unique<torch::optim::AdamW>(candidate.module->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
    for(const auto&p:candidate.module->parameters()){initial.push_back(cpu(p));parameter_count+=p.numel();}
  }
  void train_to(int64_t budget){require(budget>=int64_t(trace.size())&&budget<=512,"monotonic closed budget");candidate.module->train();torch::cuda::synchronize(0);const auto start=Clock::now();
    while(int64_t(trace.size())<budget){const auto attempt=int64_t(trace.size());const auto rows=rpb::training_detail::sampled_indices(raw.data.size(0),8,int64_t(master),attempt);
      const auto batch=scaler.transform(rpb::training_detail::selected(raw,rows),config());require(batch.data.is_cuda(),"CUDA TRAIN input");
      const auto mask=rpb::make_training_mask(batch.observed,config(),rpb::training_detail::counter_seed(int64_t(master),attempt,0x6d61736bULL));
      torch::manual_seed(rpb::training_detail::counter_seed(int64_t(master),attempt,0x746f726368ULL));require(mask.eligible_channels.any().item<bool>(),"no skipped original masking attempt");
      const auto context=rpb::context_deletion::make_plan(mask,batch.channel_ids,config(),int64_t(master),attempt,rpb::ContextDeletionRecipe::coordinate15_v1);
      for(size_t i=0;i<3;++i)context_counts[i]+=std::array<int64_t,3>{context.requested_count,context.actual_count,context.restored_count}[i];
      const rpb::Input visible{torch::where(context.visible,batch.data,torch::zeros_like(batch.data)),context.visible,batch.channel_ids,batch.endpoints,batch.sampling_interval};
      optimizer->zero_grad();const auto encoded=candidate.encode(visible);const auto z=rpb::compact_reconstruction_export(encoded,config());require(z.sizes()==torch::IntArrayRef({8,32})&&z.is_cuda(),"sole served native32 TRAIN export");
      const auto prediction=candidate.decode(z,batch.channel_ids);const auto wave=rpb::hierarchical_huber(prediction,batch.data.detach(),mask.target,mask.eligible_channels,1.);
      auto dynamic=torch::zeros({},wave.loss.options());int64_t dimensions=0;
      const auto loss=wave.loss+dynamic;require(loss.is_cuda()&&torch::isfinite(loss).item<bool>()&&wave.eligible_example_count>0,"finite eligible CUDA loss");loss.backward();
      for(const auto&p:candidate.module->parameters())require(!p.grad().defined()||(p.grad().is_cuda()&&torch::isfinite(p.grad()).all().item<bool>()),"finite CUDA gradients");
      const auto gradient=torch::nn::utils::clip_grad_norm_(candidate.module->parameters(),1.,2.,true);require(std::isfinite(gradient)&&gradient>0,"finite nonzero gradient");optimizer->step();
      trace.push_back({attempt+1,attempt+1,wave.target_cell_count,dimensions,context.requested_count,context.actual_count,context.restored_count,loss.item<double>(),wave.loss.item<double>(),dynamic.item<double>(),gradient});
    }
    torch::cuda::synchronize(0);training_seconds+=seconds(start);verify_frozen();
  }
  void verify_frozen(){validate_trainability(candidate);const auto parameters=candidate.module->named_parameters();for(size_t i=0;i<parameters.size();++i)if(!parameters[i].value().requires_grad())require(exact(parameters[i].value(),initial[i]),"frozen prior values unchanged");}
  bool changed()const{const auto p=candidate.module->parameters();for(size_t i=0;i<p.size();++i)if(!exact(p[i],initial[i]))return true;return false;}
  std::string progress()const{std::ostringstream s;s<<std::setprecision(17)<<"{\"attempted\":"<<trace.size()<<",\"completed\":"<<trace.size()<<",\"sampled_rows\":"<<8*trace.size()<<",\"parameter_count\":"<<parameter_count<<",\"cuda_parameter_count\":"<<parameter_count
      <<",\"trainable_parameter_count\":"<<candidate.trainable_parameter_count<<",\"frozen_parameter_count\":"<<candidate.frozen_parameter_count<<",\"training_seconds\":"<<training_seconds<<",\"training_device\":\"cuda:0\",\"preprocessing_id\":"<<quote(scaler.identity())<<",\"weights_changed\":"<<(changed()?"true":"false")
      <<",\"context_requested_deleted_coordinates\":"<<context_counts[0]<<",\"context_actual_deleted_coordinates\":"<<context_counts[1]<<",\"context_restored_coordinates\":"<<context_counts[2]
      <<",\"trace_columns\":[\"attempted\",\"completed\",\"target_cells\",\"total_loss\",\"waveform_loss\",\"dynamics_loss\",\"gradient_norm\",\"dynamics_valid_dimensions\",\"requested\",\"actual\",\"restored\"],\"losses\":[";
    for(size_t i=0;i<trace.size();++i){const auto&t=trace[i];if(i)s<<',';s<<'['<<t.attempted<<','<<t.completed<<','<<t.target_cells<<','<<t.loss<<','<<t.waveform<<','<<t.dynamics<<','<<t.gradient<<','<<t.dynamics_dimensions<<','<<t.requested<<','<<t.actual<<','<<t.restored<<']';}return s.str()+"]}";}
  void save(const fs::path&path){verify_frozen();torch::serialize::OutputArchive a; text(a,"artifact_kind","rpb_spectral_confirmation_checkpoint_v1");text(a,"candidate",candidate.id);text(a,"protocol",protocol);text(a,"source_fingerprint",SPECTRAL_CONFIRMATION_SOURCE_ID);text(a,"human_card_sha256",card_sha256);
    text(a,"task",task_name(task));text(a,"dataset_id",dataset_id(task));text(a,"data_recipe",data_recipe);text(a,"architecture_id",candidate.architecture);text(a,"native_layout",candidate.layout);text(a,"training_recipe_id",candidate.training_recipe);text(a,"source_ids_json",source_ids);text(a,"resolved_settings","native32/C3H32F3/P8/W64/layers3/heads4/early1/mode2/source0/difference0/dropout0;AdamW0.001/wd0.0001/clip1/B8/context0.15/mask0.25;originalcounterstreams");
    integer(a,"master",int64_t(master));integer(a,"attempted",int64_t(trace.size()));integer(a,"completed",int64_t(trace.size()));integer(a,"sampled_rows",int64_t(trace.size())*8);integer(a,"parameter_count",parameter_count);integer(a,"trainable_parameter_count",candidate.trainable_parameter_count);integer(a,"frozen_parameter_count",candidate.frozen_parameter_count);text(a,"frozen_parameter_name","grouped_odd_relation_projection.weight");
    torch::serialize::OutputArchive model,optim,scale;candidate.module->save(model);optimizer->save(optim);scaler.save(scale);a.write("model",model);a.write("optimizer",optim);a.write("scaler",scale);
    a.write("rng_cpu",at::globalContext().defaultGenerator(at::Device(at::kCPU)).get_state().clone(),true);a.write("rng_cuda_0",at::globalContext().defaultGenerator(at::Device(at::kCUDA,0)).get_state().clone(),true);
    text(a,"progress_json",progress());save_archive(path,a);
  }
};
ev::CurveSnapshot snapshot(const fs::path&path,const std::string&id,uint64_t master,int64_t updates,const std::string&card,DatasetTask task){
  RuntimeIsolation isolation;torch::serialize::InputArchive a;a.load_from(path.string(),config().device);
  require(id=="v19","closed snapshot candidate");
  const auto architecture=rpb::kSpectralTemporalRelationArchitectureId;
  const auto layout=rpb::kSpectralTemporalRelationLayout;
  const auto training_recipe=rpb::kSpectralTemporalRelationTrainingRecipeId;
  require(text(a,"artifact_kind")=="rpb_spectral_confirmation_checkpoint_v1"&&text(a,"candidate")==id&&text(a,"protocol")==protocol&&text(a,"source_fingerprint")==SPECTRAL_CONFIRMATION_SOURCE_ID&&text(a,"human_card_sha256")==card&&text(a,"architecture_id")==architecture&&text(a,"native_layout")==layout&&text(a,"training_recipe_id")==training_recipe&&text(a,"task")==task_name(task)&&text(a,"dataset_id")==dataset_id(task)&&text(a,"data_recipe")==data_recipe&&
    read(a,"parameter_count").item<int64_t>()==226877&&read(a,"trainable_parameter_count").item<int64_t>()==226445&&read(a,"frozen_parameter_count").item<int64_t>()==432&&text(a,"frozen_parameter_name")=="grouped_odd_relation_projection.weight"&&read(a,"master").item<int64_t>()==int64_t(master)&&read(a,"completed").item<int64_t>()==updates&&read(a,"attempted").item<int64_t>()==updates,"immutable snapshot identity/counters");
  auto model=std::make_shared<Candidate>(make_candidate(id));torch::serialize::InputArchive m,s;a.read("model",m);model->module->load(m);validate_trainability(*model);model->module->eval();a.read("scaler",s);auto scaler=std::make_shared<rpb::FrozenScaler>(rpb::FrozenScaler::load(s));scaler->validate(config());
  ev::CurveSnapshot out;const auto provenance=std::string(protocol)+"/"+id+"/"+task_name(task)+"/master-"+std::to_string(master)+"/point-"+std::to_string(updates)+";CUDA immutable native32;source="+SPECTRAL_CONFIRMATION_SOURCE_ID;
  out.features.provenance=provenance;out.features.surfaces["native_global"]={ev::SurfaceKind::global,"at least one original observed channel; exact CUDA served32",{}};
  out.features.extract=[model,scaler,provenance](const embedding::Batch&batch){torch::NoGradGuard guard;auto normalized=scaler->transform(input(batch),config());auto e=model->encode(normalized);auto z=rpb::compact_reconstruction_export(e,config());
    return ev::FeatureMap{{"native_global",{z.to(torch::kCPU,torch::kFloat64).contiguous(),e.sample_valid_mask.to(torch::kCPU).contiguous(),provenance}}};};
  out.reconstruct=[model,scaler](const embedding::Batch&batch,const torch::Tensor&hidden){torch::NoGradGuard guard;auto normalized=scaler->transform(input(batch),config());const auto masks=rpb::mask_from_hidden(normalized.observed,hidden.to(config().device),config());
    const rpb::Input visible{torch::where(masks.visible,normalized.data,torch::zeros_like(normalized.data)),masks.visible,normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    auto e=model->encode(visible);auto z=rpb::compact_reconstruction_export(e,config());auto p=model->decode(z,normalized.channel_ids);return ev::CurveReconstruction{cpu(p),cpu(normalized.data),cpu(masks.eligible_channels)};};return out;
}
ev::NativeCurveRun recipe(){ev::NativeCurveRun r;r.card.shape={3,32,3,torch::kFloat64,torch::kCPU};r.card.channel_ids={0,1,2};r.card.feature_units="unitless,unitless,unitless";r.card.sampling_interval=1.;r.card.train_pairs=128;r.card.validation_pairs=64;r.card.test_pairs=0;r.card.threads=1;r.development_only=true;r.stress_sweep=false;return r;}
std::string master_json(const std::vector<uint64_t>&values){std::ostringstream s;s<<'[';for(size_t i=0;i<values.size();++i){if(i)s<<',';s<<values[i];}return s.str()+']';}
int64_t fit_count(const std::string&report,const std::string&key){std::smatch m;require(std::regex_search(report,m,std::regex("\""+key+"\":([0-9]+)")),"fixed fit-count key");return std::stoll(m[1]);}
void engineering(const fs::path&root,bool &owned){require(torch::cuda::is_available(),"CUDA engineering mandatory");new_leaf(root);owned=true;
  const auto batch=loader_engineering(root);std::vector<std::string> ids(16,"artificial-engineering");
  for(const std::string id:{"v19"}){const auto dir=root/id;require(fs::create_directory(dir),"new fixture candidate");Trainer t(id,batch,910901,ids,std::string(64,'0'),DatasetTask::slow_lag_sign);t.train_to(2);require(t.changed(),"CUDA engineering changed weights");
    const auto before=t.candidate.encode(t.scaler.transform(input(batch),config()));t.save(dir/"checkpoint.pt");auto frozen=snapshot(dir/"checkpoint.pt",id,910901,2,std::string(64,'0'),DatasetTask::slow_lag_sign);
    const auto after=frozen.features.extract(batch).at("native_global");require(exact(rpb::compact_reconstruction_export(before,config()).to(torch::kFloat64),after.values),"snapshot exact saved native32");
    const auto q=torch::zeros({16,3,32,3},torch::kBool);q.narrow(2,0,8).fill_(true);const auto rec=frozen.reconstruct(batch,q);
    require(rec.prediction.sizes()==torch::IntArrayRef({16,3,32,3})&&torch::isfinite(rec.prediction).all().item<bool>()&&rec.eligible.all().item<bool>(),"original-Q CUDA decoder callback");
    write_new(dir/"progress.json",t.progress()+"\n");}
  std::cout<<"Spectral confirmation runner CUDA engineering passed: 6 serialized loader positives/7 negatives;v19 only;2 updates;immutable native32 save/reload;original-Q decoder\n";
}
} // namespace
int main(int argc,char **argv) {
  fs::path output;bool owned_output=false;
  try {
    if(argc==2 && std::string(argv[1])=="--source-id") {std::cout<<SPECTRAL_CONFIRMATION_SOURCE_ID<<'\n';return 0;}
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2)
      require(i+1<argc && std::string(argv[i]).rfind("--",0)==0 && args.emplace(argv[i],argv[i+1]).second,"unique key/value CLI");
    RuntimeIsolation isolation;torch::set_num_threads(1);
    if(args.count("--engineering")) {require(args.size()==1,"one engineering output");output=args.at("--engineering");engineering(output,owned_output);return 0;}
    const std::set<std::string> required{"--candidate","--masters","--input-root","--inputs-sha256","--sources-sha256","--output","--card","--card-sha256","--sdk-proof","--sdk-proof-sha256"};
    for(const auto &key:required)require(args.count(key),"required CLI: "+key);
    for(const auto &[key,_]:args)require(required.count(key)||key=="--admission-log"||key=="--admission-sha256","closed CLI");
    const auto id=args.at("--candidate");require(id=="v19","closed declared candidate");
    require(args.at("--masters")=="930905,930906,930907","all three prospective masters in fixed order");
    const std::vector<uint64_t> selected{930905,930906,930907};const fs::path parent=args.at("--input-root");output=args.at("--output");
    direct_directory(parent);const fs::path bound("/embedding/output/runs/rpb-spectral-confirmation");
    const auto relative=parent.lexically_relative(bound);
    require(!relative.empty()&&!relative.is_absolute()&&*relative.begin()!=".."&&parent!=bound,"fresh data in spectral confirmation protocol root");
    std::map<std::string,fs::path> roles;
    std::vector<fs::path> paths{args.at("--inputs-sha256"),args.at("--sources-sha256"),args.at("--card"),args.at("--sdk-proof")};
    for(const auto master:quality_masters)for(const auto task:{DatasetTask::slow_lag_sign,DatasetTask::component_balance})
      for(const std::string view:{"training","validation","validation-deleted"}) {
        const auto name=role_prefix(master,task)+"controlled-"+view+".pt";roles.emplace(name,parent/name);paths.push_back(parent/name);
      }
    require(roles.size()==18,"whole eighteen legal role matrix before bytes");
    if(args.count("--admission-log")) {require(args.count("--admission-sha256"),"admission SHA accompanies log");paths.push_back(args.at("--admission-log"));}
    else require(!args.count("--admission-sha256"),"no orphan admission SHA");
    admit_files(paths);
    require(is_sha(SPECTRAL_CONFIRMATION_SOURCE_ID)&&sha256(bytes(args.at("--sources-sha256")))==SPECTRAL_CONFIRMATION_SOURCE_ID,"compiled source inventory");
    require(is_sha(SPECTRAL_CONFIRMATION_CARD_SHA256)&&args.at("--card-sha256")==SPECTRAL_CONFIRMATION_CARD_SHA256&&
            sha256(bytes(args.at("--card")))==args.at("--card-sha256"),"exact compiled prospective card SHA");
    require(args.at("--sdk-proof-sha256")==sdk_sha&&sha256(bytes(args.at("--sdk-proof")))==sdk_sha,"existing internal SDK proof");
    if(args.count("--admission-log"))require(sha256(bytes(args.at("--admission-log")))==args.at("--admission-sha256"),"declared engineering log preserved");
    Guard inputs(args.at("--inputs-sha256"),roles);const auto loaded=load_all(inputs);
    // Every task/master/split/deletion record is checked before CUDA construction.
    require(torch::cuda::is_available(),"CUDA quality required");new_leaf(output);owned_output=true;
    const auto inputs_sha=sha256(bytes(args.at("--inputs-sha256")));write_new(output/"input-manifest.json",inputs.json()+"\n");
    std::ostringstream report;report<<std::setprecision(17)
      <<"{\"protocol\":"<<quote(protocol)<<",\"candidate\":"<<quote(id)
      <<",\"registered_parameter_count\":226877,\"trainable_parameter_count\":226445,\"frozen_parameter_count\":432,"
      <<"\"frozen_parameter_name\":\"grouped_odd_relation_projection.weight\",\"data_recipe\":"<<quote(data_recipe)
      <<",\"tasks\":[\"slow_lag_sign\",\"component_balance\"],\"datasets\":{\"slow_lag_sign\":\"TEMPO-4\",\"component_balance\":\"AMP-2\"},"
      <<"\"designed_complexity_level_each\":5,\"complexity_scale_max\":5,\"source_fingerprint\":"<<quote(SPECTRAL_CONFIRMATION_SOURCE_ID)
      <<",\"human_card_sha256\":"<<quote(args.at("--card-sha256"))<<",\"sdk_proof_sha256\":"<<quote(sdk_sha)
      <<",\"inputs_sha256\":"<<quote(inputs_sha)<<",\"masters\":"<<master_json(selected)<<",\"input_manifest\":"<<inputs.json()<<",\"cohorts\":[";
    int64_t pipelines=0,outer=0;bool first=true;
    for(const auto master:selected)for(const auto task:{DatasetTask::slow_lag_sign,DatasetTask::component_balance}) {
      const auto &data=loaded.at({master,task});const auto &train=data.training;const auto &val=data.validation;const auto &deleted=data.deleted;
      const auto root=output/("seed-"+std::to_string(master)+"-"+task_name(task));require(fs::create_directory(root),"new task cohort");
      auto start=Clock::now();Trainer trainer(id,train.observed,master,train.source_ids,args.at("--card-sha256"),task);
      torch::serialize::OutputArchive scale;trainer.scaler.save(scale);save_archive(root/"scaler.pt",scale);
      double binding_seconds=seconds(start),feature_seconds=0,head_seconds=0,query_seconds=0;
      std::ostringstream points;std::string tq,vq;
      for(const int64_t point:{0,512}) {
        if(point) {trainer.train_to(point);require(trainer.changed()&&trainer.trace.size()==512,"all 512 CUDA updates and changed weights");}
        const auto dir=root/("point-"+std::to_string(point));require(fs::create_directory(dir),"new point");
        start=Clock::now();trainer.save(dir/"checkpoint.pt");auto frozen=snapshot(dir/"checkpoint.pt",id,master,point,args.at("--card-sha256"),task);binding_seconds+=seconds(start);
        start=Clock::now();
        ev::FixedFeatureMethod method{"native_"+id,ev::extract_native_global(frozen,train.observed,recipe()),
            ev::extract_native_global(frozen,val.observed,recipe()),ev::extract_native_global(frozen,deleted.observed,recipe()),false,{}};
        feature_seconds+=seconds(start);ev::FixedFeatureReadoutRun heads;
        heads.output_directory=(dir/"readouts").string();heads.master_seed=master;heads.training_labels=train.labels;
        heads.validation_labels=val.labels;heads.training_source_ids=train.source_ids;heads.validation_source_ids=val.source_ids;
        heads.methods.push_back(std::move(method));start=Clock::now();const auto scores=ev::run_fixed_feature_readouts(heads);head_seconds+=seconds(start);
        const auto fits=fit_count(scores,"ridge_fits");require(fits==fit_count(scores,"tiny_fits")&&fits<=3,"fixed three paired repetitions");
        pipelines+=fits;outer+=fit_count(scores,"outer_train_normalizer_fits");
        if(point)points<<',';
        points<<"{\"updates\":"<<point<<",\"checkpoint\":"<<quote((dir/"checkpoint.pt").string())<<",\"readouts\":"<<scores<<'}';
        if(point) {
          start=Clock::now();tq=ev::write_native_patch_reconstruction((dir/"training-reconstruction.pt").string(),train,frozen,recipe());
          vq=ev::write_native_patch_reconstruction((dir/"validation-reconstruction.pt").string(),val,frozen,recipe());query_seconds+=seconds(start);
        }
      }
      inputs.verify();write_new(root/"progress.json",trainer.progress()+"\n");if(!first)report<<',';first=false;
      report<<"{\"master\":"<<master<<",\"task\":"<<quote(task_name(task))<<",\"dataset_id\":"<<quote(dataset_id(task))
        <<",\"designed_complexity_level\":5,\"complexity_scale_max\":5,\"model_tag\":"<<quote(trainer.candidate.tag)
        <<",\"points\":["<<points.str()<<"],\"progress\":"<<trainer.progress()<<",\"training_reconstruction\":"<<tq
        <<",\"validation_reconstruction\":"<<vq<<",\"costs\":{\"binding_checkpoint_snapshot_seconds\":"<<binding_seconds
        <<",\"CUDA_feature_transfer_seconds\":"<<feature_seconds<<",\"CPU_heads_and_intervals_IO_seconds\":"<<head_seconds
        <<",\"CUDA_query_transfer_IO_seconds\":"<<query_seconds<<"}}";
      std::cout<<trainer.candidate.tag<<" task="<<task_name(task)<<" master="<<master<<" CUDA_updates=512 fixed_head_pipelines="<<pipelines<<'\n'<<std::flush;
    }
    inputs.verify();require(sha256(bytes(args.at("--sources-sha256")))==SPECTRAL_CONFIRMATION_SOURCE_ID&&
      sha256(bytes(args.at("--card")))==args.at("--card-sha256")&&sha256(bytes(args.at("--sdk-proof")))==sdk_sha&&
      sha256(bytes(args.at("--inputs-sha256")))==inputs_sha,"metadata inputs unchanged");
    report<<"],\"counts\":{\"encoder_trajectories\":6,\"encoder_updates_each\":512,\"sampled_rows\":24576,\"retained_points\":12,"
      <<"\"head_pipelines\":"<<pipelines<<",\"individual_heads\":"<<2*pipelines<<",\"helper_outer_train_fits\":"<<outer
      <<",\"native_exports\":36,\"query_writer_calls\":12,\"necessary_query_forwards\":48,\"generator_calls\":0,\"information_fit_calls\":0,"
      <<"\"PCA_fits\":0,\"old_encoder_or_head_refits\":0,\"CPU_encoder_calls\":0,\"skipped_attempts\":0},"
      <<"\"tasks_averaged\":false,\"testing_accessed\":false,\"stress_accessed\":false,\"promotion\":false}";
    write_new(output/"report.json",report.str()+"\n");
    write_new(output/"complete.json","{\"status\":\"complete\",\"protocol\":"+quote(protocol)+",\"candidate\":"+quote(id)+
      ",\"masters\":[930905,930906,930907],\"tasks\":[\"slow_lag_sign\",\"component_balance\"],\"updates_each\":512,"
      "\"source_fingerprint\":"+quote(SPECTRAL_CONFIRMATION_SOURCE_ID)+",\"report_sha256\":"+quote(sha256(report.str()+"\n"))+"}\n");return 0;
  } catch(const std::exception &e) {
    if(owned_output&&fs::is_directory(output)&&!fs::exists(output/"failure.json")) {
      try{write_new(output/"failure.json","{\"status\":\"failed\",\"error\":"+quote(e.what())+"}\n");}catch(...){}
    }
    std::cerr<<e.what()<<'\n';return 1;
  }
}
