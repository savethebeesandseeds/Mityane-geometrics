// Copyright (c) 2026 Waajacu. See LICENSE for the permission notice.
// Declared TRAIN-only gradient inspection: no optimizer, fitting or model updates.
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h"
#include "embedding/data/observation_windows.h"
#include <torch/cuda.h>
#include <torch/version.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace old=embedding::encoders::causal_patch_bottleneck_mae::v16;
namespace tr=cp::training;
namespace impl=cp::implementation;
namespace ow=embedding::data::observation_windows;
namespace fs=std::filesystem;
namespace {
constexpr int64_t kBatches=8,kBatch=8,kRows=kBatches*kBatch,kMaskSeed=949103;
constexpr double kControlWeight=.1,kCandidateWeight=.03,kClipNorm=1.;
void check(bool condition,const std::string &message){TORCH_CHECK(condition,"[CPB V17 TRAIN diagnostic] ",message);}
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';for(const unsigned char c:value){
    if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;
  }out<<'"';return out.str();
}
bool sha(const std::string &value){return value.size()==64&&value.find_first_not_of("0123456789abcdef")==std::string::npos;}
std::string artifact(const fs::path &file){std::ostringstream out;out<<"{\"path\":"<<quote(file.generic_string())<<",\"bytes\":"
    <<fs::file_size(file)<<",\"sha256\":"<<quote(cp::boundary::sha256_file(file))<<'}';return out.str();}
void write_new(const fs::path &file,const std::string &value){
  check(!fs::exists(file),"existing output must be preserved");std::ofstream out(file,std::ios::binary);check(bool(out),"open output");
  out<<value<<'\n';out.close();check(bool(out),"write output");
}
struct Args {
  fs::path settings,dataset,checkpoint,output;std::string dataset_sha,checkpoint_sha;int64_t mask_seed{-1};
  Args(int argc,char **argv){
    const std::set<std::string> allowed{"--settings","--dataset","--dataset-sha256","--checkpoint","--checkpoint-sha256","--mask-seed","--output"};
    check(argc>1&&(argc-1)%2==0,"options need explicit values; use --help");std::map<std::string,std::string> values;
    for(int i=1;i<argc;i+=2)check(allowed.count(argv[i])&&values.emplace(argv[i],argv[i+1]).second,"unknown or duplicate option");
    check(values.size()==allowed.size(),"--settings --dataset --dataset-sha256 --checkpoint --checkpoint-sha256 --mask-seed --output required");
    settings=values.at("--settings");dataset=values.at("--dataset");checkpoint=values.at("--checkpoint");output=fs::absolute(values.at("--output")).lexically_normal();
    dataset_sha=values.at("--dataset-sha256");checkpoint_sha=values.at("--checkpoint-sha256");check(sha(dataset_sha)&&sha(checkpoint_sha),"independent lowercase SHA256 pins required");
    const auto text=values.at("--mask-seed");check(!text.empty()&&text.find_first_not_of("0123456789")==std::string::npos,"decimal mask seed required");
    mask_seed=std::stoll(text);check(mask_seed==kMaskSeed,"mask seed differs from prospective declaration");
    for(auto *file:{&settings,&dataset,&checkpoint}){check(fs::is_regular_file(*file),"regular settings/dataset/checkpoint files required");*file=fs::canonical(*file);}
    check(!fs::exists(output)&&fs::is_directory(output.parent_path()),"new output leaf in existing parent required");
  }
};
struct Snapshot {
  std::map<std::string,torch::Tensor> parameters,buffers;
  std::map<std::string,bool> learned;
  std::vector<torch::Tensor> scaler_values,bank_values;
  std::string scaler,bank,state,bank_source,bank_scaler,initial_state;int64_t updates,seed;double scale_floor;
  template<class Model>Snapshot(const Model &model,const cp::FrozenScaler &input):
      scaler(input.identity()),bank(model->normalizer().identity()),state(model->state_identity()),
      bank_source(model->normalizer().source_identity),bank_scaler(model->normalizer().input_scaler_identity),initial_state(model->initial_state_identity()),
      updates(model->causal_updates()),seed(model->initialization_seed()),scale_floor(input.scale_floor){
    for(const auto &p:model->named_parameters()){parameters.emplace(p.key(),p.value().detach().cpu().clone());learned.emplace(p.key(),p.value().requires_grad());}
    for(const auto &p:model->named_buffers())buffers.emplace(p.key(),p.value().detach().cpu().clone());
    for(const auto &x:std::vector<torch::Tensor>{input.mean,input.scale,input.count,input.channel_ids,input.floor_applied})scaler_values.push_back(x.detach().cpu().clone());
    const auto &n=model->normalizer();for(const auto &x:std::vector<torch::Tensor>{n.counts,n.sum,n.sumsq,n.mean,n.scale,n.support})bank_values.push_back(x.detach().cpu().clone());
  }
  template<class Model>void unchanged(const Model &model,const cp::FrozenScaler &input)const{
    const auto p=model->named_parameters(),b=model->named_buffers();check(p.size()==parameters.size()&&b.size()==buffers.size(),"model inventory changed");
    for(const auto &item:parameters)check(p.contains(item.first)&&cp::boundary::same_bits(item.second,p[item.first])&&
        p[item.first].requires_grad()==learned.at(item.first)&&!p[item.first].grad().defined(),"parameter values/flags/gradient slots changed");
    for(const auto &item:buffers)check(b.contains(item.first)&&cp::boundary::same_bits(item.second,b[item.first]),"registered buffer changed");
    const std::vector<torch::Tensor> s{input.mean,input.scale,input.count,input.channel_ids,input.floor_applied};const auto &n=model->normalizer();
    const std::vector<torch::Tensor> normalizer{n.counts,n.sum,n.sumsq,n.mean,n.scale,n.support};
    for(size_t i=0;i<s.size();++i)check(cp::boundary::same_bits(scaler_values[i],s[i]),"input preprocessing tensor bytes changed");
    for(size_t i=0;i<normalizer.size();++i)check(cp::boundary::same_bits(bank_values[i],normalizer[i]),"bank preprocessing tensor bytes changed");
    check(input.identity()==scaler&&model->normalizer().identity()==bank&&model->state_identity()==state&&
        model->causal_updates()==updates&&model->initialization_seed()==seed&&model->initial_state_identity()==initial_state&&
        n.source_identity==bank_source&&n.input_scaler_identity==bank_scaler&&n.frozen&&input.scale_floor==scale_floor,
        "frozen preprocessing/model metadata or update count changed");
  }
};
cp::Encoder adapt_reference(const old::Loaded &reference){
  // Independent tensor storage, evaluation/diagnostic only. Never export it as a fresh instance.
  auto core=impl::LocationScaleDecoderModel(reference.config,reference.model->normalizer());
  const auto source=reference.model->core()->named_parameters(),destination=core->named_parameters();
  const auto source_buffers=reference.model->core()->named_buffers(),destination_buffers=core->named_buffers();
  check(source.size()==destination.size()&&source_buffers.size()==destination_buffers.size(),"reference adaptation inventory");
  {torch::NoGradGuard guard;
    for(const auto &p:source){check(destination.contains(p.key()),"reference adaptation parameter key");destination[p.key()].copy_(p.value());
      check(destination[p.key()].const_data_ptr()!=p.value().const_data_ptr()&&cp::boundary::same_bits(destination[p.key()],p.value()),"independent exact parameter adaptation");}
    for(const auto &p:source_buffers){check(destination_buffers.contains(p.key()),"reference adaptation buffer key");destination_buffers[p.key()].copy_(p.value());
      check(cp::boundary::same_bits(destination_buffers[p.key()],p.value())&&(p.value().numel()==0||destination_buffers[p.key()].const_data_ptr()!=p.value().const_data_ptr()),"independent exact buffer adaptation");}
  }
  auto result=cp::Encoder(core,reference.initialization_seed,reference.scaler.identity(),reference.causal_updates,reference.initial_state_identity,cp::history_length_bar_counts(reference.config));
  cp::validate_model_counts(result);result->eval();result->clear_gradients();return result;
}
struct GradientSet {
  std::vector<torch::Tensor> gaussian,temporal,control,candidate;
};
std::vector<torch::Tensor> derivatives(const torch::Tensor &loss,const std::vector<torch::Tensor> &parameters,bool retain){
  auto values=torch::autograd::grad({loss},parameters,{},retain,false,true);
  check(values.size()==parameters.size(),"gradient inventory");
  for(size_t i=0;i<values.size();++i)if(values[i].defined())check(values[i].sizes()==parameters[i].sizes()&&values[i].device()==parameters[i].device()&&
      values[i].scalar_type()==parameters[i].scalar_type()&&torch::isfinite(values[i]).all().item<bool>(),"finite aligned actual parameter gradient required");
  return values;
}
struct Metrics {
  int64_t tensors{0},values{0},gaussian_tensors{0},temporal_tensors{0},control_tensors{0},candidate_tensors{0};
  double gaussian_squared{0},temporal_squared{0},dot{0},control_squared{0},candidate_squared{0};
  std::vector<torch::Tensor> control_norms,candidate_norms;
  void add(size_t i,const std::vector<torch::Tensor> &parameters,const GradientSet &g){
    ++tensors;values+=parameters[i].numel();torch::Tensor a,b;
    if(g.gaussian[i].defined()){++gaussian_tensors;a=g.gaussian[i].detach().cpu().to(torch::kFloat64);gaussian_squared+=a.square().sum().item<double>();}
    if(g.temporal[i].defined()){++temporal_tensors;b=g.temporal[i].detach().cpu().to(torch::kFloat64);temporal_squared+=b.square().sum().item<double>();}
    if(a.defined()&&b.defined())dot+=(a*b).sum().item<double>();
    if(g.control[i].defined()){++control_tensors;const auto x=g.control[i].detach();control_squared+=x.cpu().to(torch::kFloat64).square().sum().item<double>();control_norms.push_back(x.norm(2));}
    if(g.candidate[i].defined()){++candidate_tensors;const auto x=g.candidate[i].detach();candidate_squared+=x.cpu().to(torch::kFloat64).square().sum().item<double>();candidate_norms.push_back(x.norm(2));}
  }
  std::string json()const{
    check(tensors>0&&std::isfinite(gaussian_squared)&&std::isfinite(temporal_squared)&&std::isfinite(dot)&&
        std::isfinite(control_squared)&&std::isfinite(candidate_squared),"finite gradient summary required");
    const auto norm=[](const std::vector<torch::Tensor> &parts){return parts.empty()?0.:torch::stack(parts).norm(2).item<double>();};
    const double n=std::sqrt(gaussian_squared),t=std::sqrt(temporal_squared),control=norm(control_norms),candidate=norm(candidate_norms);
    check(std::isfinite(control)&&std::isfinite(candidate),"finite same-dtype global norm required");
    std::ostringstream out;out<<std::setprecision(17)<<"{\"parameter_tensors\":"<<tensors<<",\"parameter_values\":"<<values
      <<",\"gaussian_gradient_tensors\":"<<gaussian_tensors<<",\"temporal_gradient_tensors\":"<<temporal_tensors
      <<",\"control_gradient_tensors\":"<<control_tensors<<",\"candidate_gradient_tensors\":"<<candidate_tensors
      <<",\"gaussian_l2_f64\":"<<n<<",\"temporal_l2_f64\":"<<t<<",\"dot_f64\":"<<dot<<",\"cosine_f64\":";
    if(n>0&&t>0)out<<dot/(n*t);else out<<"null";
    out<<",\"weighted_temporal_to_gaussian_ratio_0_1\":";if(n>0)out<<kControlWeight*t/n;else out<<"null";
    out<<",\"weighted_temporal_to_gaussian_ratio_0_03\":";if(n>0)out<<kCandidateWeight*t/n;else out<<"null";
    out<<",\"actual_combined_0_1_l2_f64\":"<<std::sqrt(control_squared)<<",\"actual_combined_0_03_l2_f64\":"<<std::sqrt(candidate_squared)
      <<",\"linear_combination_from_components_0_1_l2_f64\":"<<std::sqrt(std::max(0.,gaussian_squared+kControlWeight*kControlWeight*temporal_squared+2*kControlWeight*dot))
      <<",\"linear_combination_from_components_0_03_l2_f64\":"<<std::sqrt(std::max(0.,gaussian_squared+kCandidateWeight*kCandidateWeight*temporal_squared+2*kCandidateWeight*dot))
      <<",\"actual_combined_0_1_l2_model_dtype\":"<<control<<",\"actual_combined_0_03_l2_model_dtype\":"<<candidate
      <<",\"hypothetical_clip1_scale_0_1\":"<<std::min(1.,kClipNorm/(control+1e-6))
      <<",\"hypothetical_clip1_scale_0_03\":"<<std::min(1.,kClipNorm/(candidate+1e-6))<<'}';return out.str();
  }
};
std::string describe_gradients(const std::vector<std::string> &names,const std::vector<torch::Tensor> &parameters,const GradientSet &g){
  Metrics all,encoder,backbone,decoder,variance;std::ostringstream individual;individual<<'[';
  for(size_t i=0;i<names.size();++i){
    all.add(i,parameters,g);
    if(names[i].rfind("core.encoder.",0)==0)encoder.add(i,parameters,g);
    if(names[i].rfind("core.encoder.level.base.backbone.",0)==0&&names[i].find(".decoder_")==std::string::npos)backbone.add(i,parameters,g);
    if(names[i].find(".decoder_")!=std::string::npos||names[i]=="core.encoder.change_decoder")decoder.add(i,parameters,g);
    if(names[i].rfind("core.variance_",0)==0)variance.add(i,parameters,g);
    Metrics one;one.add(i,parameters,g);if(i)individual<<',';individual<<"{\"name\":"<<quote(names[i])<<",\"metrics\":"<<one.json()<<'}';
  }individual<<']';
  return "{\"all_learned\":"+all.json()+",\"encoder_and_mean_decoder\":"+encoder.json()+
      ",\"backbone_without_waveform_decoder\":"+backbone.json()+",\"waveform_mean_decoder\":"+decoder.json()+
      ",\"variance_head\":"+variance.json()+",\"per_parameter\":"+individual.str()+'}';
}
std::string row_ids(const ow::Dataset &batch){std::ostringstream out;out<<'[';for(int64_t i=0;i<batch.rows();++i){if(i)out<<',';out<<batch.row_ids[i].item<int64_t>();}return out.str()+']';}
int execute(int argc,char **argv){
  if(argc==2&&std::string(argv[1])=="--help"){
    std::cout<<"cpb_v17_gradient_diagnostic --settings CPB_FAMILY_CONF --dataset TRAIN_PT --dataset-sha256 SHA256 --checkpoint ORIGINAL_V16_PT --checkpoint-sha256 SHA256 --mask-seed 949103 --output NEWLEAF\n"
      <<"Fixed64 TRAIN rows,8 contiguous B8 batches, context deletion NONE, no fitting/optimizer/update/export. Original checkpoint is copied for diagnostic gradients only.\n";return 0;
  }
  const Args args(argc,argv);const auto settings_sha=cp::boundary::sha256_file(args.settings);
  check(cp::boundary::sha256_file(args.dataset)==args.dataset_sha&&cp::boundary::sha256_file(args.checkpoint)==args.checkpoint_sha,"independent input pin mismatch");
  const auto executable=fs::canonical("/proc/self/exe");const auto executable_sha=cp::boundary::sha256_file(executable);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);check(torch::cuda::is_available(),"actual CUDA required");
  const auto config=cp::read_settings(args.settings);const auto dataset=ow::read(args.dataset);
  check(dataset.role=="TRAIN"&&dataset.rows()==kRows,"exact64 caller-admitted TRAIN rows required");
  check(dataset.values.size(1)==config.level.channel_count&&dataset.values.size(3)==config.level.input_width&&
      dataset.sampling_interval==config.level.sampling_interval,"dataset/configuration geometry or grid differs");
  check(cp::boundary::sha256_file(args.settings)==settings_sha&&cp::boundary::sha256_file(args.dataset)==args.dataset_sha,"input changed during reading");
  auto reference=old::load_checkpoint(args.checkpoint.string(),config.level.device,&config);
  reference.model->eval();reference.model->clear_gradients();reference.model->validate_frozen();
  auto model=adapt_reference(reference);const Snapshot original_before(reference.model,reference.scaler),adapted_before(model,reference.scaler);
  const auto state_before=model->state_identity();const auto scaler_before=reference.scaler.identity(),bank_before=model->normalizer().identity();
  std::vector<std::string> names;std::vector<torch::Tensor> parameters;
  for(const auto &p:model->named_parameters())if(p.value().requires_grad()){names.push_back(p.key());parameters.push_back(p.value());}
  check(!parameters.empty(),"learned parameter inventory required");
  check(fs::create_directory(args.output),"exclusive output leaf required");
  std::ofstream events(args.output/"batches.jsonl",std::ios::binary);check(bool(events),"open batch diagnostics");
  for(int64_t attempt=0;attempt<kBatches;++attempt){
    std::vector<int64_t> indices;for(int64_t i=0;i<kBatch;++i)indices.push_back(attempt*kBatch+i);const auto batch=ow::subset(dataset,indices);
    const cp::RawHistory raw{batch.values,batch.observed,batch.channel_ids,batch.timestamps,batch.sampling_interval};
    const auto masks=tr::make_mask(raw,batch.cutoffs,config,args.mask_seed,attempt);check(masks.eligible_channels.any().item<bool>(),"all predeclared batches need eligible hidden targets");
    model->train();const auto trace=model->forward_at(raw,batch.cutoffs,masks.hidden,reference.scaler);
    check(trace.nll.eligible_example_count>0&&torch::isfinite(trace.nll.loss).item<bool>()&&torch::isfinite(trace.objective.temporal.loss).item<bool>(),"finite supported losses required");
    const auto control_loss=trace.nll.loss+kControlWeight*trace.objective.temporal.loss;
    const auto candidate_loss=trace.nll.loss+kCandidateWeight*trace.objective.temporal.loss;
    check(cp::boundary::same_bits(control_loss,trace.output.loss),"control objective differs from preserved V16 objective");
    GradientSet gradients;
    gradients.gaussian=derivatives(trace.nll.loss,parameters,true);gradients.temporal=derivatives(trace.objective.temporal.loss,parameters,true);
    gradients.control=derivatives(control_loss,parameters,true);gradients.candidate=derivatives(candidate_loss,parameters,false);
    std::ostringstream row;row<<std::setprecision(17)<<"{\"attempt\":"<<attempt<<",\"row_ids\":"<<row_ids(batch)
      <<",\"mask_counter_seed\":"<<quote(std::to_string(tr::training_counter_seed(args.mask_seed,attempt)))<<",\"gaussian_nll\":"<<trace.nll.loss.item<double>()
      <<",\"temporal_loss\":"<<trace.objective.temporal.loss.item<double>()<<",\"control_objective\":"<<control_loss.item<double>()
      <<",\"candidate_objective\":"<<candidate_loss.item<double>()<<",\"eligible_examples\":"<<trace.nll.eligible_example_count
      <<",\"target_cells\":"<<trace.nll.target_cell_count<<",\"lags\":[";
    for(size_t i=0;i<trace.objective.temporal.lags.size();++i){const auto &lag=trace.objective.temporal.lags[i];if(i)row<<',';
      row<<"{\"lag\":"<<lag.lag<<",\"supported\":"<<(lag.pair_count>0?"true":"false")<<",\"pairs\":"<<lag.pair_count
        <<",\"eligible_channels\":"<<lag.eligible_channel_count<<",\"eligible_examples\":"<<lag.eligible_example_count<<",\"loss\":";
      if(lag.pair_count>0)row<<lag.loss.item<double>();else row<<"null";row<<'}';}
    row<<"],\"gradients\":"<<describe_gradients(names,parameters,gradients)<<'}';events<<row.str()<<'\n';events.flush();check(bool(events),"write batch diagnostics");
    model->eval();model->clear_gradients();model->validate_frozen();original_before.unchanged(reference.model,reference.scaler);adapted_before.unchanged(model,reference.scaler);
  }
  events.close();check(bool(events),"close batch diagnostics");
  check(cp::boundary::sha256_file(args.checkpoint)==args.checkpoint_sha&&cp::boundary::sha256_file(args.dataset)==args.dataset_sha&&
      cp::boundary::sha256_file(args.settings)==settings_sha&&cp::boundary::sha256_file(executable)==executable_sha,"bound input/executable changed during diagnostic");
  original_before.unchanged(reference.model,reference.scaler);adapted_before.unchanged(model,reference.scaler);
  std::ostringstream report;report<<std::setprecision(17)<<"{\"kind\":\"cpb_v17_TRAIN_parameter_gradient_diagnostic_v1\",\"status\":\"passed\",\"dataset\":"<<artifact(args.dataset)
    <<",\"settings\":"<<artifact(args.settings)<<",\"checkpoint\":"<<artifact(args.checkpoint)<<",\"executable\":"<<artifact(executable)
    <<",\"source_identity\":"<<quote(dataset.source_identity)<<",\"split_identity\":"<<quote(dataset.split_identity)<<",\"role\":\"TRAIN\""
    <<",\"membership_admission_owner\":\"bound dataset producer and study admission; not independently assessed here\",\"membership_independently_proved_by_this_program\":false"
    <<",\"batches\":"<<kBatches<<",\"batch_size\":"<<kBatch<<",\"rows\":"<<kRows<<",\"mask_seed\":"<<args.mask_seed
    <<",\"context_deletion\":\"none\",\"control_temporal_weight\":"<<kControlWeight<<",\"candidate_temporal_weight\":"<<kCandidateWeight
    <<",\"global_clip_norm\":"<<kClipNorm<<",\"hypothetical_clip_epsilon\":1e-6,\"parameter_gradient_evaluations\":"<<4*kBatches
    <<",\"learned_parameter_tensors\":"<<parameters.size()<<",\"native_width\":"<<cp::dimensions(config).native_width
    <<",\"checkpoint_initialization_seed\":"<<reference.initialization_seed<<",\"checkpoint_updates\":"<<reference.causal_updates
    <<",\"diagnostic_state_identity\":"<<quote(state_before)<<",\"scaler_identity\":"<<quote(scaler_before)<<",\"bank_identity\":"<<quote(bank_before)
    <<",\"original_and_adapted_state_unchanged\":true,\"preprocessing_fits\":0,\"optimizer_instances\":0,\"physical_training_updates\":0"
    <<",\"inference_artifacts_exported\":0,\"heldout_readouts\":0,\"quality_evaluated\":false,\"batches_artifact\":"<<artifact(args.output/"batches.jsonl")
    <<",\"runtime\":{\"torch\":"<<quote(TORCH_VERSION)<<",\"device\":"<<quote(config.level.device.str())<<",\"cpu_threads\":1,\"cpu_interop_threads\":1,\"actual_CUDA_parameter_gradients\":true}"
    <<",\"group_definitions\":{\"all_learned\":\"all parameters with requires_grad=true\",\"encoder_and_mean_decoder\":\"parameter key starts with core.encoder.\",\"backbone_without_waveform_decoder\":\"parameter key starts with core.encoder.level.base.backbone. and does not contain .decoder_\",\"waveform_mean_decoder\":\"parameter key contains .decoder_ or equals core.encoder.change_decoder\",\"variance_head\":\"parameter key starts with core.variance_\"}"
    <<",\"limitations\":[\"Frozen trained V16 checkpoint and 64 predeclared TRAIN windows only; not a fresh-trajectory diagnosis.\",\"Cosines describe actual separate parameter derivatives at these points, not the cause of historical quality regressions.\",\"Combined derivatives are independently evaluated by autograd; clip scales are hypothetical with no clipping or optimizer step performed.\",\"Loss strength is prospectively fixed; this diagnostic does not select another coefficient.\"]}";
  write_new(args.output/"summary.json",report.str());std::cout<<report.str()<<std::endl;return 0;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
