// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// One separately declared causal trajectory from fresh initialization. No readout fitting.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v14.h"
#include "include/market_data.h"
#include "include/context_deletion.h"
#include "include/calibration.h"
#include <torch/cuda.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v14;
namespace impl=cp::implementation;
namespace tr=cp::training;
namespace md=causal_fresh_v14::market;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
constexpr int64_t batch=8,updates=2048,max_attempts=4096,row_seed=943002,mask_seed=943003,context_seed=943004,query_seed=943005;
void check(bool yes,const std::string &why){TORCH_CHECK(yes,"[CPB-v14 train] ",why);}
double elapsed(const Clock::time_point &start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::string quote(const std::string &s){std::ostringstream out;out<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}return out.str()+'"';}
std::string dispersion_json_fields(){return "\"temporal_dispersion_transform_policy\":"+quote(cp::kTemporalDispersionTransformPolicy)+",\"temporal_dispersion_regularizer\":0.0009765625";}
std::string objective_json_fields(){return "\"objective_policy\":"+quote(cp::kObjectivePolicy)+",\"direct_gradient_routes\":"+quote(cp::implementation::kLocationScaleDecoderDirectGradientRoutes)+",\"mean_huber_delta\":1,\"loss_coefficients\":[1,1],\"variance_error_detached\":true,\"reporting_nll_backpropagated\":false,\"shared_global_clip\":true,\"optimizer_independence_claimed\":false,\"temporal_pooling_policy\":"+quote(cp::kTemporalPoolingPolicy)+",\"temporal_pooling_geometry\":[4,16,32,8],\"temporal_pooling_trailing_lengths\":[4,8,12,16],\"temporal_pooling_mix\":[0.5,0.5],\"empty_local_bin_policy\":"+quote(cp::kEmptyLocalBinPolicy)+",\"temporal_moment_policy\":"+quote(cp::kTemporalMomentPolicy)+",\"temporal_export_layout\":"+quote(cp::kTemporalExportLayout)+",\"temporal_moment_partition\":[4,4],"+dispersion_json_fields()+",\"registered_learned_tensors\":76,\"mandatory_identifiable_learned_tensors\":75,\"data_gradient_gauge_exemption\":\"core.encoder.level.base.backbone.pool_score.bias\"";}
void write(const fs::path &path,const std::string &contents){check(!fs::exists(path),"preserve existing output: "+path.string());std::ofstream out(path,std::ios::binary);check(bool(out),"create output");out<<contents<<'\n';out.close();check(bool(out),"save output");}
void publish(torch::serialize::OutputArchive &archive,const fs::path &path){check(!fs::exists(path),"exclusive archive");archive.save_to(path.string());}
std::string vector_json(const torch::Tensor &tensor){const auto x=tensor.detach().cpu().to(torch::kFloat64).flatten().contiguous();std::ostringstream out;out<<std::setprecision(17)<<'[';for(int64_t i=0;i<x.numel();++i){if(i)out<<',';out<<x[i].item<double>();}return out.str()+']';}
struct Pin{fs::path path;std::string hash;};
void verify(const std::vector<Pin> &pins){for(const auto &p:pins)check(cp::boundary::sha256_file(p.path)==p.hash,"frozen input/source changed: "+p.path.string());}
struct Args {
  fs::path settings,shared,output;std::vector<fs::path> caches;
  Args(int argc,char **argv){for(int i=1;i<argc;i+=2){check(i+1<argc,"flag/value pairs required");const std::string flag=argv[i];const fs::path value=argv[i+1];
      if(flag=="--cache")caches.push_back(value);
      else if(flag=="--settings"){check(settings.empty(),"duplicate settings");settings=value;}
      else if(flag=="--shared"){check(shared.empty(),"duplicate shared");shared=value;}
      else if(flag=="--output"){check(output.empty(),"duplicate output");output=value;}else check(false,"unknown argument: "+flag);}
    check(!settings.empty()&&!shared.empty()&&!output.empty()&&caches.size()==3,"--settings --shared --cache BTC --cache ETH --cache SOL --output NEWLEAF required");
    settings=fs::canonical(settings);shared=fs::canonical(shared);output=fs::absolute(output).lexically_normal();
    for(auto &p:caches)p=fs::canonical(p);check(!fs::exists(output)&&fs::exists(output.parent_path()),"exclusive leaf in existing attempt required");
  }
};
struct Selections {
  std::vector<int64_t> train,probe_train,probe_validation;
  explicit Selections(const fs::path &shared){
    train=md::anchors(md::train_begin,md::train_end,256,4096,false);
    probe_train=md::anchors(md::train_begin,md::train_end,256,112,true);
    probe_validation=md::anchors(md::validation_begin,md::validation_end,256,32,true);
    torch::serialize::InputArchive archive;archive.load_from((shared/"selections.pt").string(),torch::kCPU);
    const auto read=[&](const char *key,const std::vector<int64_t> &rows){torch::Tensor actual;archive.read(key,actual,true);check(actual.device().is_cpu()&&actual.scalar_type()==torch::kInt64&&torch::equal(actual,torch::tensor(rows,torch::kInt64)),"exact frozen selection: "+std::string(key));};
    read("train_anchors",train);read("probe_train_anchors",probe_train);read("probe_validation_anchors",probe_validation);
    read("roles_and_strides",{md::train_begin,md::train_end,md::validation_begin,md::validation_end,md::unused_begin,60000,256,128,256*60000});
  }
};
struct QueryPlan{
  torch::Tensor hidden,target,eligible;
  QueryPlan(const fs::path &shared,const std::string &role,const std::vector<int64_t> &rows){torch::serialize::InputArchive archive;archive.load_from((shared/("query-plan-"+role+".pt")).string(),torch::kCPU);
    torch::Tensor ids;archive.read("row_ids",ids,true);check(ids.scalar_type()==torch::kInt64&&torch::equal(ids,torch::tensor(rows,torch::kInt64)),"fixed query row association");
    archive.read("hidden_recent128",hidden,true);archive.read("target_recent128",target,true);archive.read("eligible_channels",eligible,true);
    for(const auto &x:{hidden,target})check(x.device().is_cpu()&&x.scalar_type()==torch::kBool&&x.sizes()==torch::IntArrayRef({static_cast<int64_t>(rows.size()),3,128,9}),"fixed query mask");
    check(eligible.device().is_cpu()&&eligible.scalar_type()==torch::kBool&&eligible.sizes()==torch::IntArrayRef({static_cast<int64_t>(rows.size()),3}),"fixed query eligibility");
  }
};
struct SavedTensor{std::string name;torch::Tensor value;};
struct Frozen {
  std::vector<SavedTensor> prior,buffers;cp::FrozenScaler scaler;decltype(std::declval<cp::Encoder>()->normalizer().clone()) normalizer;
  Frozen(const cp::Encoder &model,const cp::FrozenScaler &s):scaler{s.mean.clone(),s.scale.clone(),s.count.clone(),s.channel_ids.clone(),s.scale_floor,s.floor_applied.clone()},normalizer(model->normalizer().clone()){
    for(const auto &p:model->named_parameters())if(!p.value().requires_grad())prior.push_back({p.key(),p.value().detach().cpu().clone()});
    for(const auto &p:model->named_buffers())buffers.push_back({p.key(),p.value().detach().cpu().clone()});
    check(!prior.empty(),"literal fixed prior is present");
  }
  void validate(const cp::Encoder &model,const cp::FrozenScaler &s)const{
    model->core()->validate_state();model->core()->validate_fixed_prior();
    for(const auto &p:prior){const auto x=model->named_parameters()[p.name];check(!x.requires_grad()&&!x.grad().defined()&&cp::boundary::same_bits(x,p.value),"fixed literal prior bytes/gradient");}
    check(model->named_buffers().size()==buffers.size(),"same buffer registration");for(const auto &p:buffers)check(cp::boundary::same_bits(model->named_buffers()[p.name],p.value),"immutable registered buffer");
    const auto &n=model->normalizer();check(n.identity()==normalizer.identity()&&n.source_identity==normalizer.source_identity&&n.input_scaler_identity==normalizer.input_scaler_identity&&n.frozen,"fixed bank metadata");
    for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{n.counts,normalizer.counts},{n.sum,normalizer.sum},{n.sumsq,normalizer.sumsq},{n.mean,normalizer.mean},{n.scale,normalizer.scale},{n.support,normalizer.support}})check(cp::boundary::same_bits(a,b),"fixed normalizer tensor");
    check(s.identity()==scaler.identity()&&s.scale_floor==scaler.scale_floor,"fixed input scaler metadata");
    for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{s.mean,scaler.mean},{s.scale,scaler.scale},{s.count,scaler.count},{s.channel_ids,scaler.channel_ids},{s.floor_applied,scaler.floor_applied}})check(cp::boundary::same_bits(a,b),"fixed scaler tensor");
  }
};
struct QueryCapture {double mae{0},nll{0},mean_huber{0},variance_nll{0},total{0},seconds{0};int64_t valid{0},cells{0},calls{0};torch::Tensor row_mae,row_nll,row_mean_huber,row_variance_nll,row_total,row_cells;};
QueryCapture capture(md::Dataset &data,cp::Encoder &model,const cp::FrozenScaler &scaler,const std::vector<int64_t> &rows,
    const QueryPlan &plan,int64_t begin,int64_t end,const fs::path &witness){
  torch::NoGradGuard no_grad;model->eval();QueryCapture result;std::vector<torch::Tensor> maes,nlls,hubers,variance_nlls,totals,counts;
  const auto start=Clock::now();
  for(size_t offset=0;offset<rows.size();offset+=batch){const auto anchors=md::slice(rows,offset,batch);const auto raw=data.load(anchors,begin,end);
    const auto hidden=plan.hidden.narrow(0,offset,anchors.size());
    auto query_config=model->config().level;query_config.mask_ratio=1./16.;
    const auto generated=impl::configurable_spectral_mask_from_hidden(raw.observed,hidden,query_config);
    check(torch::equal(generated.hidden.reshape({static_cast<int64_t>(anchors.size()),3,16,72}).select(-1,0).sum(-1),generated.eligible_channels.to(torch::kInt64)),"saved fixed query hides exactly one of sixteen patches per eligible channel");
    check(torch::equal(hidden,generated.hidden)&&torch::equal(plan.target.narrow(0,offset,anchors.size()),generated.target)&&torch::equal(plan.eligible.narrow(0,offset,anchors.size()),generated.eligible_channels),"unchanged saved query recipe/support");
    const auto trace=model->forward_at(raw,torch::tensor(anchors,torch::kFloat64),hidden,scaler);++result.calls;
    const auto support=trace.target_mask,eligible=trace.nll.eligible_channels;
    check(torch::equal(support.cpu(),plan.target.narrow(0,offset,anchors.size()))&&torch::equal(eligible.cpu(),plan.eligible.narrow(0,offset,anchors.size())),"actual query target/eligibility parity");
    const auto error=torch::where(support,trace.distribution.mean.to(torch::kFloat64)-trace.targets.to(torch::kFloat64),torch::zeros_like(trace.targets,trace.targets.options().dtype(torch::kFloat64))).abs();
    const auto per_channel=error.sum(std::vector<int64_t>{2,3})/trace.nll.target_counts.clamp_min(1).to(torch::kFloat64);
    const auto per_example=torch::where(eligible,per_channel,torch::zeros_like(per_channel)).sum(1)/eligible.sum(1).clamp_min(1).to(torch::kFloat64);
    const auto valid=trace.nll.eligible_examples;
    check(per_example.is_cuda()&&trace.nll.loss.is_cuda()&&trace.nll.loss.scalar_type()==torch::kFloat64&&torch::isfinite(per_example).all().item<bool>(),"finite actual CUDA standardized query MAE and separate NLL");
    result.mae+=per_example.masked_select(valid).sum().item<double>();result.nll+=trace.nll.example_loss.masked_select(valid).sum().item<double>();result.valid+=valid.sum().item<int64_t>();result.cells+=trace.nll.target_cell_count;
    const auto total_rows=trace.objective.mean_huber.example_loss+trace.objective.variance_nll.example_loss;
    result.mean_huber+=trace.objective.mean_huber.example_loss.masked_select(valid).sum().item<double>();
    result.variance_nll+=trace.objective.variance_nll.example_loss.masked_select(valid).sum().item<double>();result.total+=total_rows.masked_select(valid).sum().item<double>();
    hubers.push_back(trace.objective.mean_huber.example_loss.cpu());variance_nlls.push_back(trace.objective.variance_nll.example_loss.cpu());totals.push_back(total_rows.cpu());
    maes.push_back(per_example.cpu());nlls.push_back(trace.nll.example_loss.cpu());counts.push_back(trace.nll.target_counts.sum(1).cpu());
    if(offset==0){torch::serialize::OutputArchive archive;archive.write("row_ids",torch::tensor(anchors,torch::kInt64),true);archive.write("raw",raw.data,true);archive.write("observed",raw.observed,true);archive.write("timestamps",raw.timestamps,true);archive.write("hidden",hidden,true);archive.write("visible",trace.visible.cpu(),true);archive.write("target_mask",support.cpu(),true);archive.write("targets",trace.targets.cpu(),true);archive.write("mean",trace.distribution.mean.cpu(),true);archive.write("variance",trace.distribution.variance.cpu(),true);archive.write("native",trace.output.encoding.z_contextual_global.cpu(),true);archive.write("row_mae",per_example.cpu(),true);archive.write("row_nll",trace.nll.example_loss.cpu(),true);archive.write("row_mean_huber",trace.objective.mean_huber.example_loss.cpu(),true);archive.write("row_variance_nll",trace.objective.variance_nll.example_loss.cpu(),true);archive.write("row_total_objective",total_rows.cpu(),true);publish(archive,witness);}
  }
  torch::cuda::synchronize();result.seconds=elapsed(start);check(result.valid==static_cast<int64_t>(rows.size()),"complete eligible query population");result.mae/=result.valid;result.nll/=result.valid;
  result.mean_huber/=result.valid;result.variance_nll/=result.valid;result.total/=result.valid;
  result.row_mae=torch::cat(maes);result.row_nll=torch::cat(nlls);result.row_mean_huber=torch::cat(hubers);result.row_variance_nll=torch::cat(variance_nlls);result.row_total=torch::cat(totals);result.row_cells=torch::cat(counts);return result;
}
std::string query_json(const QueryCapture &q){std::ostringstream out;out<<std::setprecision(17)<<"{\"standardized_fixed_query_MAE\":"<<q.mae<<",\"Gaussian_NLL_F64\":"<<q.nll<<",\"mean_Huber_F64\":"<<q.mean_huber<<",\"variance_NLL_F64\":"<<q.variance_nll<<",\"total_objective_F64\":"<<q.total<<",\"valid_rows\":"<<q.valid<<",\"target_cells\":"<<q.cells<<",\"CUDA_encoder_decoder_calls\":"<<q.calls<<",\"seconds_including_loader_witness_IO\":"<<q.seconds<<",\"row_mae\":"<<vector_json(q.row_mae)<<",\"row_nll\":"<<vector_json(q.row_nll)<<",\"row_mean_huber\":"<<vector_json(q.row_mean_huber)<<",\"row_variance_nll\":"<<vector_json(q.row_variance_nll)<<",\"row_total_objective\":"<<vector_json(q.row_total)<<",\"row_target_cells\":"<<vector_json(q.row_cells)<<'}';return out.str();}
std::string category(const std::string &name){
  if(name.find("uncertainty_projection")!=std::string::npos)return"uncertainty_projection";if(name.find("variance_projection")!=std::string::npos)return"variance_matrix";if(name.find("variance_bias")!=std::string::npos)return"variance_bias";
  if(name.find("change_projection")!=std::string::npos)return"change_projection";if(name.find("change_decoder")!=std::string::npos)return"change_wave_decoder";
  if(name.find("relation_projection")!=std::string::npos)return"parent_content_relation";
  if(name.find("decoder")!=std::string::npos)return"parent_decoder";if(name.find("global")!=std::string::npos)return"parent_global_pool";
  return"parent_backbone_projection";
}
struct ParameterEvidence{int64_t values{0},gradient_steps{0},moved_steps{0};bool trainable{false};torch::Tensor initial;};
using Evidence=std::map<std::string,ParameterEvidence>;
Evidence initial_evidence(const cp::Encoder &model){Evidence out;for(const auto &p:model->named_parameters()){check(p.value().is_cuda()&&p.value().scalar_type()==torch::kFloat32,"actual CUDA F32 parameter");out.emplace(p.key(),ParameterEvidence{p.value().numel(),0,0,p.value().requires_grad(),p.value().detach().cpu().clone()});}return out;}
constexpr const char *export_weight="core.encoder.level.base.backbone.export_projection.weight";
constexpr const char *export_bias="core.encoder.level.base.backbone.export_projection.bias";
constexpr const char *pool_positions="core.encoder.level.base.backbone.pool_positions.weight";
constexpr const char *pool_gauge="core.encoder.level.base.backbone.pool_score.bias";
torch::Tensor prior_value(const std::vector<SavedTensor> &before,const std::string &name){for(const auto &v:before)if(v.name==name)return v.value;check(false,"missing actual pre-step parameter: "+name);return {};}
double norm64(const torch::Tensor &value){return value.detach().to(torch::kFloat64).square().sum().sqrt().item<double>();}
torch::Tensor centered64(const torch::Tensor &value){const auto x=value.detach().to(torch::kFloat64);return x-x.mean();}
torch::Tensor between_bin_coefficients64(const torch::Tensor &value){
  const auto x=value.detach().to(torch::kFloat64);check(x.numel()==16,"four original position blocks require16 values");
  const auto means=x.reshape({4,4}).mean(1);
  const auto basis=torch::tensor(std::vector<double>{1,1,-1,-1,1,-1,1,-1,1,-1,-1,1},means.options()).reshape({3,4});
  return torch::matmul(basis,means); // Full R16 directions repeat each sign four times and divide by4.
}
double between_bin_norm64(const torch::Tensor &value){
  const auto means=value.detach().to(torch::kFloat64).reshape({4,4}).mean(1);
  const double squared=(4.*(means-means.mean()).square().sum()).item<double>();
  const double coefficients=between_bin_coefficients64(value).square().sum().item<double>();
  check(std::isfinite(squared)&&std::abs(squared-coefficients)<=1e-12+1e-10*std::abs(squared),"F64 full-position Hadamard Parseval identity");
  return std::sqrt(squared);
}
struct TemporalLearning{
  struct Group{int64_t weight_gradient_steps{0},bias_gradient_steps{0},weight_movement_steps{0},bias_movement_steps{0},position_centered_gradient_steps{0},position_centered_movement_steps{0};};
  std::array<Group,4> groups;std::array<Group,8> moment_groups;std::array<int64_t,3> between_gradient_steps{},between_movement_steps{};std::ofstream rows,gauge,between,moments;torch::Tensor initial_weight,initial_bias,initial_positions,initial_gauge;
  int64_t recorded_updates{0},gauge_defined_steps{0},gauge_nonzero_gradient_steps{0},gauge_movement_steps{0};double gauge_max_abs_gradient{0};
  TemporalLearning(const cp::Encoder &model,const fs::path &output):rows(output/"temporal-projection-learning.csv"),gauge(output/"temporal-pool-score-bias-learning.csv"),between(output/"temporal-between-bin-position-learning.csv"),moments(output/"temporal-moment-subgroup-learning.csv"){
    const auto named=model->named_parameters();
    check(named.find(export_weight)&&named.find(export_bias)&&named.find(pool_positions)&&named.find(pool_gauge),"canonical actual temporal parameter paths");
    check(named[export_weight].sizes()==torch::IntArrayRef({32,64})&&named[export_bias].sizes()==torch::IntArrayRef({32})&&named[pool_positions].sizes()==torch::IntArrayRef({16,1})&&named[pool_gauge].sizes()==torch::IntArrayRef({1}),"declared temporal row/position/gauge geometry");
    initial_weight=named[export_weight].detach().cpu().clone();initial_bias=named[export_bias].detach().cpu().clone();initial_positions=named[pool_positions].detach().cpu().clone();initial_gauge=named[pool_gauge].detach().cpu().clone();
    check(bool(rows)&&bool(gauge)&&bool(between)&&bool(moments),"create exclusive temporal learning logs");
    moments<<"update,bin,moment,export_row_begin,export_row_end,weight_gradient_l2,bias_gradient_l2,weight_movement_l2,bias_movement_l2,nonzero_export_gradient_values,moved_export_values\n"<<std::setprecision(17);
    rows<<"update,bin,export_row_begin,export_row_end,position_begin,position_end,weight_gradient_l2,bias_gradient_l2,weight_movement_l2,bias_movement_l2,nonzero_export_gradient_values,moved_export_values,pool_position_centered_gradient_l2,pool_position_centered_movement_l2,nonzero_centered_position_gradient_values,moved_centered_position_values\n"<<std::setprecision(17);
    gauge<<"update,gradient_defined,clipped_data_gradient,step_movement,weight_before,weight_after\n"<<std::setprecision(17);
    between<<"update,between_bin_gradient_l2,between_bin_movement_l2,contrast_0_gradient,contrast_1_gradient,contrast_2_gradient,contrast_0_movement,contrast_1_movement,contrast_2_movement\n"<<std::setprecision(17);
  }
  void record(const cp::Encoder &model,const std::vector<SavedTensor> &before,int64_t update){
    check(update==++recorded_updates,"temporal evidence exact successful update order");const auto named=model->named_parameters();
    const auto w=named[export_weight],b=named[export_bias],p=named[pool_positions],g=named[pool_gauge];
    check(w.grad().defined()&&b.grad().defined()&&p.grad().defined(),"actual temporal data gradient arrays");
    const auto dw=w.detach().to(torch::kFloat64)-prior_value(before,export_weight).to(torch::kFloat64),db=b.detach().to(torch::kFloat64)-prior_value(before,export_bias).to(torch::kFloat64),dp=p.detach().to(torch::kFloat64)-prior_value(before,pool_positions).to(torch::kFloat64);
    const auto pg_between=between_bin_coefficients64(p.grad()),pm_between=between_bin_coefficients64(dp);
    between<<update<<','<<between_bin_norm64(p.grad())<<','<<between_bin_norm64(dp);
    for(int64_t direction=0;direction<3;++direction){const double v=pg_between[direction].item<double>();check(std::isfinite(v),"finite observed between-bin data gradient");between_gradient_steps[direction]+=v!=0.;between<<','<<v;}
    for(int64_t direction=0;direction<3;++direction){const double v=pm_between[direction].item<double>();check(std::isfinite(v),"finite actual between-bin movement");between_movement_steps[direction]+=v!=0.;between<<','<<v;}between<<'\n';
    for(int64_t bin=0;bin<4;++bin){const auto wg=w.grad().narrow(0,bin*8,8),bg=b.grad().narrow(0,bin*8,8),wm=dw.narrow(0,bin*8,8),bm=db.narrow(0,bin*8,8);
      const auto pg=centered64(p.grad().narrow(0,bin*4,4)),pm=centered64(dp.narrow(0,bin*4,4));
      const auto nw=norm64(wg),nb=norm64(bg),mw=norm64(wm),mb=norm64(bm),ng=norm64(pg),nm=norm64(pm);
      check(std::isfinite(nw)&&std::isfinite(nb)&&std::isfinite(mw)&&std::isfinite(mb)&&std::isfinite(ng)&&std::isfinite(nm),"finite actual clipped temporal learning evidence");
      auto &counts=groups[bin];counts.weight_gradient_steps+=nw>0;counts.bias_gradient_steps+=nb>0;counts.weight_movement_steps+=mw>0;counts.bias_movement_steps+=mb>0;counts.position_centered_gradient_steps+=ng>0;counts.position_centered_movement_steps+=nm>0;
      rows<<update<<','<<bin<<','<<bin*8<<','<<(bin+1)*8<<','<<bin*4<<','<<(bin+1)*4<<','<<nw<<','<<nb<<','<<mw<<','<<mb<<','<<(wg.ne(0).sum().item<int64_t>()+bg.ne(0).sum().item<int64_t>())<<','<<(wm.ne(0).sum().item<int64_t>()+bm.ne(0).sum().item<int64_t>())<<','<<ng<<','<<nm<<','<<pg.ne(0).sum().item<int64_t>()<<','<<pm.ne(0).sum().item<int64_t>()<<'\n';
      for(int64_t moment=0;moment<2;++moment){const int64_t row=bin*8+moment*4;
        const auto swg=w.grad().narrow(0,row,4),sbg=b.grad().narrow(0,row,4),swm=dw.narrow(0,row,4),sbm=db.narrow(0,row,4);
        const double gnw=norm64(swg),gnb=norm64(sbg),mnw=norm64(swm),mnb=norm64(sbm);
        check(std::isfinite(gnw)&&std::isfinite(gnb)&&std::isfinite(mnw)&&std::isfinite(mnb),"finite observed intermediate pooling moment subgroup evidence");
        auto &c=moment_groups[bin*2+moment];c.weight_gradient_steps+=gnw>0;c.bias_gradient_steps+=gnb>0;c.weight_movement_steps+=mnw>0;c.bias_movement_steps+=mnb>0;
        moments<<update<<','<<bin<<','<<(moment?"regularized_stddev":"mean")<<','<<row<<','<<row+4<<','<<gnw<<','<<gnb<<','<<mnw<<','<<mnb<<','<<(swg.ne(0).sum().item<int64_t>()+sbg.ne(0).sum().item<int64_t>())<<','<<(swm.ne(0).sum().item<int64_t>()+sbm.ne(0).sum().item<int64_t>())<<'\n';
        check(torch::equal(torch::cat({wg.narrow(0,0,4),wg.narrow(0,4,4)}),wg)&&torch::equal(torch::cat({bg.narrow(0,0,4),bg.narrow(0,4,4)}),bg),"mean/regularized-stddev subgroups partition the actual eight-row export slice");
      }
    }
    const bool defined=g.grad().defined();const double gradient=defined?g.grad().item<double>():0.,old=prior_value(before,pool_gauge).item<double>(),now=g.item<double>();
    check(std::isfinite(gradient)&&std::isfinite(now),"finite observed gauge scalar");gauge_defined_steps+=defined;gauge_nonzero_gradient_steps+=gradient!=0;gauge_movement_steps+=now!=old;gauge_max_abs_gradient=std::max(gauge_max_abs_gradient,std::abs(gradient));
    gauge<<update<<','<<(defined?1:0)<<','<<gradient<<','<<(now-old)<<','<<old<<','<<now<<'\n';
  }
  void witness(torch::serialize::OutputArchive &archive,const cp::Encoder &model,const std::vector<SavedTensor> &before)const{
    const auto named=model->named_parameters();for(const auto &[key,name]:std::vector<std::pair<std::string,std::string>>{{"temporal_export_weight",export_weight},{"temporal_export_bias",export_bias},{"temporal_pool_positions",pool_positions},{"temporal_pool_score_bias",pool_gauge}}){
      const auto value=named[name],gradient=value.grad().defined()?value.grad():torch::zeros_like(value);
      archive.write(key+"_before",prior_value(before,name).detach().cpu(),true);archive.write(key+"_after",value.detach().cpu(),true);archive.write(key+"_clipped_gradient",gradient.detach().cpu(),true);
    }
  }
  void finish(const cp::Encoder &model,const fs::path &output){
    rows.close();gauge.close();between.close();moments.close();check(bool(rows)&&bool(gauge)&&bool(between)&&bool(moments)&&recorded_updates==2048,"persist all8192 group,16384 moment subgroup,2048 gauge and2048 between-bin rows");const auto named=model->named_parameters();std::ostringstream out;out<<std::setprecision(17)<<"{\"kind\":\"cpb_v14_actual_temporal_pooling_learning_v1\",\"status\":\"passed\",\"updates\":2048,\"group_rows\":8192,\"moment_subgroup_rows\":16384,\"temporal_moment_policy\":"<<quote(cp::kTemporalMomentPolicy)<<","<<dispersion_json_fields()<<",\"temporal_export_layout\":"<<quote(cp::kTemporalExportLayout)<<",\"temporal_moment_partition\":[4,4],\"temporal_pooling_policy\":"<<quote(cp::kTemporalPoolingPolicy)<<",\"temporal_pooling_geometry\":[4,16,32,8],\"temporal_pooling_trailing_lengths\":[4,8,12,16],\"temporal_pooling_mix\":[0.5,0.5],\"empty_local_bin_policy\":"<<quote(cp::kEmptyLocalBinPolicy)<<",\"position_fields_semantics\":\"position_begin/end are fixed disjoint quarters for positional-contrast diagnostics only; nested local supports are defined solely by temporal_pooling_trailing_lengths\",\"gradients\":\"observed-after-unchanged-global-clip1-before-zero-grad;not-independent-upstream-reconstruction\",\"movement_includes_AdamW_weight_decay\":true,\"weight_decay_counted_as_data_gradient\":false,\"extra_encoder_calls\":0,\"groups\":[";
    for(int64_t bin=0;bin<4;++bin){const auto &c=groups[bin];check(c.weight_gradient_steps>0&&c.bias_gradient_steps>0&&c.weight_movement_steps>0&&c.bias_movement_steps>0&&c.position_centered_gradient_steps>0&&c.position_centered_movement_steps>0,"each temporal group has observed data gradients and movement at least once, not every update");
      const bool changed_weight=!cp::boundary::same_bits(named[export_weight].narrow(0,bin*8,8),initial_weight.narrow(0,bin*8,8)),changed_bias=!cp::boundary::same_bits(named[export_bias].narrow(0,bin*8,8),initial_bias.narrow(0,bin*8,8));
      const double position_change=norm64(centered64(named[pool_positions].narrow(0,bin*4,4))-centered64(initial_positions.narrow(0,bin*4,4).to(named[pool_positions].device())));check(changed_weight&&changed_bias&&position_change>0,"final export rows and identifiable position contrasts changed");
      if(bin)out<<',';out<<"{\"bin\":"<<bin<<",\"export_row_begin\":"<<bin*8<<",\"export_row_end\":"<<(bin+1)*8<<",\"position_begin\":"<<bin*4<<",\"position_end\":"<<(bin+1)*4<<",\"weight_nonzero_data_gradient_steps\":"<<c.weight_gradient_steps<<",\"bias_nonzero_data_gradient_steps\":"<<c.bias_gradient_steps<<",\"weight_movement_steps\":"<<c.weight_movement_steps<<",\"bias_movement_steps\":"<<c.bias_movement_steps<<",\"position_centered_data_gradient_steps\":"<<c.position_centered_gradient_steps<<",\"position_centered_movement_steps\":"<<c.position_centered_movement_steps<<",\"final_weight_changed\":true,\"final_bias_changed\":true,\"final_position_centered_change_l2\":"<<position_change<<'}';
    }
    out<<"],\"moment_subgroups\":[";
    for(int64_t bin=0;bin<4;++bin)for(int64_t moment=0;moment<2;++moment){const int64_t index=bin*2+moment,row=bin*8+moment*4;const auto &c=moment_groups[index];
      const bool changed_weight=!cp::boundary::same_bits(named[export_weight].narrow(0,row,4),initial_weight.narrow(0,row,4)),changed_bias=!cp::boundary::same_bits(named[export_bias].narrow(0,row,4),initial_bias.narrow(0,row,4));
      check(c.weight_gradient_steps>0&&c.bias_gradient_steps>0&&c.weight_movement_steps>0&&c.bias_movement_steps>0&&changed_weight&&changed_bias,"each intermediate pooling mean4/regularized-stddev4 subgroup has data-gradient events, actual movement and final changes; not every scalar/update");
      if(index)out<<',';out<<"{\"bin\":"<<bin<<",\"moment\":"<<quote(moment?"regularized_stddev":"mean")<<",\"export_row_begin\":"<<row<<",\"export_row_end\":"<<row+4<<",\"weight_nonzero_data_gradient_steps\":"<<c.weight_gradient_steps<<",\"bias_nonzero_data_gradient_steps\":"<<c.bias_gradient_steps<<",\"weight_movement_steps\":"<<c.weight_movement_steps<<",\"bias_movement_steps\":"<<c.bias_movement_steps<<",\"final_weight_changed\":true,\"final_bias_changed\":true}";
    }
    const auto final_delta=named[pool_positions].detach().to(torch::kFloat64)-initial_positions.to(named[pool_positions].device()).to(torch::kFloat64);
    const auto final_coefficients=between_bin_coefficients64(final_delta);const double final_between=between_bin_norm64(final_delta);check(final_between>0,"final between-bin position projection moved");
    out<<"],\"between_bin_positions\":{\"basis\":\"hadamard-full16-signs-div4-v1\",\"basis_signs\":[[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],\"full_position_divisor\":4,\"position_bin_width\":4,\"rows\":2048,\"gradient_steps\":[";
    for(int64_t direction=0;direction<3;++direction){check(between_gradient_steps[direction]>0&&between_movement_steps[direction]>0,"all three restored directions have data-gradient and movement events");if(direction)out<<',';out<<between_gradient_steps[direction];}out<<"],\"movement_steps\":[";
    for(int64_t direction=0;direction<3;++direction){if(direction)out<<',';out<<between_movement_steps[direction];}out<<"],\"final_contrast_changes\":[";
    for(int64_t direction=0;direction<3;++direction){if(direction)out<<',';out<<final_coefficients[direction].item<double>();}out<<"],\"final_projection_change_l2\":"<<final_between<<",\"gradient_definition\":\"observed-post-globalclip-data-gradient\",\"movement_includes_AdamW_weight_decay\":true,\"weight_decay_counted_as_data_learning\":false,\"extra_encoder_calls\":0},\"gauge\":{\"name\":"<<quote(pool_gauge)<<",\"data_gradient_requirement_exempt\":true,\"registered_in_same_AdamW\":true,\"gradient_defined_steps\":"<<gauge_defined_steps<<",\"nonzero_recorded_gradient_steps\":"<<gauge_nonzero_gradient_steps<<",\"maximum_absolute_recorded_gradient\":"<<gauge_max_abs_gradient<<",\"movement_steps\":"<<gauge_movement_steps<<",\"final_movement\":"<<(named[pool_gauge].item<double>()-initial_gauge.item<double>())<<",\"movement_is_not_data_learning_proof\":true},\"artifacts\":[";
    bool first=true;for(const auto &name:{"temporal-projection-learning.csv","temporal-pool-score-bias-learning.csv","temporal-between-bin-position-learning.csv","temporal-moment-subgroup-learning.csv"}){const auto path=output/name;if(!first)out<<',';first=false;out<<"{\"path\":"<<quote(path.string())<<",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":"<<quote(cp::boundary::sha256_file(path))<<'}';}out<<"]}";write(output/"temporal-projection-learning.json",out.str());
  }
};
struct ResumeSnapshot {
  std::vector<SavedTensor> parameters,buffers,moments;std::vector<int64_t> steps;
  ResumeSnapshot(const cp::Encoder &model,const torch::optim::AdamW &optimizer){
    for(const auto &p:model->named_parameters()){parameters.push_back({p.key(),p.value().detach().cpu().clone()});const auto it=optimizer.state().find(p.value().unsafeGetTensorImpl());if(it==optimizer.state().end()){steps.push_back(-1);continue;}
      const auto *state=dynamic_cast<const torch::optim::AdamWParamState*>(it->second.get());check(state,"typed AdamW state");steps.push_back(state->step());moments.push_back({p.key()+".mean",state->exp_avg().detach().cpu().clone()});moments.push_back({p.key()+".variance",state->exp_avg_sq().detach().cpu().clone()});}
    for(const auto &p:model->named_buffers())buffers.push_back({p.key(),p.value().detach().cpu().clone()});
  }
  void compare(const ResumeSnapshot &other)const{check(steps==other.steps,"typed resume exact optimizer steps");const auto compare=[](const auto &a,const auto &b){check(a.size()==b.size(),"resume state inventory");for(size_t i=0;i<a.size();++i)check(a[i].name==b[i].name&&cp::boundary::same_bits(a[i].value,b[i].value),"typed resume exact values/moments/order");};compare(parameters,other.parameters);compare(buffers,other.buffers);compare(moments,other.moments);}
};
int run(int argc,char **argv){
  check(fs::exists("/.dockerenv")&&torch::cuda::is_available(),"existing CUDA container required");torch::set_num_threads(1);torch::set_num_interop_threads(1);Args args(argc,argv);
  const auto config=cp::read_settings(args.settings);check(config.level.device==torch::Device(torch::kCUDA,0)&&config.level.dropout==0&&config.level.mask_ratio==.25,"frozen CUDA0 zero-dropout mask25 geometry");
  check(cp::temporal_pooling_geometry(config)==std::vector<int64_t>({4,16,32,8})&&cp::temporal_pooling_trailing_lengths()==std::vector<int64_t>({4,8,12,16}),"declared fixed nested trailing4/8/12/16 support and eight-row export geometry");
  check(cp::temporal_pooling_mix()==std::vector<double>({.5,.5})&&std::string(cp::kEmptyLocalBinPolicy)=="zero-whole-export-slice-after-bias-v1","fixed half local/global mixture and unchanged empty-local-bin zeros");
  check(std::string(cp::kTemporalDispersionTransformPolicy)=="sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1"&&
      cp::kTemporalDispersionRegularizer==0.0009765625&&cp::kTemporalDispersionRootOffset==0.03125,
      "fixed binary-exact tau/root-offset and F32 transform graph, never learned or fitted");
  std::vector<Pin> pins{{args.settings,cp::boundary::sha256_file(args.settings)},
    {args.shared/"selections.pt","10df626e83cc652edfdce2f0cb2eae72ccf802a3a0081cf5db8db71a7ef8bdb2"},
    {args.shared/"query-plan-train.pt","1853a271a32bd6635c122b36dfd6373247815178cff1525e95d852c5aafc137e"},
    {args.shared/"query-plan-validation.pt","5a98906368f3b56508d81b254d598594573545babd3c8e8ff67979a30b12018d"}};
  const std::array<const char*,3> cache_hashes{"67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58","0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa","364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7"};
  for(size_t i=0;i<3;++i)pins.push_back({args.caches[i],cache_hashes[i]});verify(pins);
  Selections selections(args.shared);QueryPlan query_train(args.shared,"train",selections.probe_train),query_validation(args.shared,"validation",selections.probe_validation);md::Dataset data(args.caches);
  check(fs::create_directory(args.output),"exclusive new fresh run leaf");fs::create_directory(args.output/"checkpoints");fs::create_directory(args.output/"witnesses");
  const std::string calibration_source="CPB-v14.market-01;fresh-TRAIN-only-CACHE88:67745d6a/0aad9d11/364fbf2c;selection10df626e;4096windows;recent128;IDs0,1,2;no-encoder-calibration";
  auto calibration=causal_fresh_v14::calibration::fit(data,config,selections.train,args.output/"calibration",calibration_source);
  for(const auto &name:{"input-scaler.pt","bank-standardizer.pt","first-TRAIN-batch-witness.pt","summary.json"}){const auto path=args.output/"calibration"/name;pins.push_back({path,cp::boundary::sha256_file(path)});}
  auto loaded=cp::fresh(config,calibration.scaler,calibration.bank,943001);auto model=loaded.model;auto scaler=loaded.scaler;
  check(loaded.source_updates==0&&loaded.causal_updates==0&&model->source_updates()==0&&model->initialization_seed()==943001,"fresh factory has zero inherited updates and declared seed");
  int64_t constructor_parameter_tensors=0,constructor_values=0,constructor_buffer_tensors=0,constructor_learned_values=0,constructor_fixed_values=0;std::string direct_state_identity;
  {
    torch::NoGradGuard guard;torch::manual_seed(943001);impl::LocationScaleDecoderModel direct(config,calibration.bank);
    const auto actual=model->core()->named_parameters(),reference=direct->named_parameters();check(actual.size()==reference.size(),"fresh constructor parameter inventory");
    for(const auto &p:reference){const auto *q=actual.find(p.key());check(q&&q->requires_grad()==p.value().requires_grad()&&cp::boundary::same_bits(*q,p.value()),"all initial tensors equal direct seeded constructor, no imported weights");++constructor_parameter_tensors;constructor_values+=p.value().numel();if(p.value().requires_grad())constructor_learned_values+=p.value().numel();else constructor_fixed_values+=p.value().numel();}
    const auto actual_buffers=model->core()->named_buffers(),reference_buffers=direct->named_buffers();check(actual_buffers.size()==reference_buffers.size(),"fresh constructor buffer inventory");
    for(const auto &p:reference_buffers){const auto *q=actual_buffers.find(p.key());check(q&&cp::boundary::same_bits(*q,p.value()),"fresh constructor buffer bytes");++constructor_buffer_tensors;}
    direct->validate_fixed_prior();direct_state_identity=cp::boundary::state_identity(direct);check(direct_state_identity==model->state_identity(),"full fresh seeded state identity");
    check(constructor_parameter_tensors==77&&constructor_values==298640&&constructor_learned_values==294752&&constructor_fixed_values==3888,"fresh learned/fixed parameter partition");
  }
  Frozen frozen(model,scaler);frozen.validate(model,scaler);
  std::ostringstream bindings;bindings<<"[";for(size_t i=0;i<pins.size();++i){if(i)bindings<<',';bindings<<"{\"path\":"<<quote(pins[i].path.string())<<",\"sha256\":"<<quote(pins[i].hash)<<'}';}bindings<<']';write(args.output/"input-bindings.json",bindings.str());
  const tr::TrainingIdentity identity{"MKT-HIST-2;CACHE88:67745d6a/0aad9d11/364fbf2c;genuine-F64;C3H128F9;IDs0,1,2",
      "selection-sha256:10df626e83cc652edfdce2f0cb2eae72ccf802a3a0081cf5db8db71a7ef8bdb2;same4096TRAIN112probeTRAIN32VAL;past-only",
      std::string("CPB-v14.market-01;fresh-constructor-seed943001;source-updates0;no-import;huber1-mean-plus-detached-error-gaussian-variance-v1;delta1;unit-component-weights;Huber-mean-live-and-detached-error-variance-routes;reporting-NLL-never-backward;conditional-variance-head-live;new-TRAIN-scaler:")+scaler.identity()+";new-TRAIN-bank:"+model->normalizer().identity()+";fixed-literal-prior;B8;2048;mask25/context15;single-AdamW.001/.0001;shared-global-clip1-couples-branch-scaling;original-position-nested-trailing-local4/8/12/16-endpoint16;per-window-mean4-regularized-stddev4;F32-add-sqrt-subtract;tau2^-10;root-offset1/32;half-local-half-shared-global-attention;zero-empty-local-slices-after-bias;three-Hadamard-between-position-directions;75-identifiable-plus-pool-score-bias-gauge;row943002/mask943003/context943004;fixed-saved-query-masks-no-query-RNG;no-label-loss"};
  tr::TrainingProgress progress{0,0,row_seed,mask_seed,context_seed};auto optimizer=tr::make_optimizer(model);tr::validate_optimizer(model,*optimizer,progress);Evidence evidence=initial_evidence(model);TemporalLearning temporal_learning(model,args.output);
  double checkpoint_seconds=0,loader_mask_seconds=0,compute_seconds=0,evidence_seconds=0;auto io_start=Clock::now();model->eval();model->clear_gradients();const auto initial_checkpoint=args.output/"checkpoints/inference-0.pt";cp::save_checkpoint(initial_checkpoint.string(),model,scaler,0);checkpoint_seconds+=elapsed(io_start);
  std::ostringstream initialization;initialization<<"{\"kind\":\"cpb_v14_fresh_initialization_v1\",\"status\":\"passed\","<<objective_json_fields()<<",\"initialization_kind\":"<<quote(cp::kInitializationKind)<<",\"initialization_seed\":943001,\"pretrained_ancestor\":\"none\",\"inherited_learned_parameters\":0,\"fresh_learned_parameter_values\":"<<constructor_learned_values<<",\"analytic_fixed_prior_values\":"<<constructor_fixed_values<<",\"source_updates\":0,\"causal_updates\":0,\"old_model_artifacts_opened\":0,\"old_normalization_artifacts_opened\":0,\"all_initial_parameters_equal_direct_seeded_constructor\":true,\"parameter_tensors_checked\":"<<constructor_parameter_tensors<<",\"parameter_values_checked\":"<<constructor_values<<",\"buffer_tensors_checked\":"<<constructor_buffer_tensors<<",\"direct_constructor_state_identity\":"<<quote(direct_state_identity)<<",\"initial_state_identity\":"<<quote(model->initial_state_identity())<<",\"fresh_input_scaler_identity\":"<<quote(scaler.identity())<<",\"fresh_bank_identity\":"<<quote(model->normalizer().identity())<<",\"initial_checkpoint\":{\"path\":"<<quote(initial_checkpoint.string())<<",\"bytes\":"<<fs::file_size(initial_checkpoint)<<",\"sha256\":"<<quote(cp::boundary::sha256_file(initial_checkpoint))<<"}}";write(args.output/"fresh-initialization.json",initialization.str());
  const auto initial_train=capture(data,model,scaler,selections.probe_train,query_train,md::train_begin,md::train_end,args.output/"witnesses/initial-train-query.pt");
  const auto initial_validation=capture(data,model,scaler,selections.probe_validation,query_validation,md::validation_begin,md::validation_end,args.output/"witnesses/initial-validation-query.pt");
  write(args.output/"query-initial.json","{\"train\":"+query_json(initial_train)+",\"validation\":"+query_json(initial_validation)+"}");
  std::ofstream log(args.output/"training.csv");check(bool(log),"new training log");log<<"attempt,update,Gaussian_NLL_F64,mean_Huber_F64,variance_NLL_F64,total_objective_F64,pre_clip_global_norm,clip_factor,eligible_examples,target_cells,nonzero_gradient_tensors,changed_tensors,row_seed,mask_seed,anchors\n"<<std::setprecision(17);
  model->train();const auto trajectory_start=Clock::now();bool resume_verified=false;int64_t trained_cells=0;double first_loss=0,last_loss=0,first_mean=0,last_mean=0,first_variance=0,last_variance=0,first_total=0,last_total=0;
  while(progress.completed_updates<updates&&progress.attempted_batches<max_attempts){
    auto stage=Clock::now();const auto attempt=progress.attempted_batches,row_stream=tr::training_counter_seed(row_seed,attempt),mask_stream=tr::training_counter_seed(mask_seed,attempt);
    std::mt19937_64 random(row_stream);std::uniform_int_distribution<int64_t> pick(0,selections.train.size()-1);std::vector<int64_t> rows;for(int64_t i=0;i<batch;++i)rows.push_back(selections.train[pick(random)]);
    const auto raw=data.load(rows,md::train_begin,md::train_end);const auto masks=impl::configurable_spectral_make_training_mask(raw.observed,config.level,mask_stream);
    if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;loader_mask_seconds+=elapsed(stage);log<<attempt<<','<<progress.completed_updates;for(int column=2;column<14;++column)log<<',';log<<",\"\"\n";continue;}
    const auto context=impl::context_deletion::make_plan(masks,raw.channel_ids,config.level,context_seed,attempt,impl::ContextDeletionRecipe::coordinate15_v1);loader_mask_seconds+=elapsed(stage);
    stage=Clock::now();std::vector<SavedTensor> before;for(const auto &p:model->named_parameters())if(p.value().requires_grad())before.push_back({p.key(),p.value().detach().clone()});torch::cuda::synchronize();evidence_seconds+=elapsed(stage);
    optimizer->zero_grad(true);torch::cuda::synchronize();stage=Clock::now();
    const auto trace=model->forward_at(raw,torch::tensor(rows,torch::kFloat64),masks.hidden,scaler,context.deleted);
    check(trace.objective.total_loss.is_cuda()&&trace.objective.total_loss.scalar_type()==torch::kFloat64&&torch::isfinite(trace.objective.total_loss).item<bool>()&&trace.objective.total_loss.requires_grad()&&trace.objective.variance_nll.loss.requires_grad()&&!trace.nll.loss.requires_grad(),"finite explicit Huber mean plus detached-error variance F64 objective and separate diagnostic-only NLL copy");trace.objective.total_loss.backward();
    for(const auto &p:model->named_parameters())if(p.value().grad().defined())check(p.value().requires_grad()&&p.value().grad().is_cuda()&&torch::isfinite(p.value().grad()).all().item<bool>(),"finite learned CUDA gradients before optimizer mutation");
    const double pre_clip_norm=torch::nn::utils::clip_grad_norm_(model->parameters(),1.);
    const double clip_factor=std::min(1.,1./(pre_clip_norm+1e-6));optimizer->step();torch::cuda::synchronize();compute_seconds+=elapsed(stage);
    stage=Clock::now();check(torch::equal(trace.visible.cpu(),context.visible)&&torch::equal(trace.target_mask.cpu(),masks.target)&&!(trace.visible&trace.target_mask).any().item<bool>(),"target/context separation with exact final mask");
    int64_t gradient_tensors=0,changed_tensors=0;const auto named=model->named_parameters();
    for(const auto &p:named){const auto gradient=p.value().grad();if(gradient.defined()){check(p.value().requires_grad()&&gradient.is_cuda()&&torch::isfinite(gradient).all().item<bool>(),"finite actual CUDA gradient");if(gradient.abs().sum().item<double>()>0){++evidence.at(p.key()).gradient_steps;++gradient_tensors;}}else check(!p.value().requires_grad()||p.value().is_cuda(),"CUDA inactive parameter");}
    for(const auto &p:before)if(!torch::equal(p.value,named[p.name].detach())){++evidence.at(p.name).moved_steps;++changed_tensors;}
    ++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
    check(model->causal_updates()==progress.completed_updates,"live causal update counter matches completed CUDA steps");
    last_loss=trace.nll.loss.item<double>();last_mean=trace.objective.mean_huber.loss.item<double>();last_variance=trace.objective.variance_nll.loss.item<double>();last_total=trace.objective.total_loss.item<double>();
    if(progress.completed_updates==1){first_loss=last_loss;first_mean=last_mean;first_variance=last_variance;first_total=last_total;}trained_cells+=trace.nll.target_cell_count;
    check(gradient_tensors>0&&changed_tensors>0,"actual CUDA learning at every successful update");temporal_learning.record(model,before,progress.completed_updates);evidence_seconds+=elapsed(stage);
    stage=Clock::now();log<<attempt<<','<<progress.completed_updates<<','<<last_loss<<','<<last_mean<<','<<last_variance<<','<<last_total<<','<<pre_clip_norm<<','<<clip_factor<<','<<trace.nll.eligible_example_count<<','<<trace.nll.target_cell_count<<','<<gradient_tensors<<','<<changed_tensors<<','<<row_stream<<','<<mask_stream<<",\"";
    for(size_t i=0;i<rows.size();++i){if(i)log<<';';log<<rows[i];}log<<"\"\n";
    if(progress.completed_updates==1||progress.completed_updates==2048){
      // These tensors are from this actual pre-step forward, not a replay.
      torch::serialize::OutputArchive a;
      a.write("update",torch::tensor(progress.completed_updates,torch::kInt64),true);a.write("attempt",torch::tensor(attempt,torch::kInt64),true);
      a.write("row_ids",torch::tensor(rows,torch::kInt64),true);a.write("timestamps",raw.timestamps,true);
      a.write("observed",raw.observed,true);a.write("hidden",masks.hidden,true);a.write("context_deleted",context.deleted,true);
      a.write("visible",trace.visible.detach().cpu(),true);a.write("target_mask",trace.target_mask.cpu(),true);a.write("eligible_channels",trace.nll.eligible_channels.cpu(),true);
      a.write("targets",trace.targets.detach().cpu(),true);a.write("mean",trace.distribution.mean.detach().cpu(),true);a.write("variance",trace.distribution.variance.detach().cpu(),true);
      a.write("native",trace.output.encoding.z_contextual_global.detach().cpu(),true);
      a.write("row_mean_huber",trace.objective.mean_huber.example_loss.detach().cpu(),true);a.write("row_variance_nll",trace.objective.variance_nll.example_loss.detach().cpu(),true);
      a.write("row_total_objective",(trace.objective.mean_huber.example_loss+trace.objective.variance_nll.example_loss).detach().cpu(),true);
      a.write("mean_Huber_F64",trace.objective.mean_huber.loss.detach().cpu(),true);a.write("variance_NLL_F64",trace.objective.variance_nll.loss.detach().cpu(),true);
      a.write("total_objective_F64",trace.objective.total_loss.detach().cpu(),true);a.write("Gaussian_NLL_F64",trace.nll.loss.detach().cpu(),true);
      a.write("pre_clip_global_norm",torch::tensor(pre_clip_norm,torch::kFloat64),true);a.write("clip_factor",torch::tensor(clip_factor,torch::kFloat64),true);
      temporal_learning.witness(a,model,before);
      // Same pre-step encoder call, never a replay. R is the actual compacted
      // nonempty batch/channel-row count and T is the actual packed token width.
      for(const auto &value:{trace.temporal.pool_local_weights,trace.temporal.contextual_pool_local_weights}){
        check(value.defined()&&value.is_cuda()&&value.scalar_type()==torch::kFloat32&&
            value.sizes()==torch::IntArrayRef({trace.temporal.patch_positions.size(0),4,trace.temporal.patch_positions.size(1)}),
            "same-forward exact per-window local softmax F32[R,4,T], never an aggregate or mixture inversion");
      }
      for(const auto &value:{trace.temporal.pool_bin_regularized_root,trace.temporal.contextual_pool_bin_regularized_root,
          trace.temporal.pool_bin_dispersion,trace.temporal.contextual_pool_bin_dispersion}){
        check(value.defined()&&value.is_cuda()&&value.scalar_type()==torch::kFloat32&&
            value.sizes()==torch::IntArrayRef({trace.temporal.patch_positions.size(0),4,64}),
            "same-forward actual F32 regularized-root/dispersion trace shape");
      }
      for(const auto &[key,value]:std::vector<std::pair<std::string,torch::Tensor>>{
          {"temporal_patch_positions",trace.temporal.patch_positions},
          {"temporal_patch_valid",trace.temporal.patch_valid},{"temporal_patch_rows",trace.temporal.patch_rows},
          {"temporal_hidden",trace.temporal.temporal_hidden},{"temporal_contextual_hidden",trace.temporal.contextual_temporal_hidden},
          {"temporal_pool_local_weights",trace.temporal.pool_local_weights},{"temporal_contextual_pool_local_weights",trace.temporal.contextual_pool_local_weights},
          {"temporal_pool_global_weights",trace.temporal.pool_global_weights},{"temporal_contextual_pool_global_weights",trace.temporal.contextual_pool_global_weights},
          {"temporal_pool_mixture_weights",trace.temporal.pool_mixture_weights},{"temporal_contextual_pool_mixture_weights",trace.temporal.contextual_pool_mixture_weights},
          {"temporal_pool_bin_valid",trace.temporal.pool_bin_valid},{"temporal_contextual_pool_bin_valid",trace.temporal.contextual_pool_bin_valid},
          {"temporal_pool_bin_mean",trace.temporal.pool_bin_mean},{"temporal_contextual_pool_bin_mean",trace.temporal.contextual_pool_bin_mean},
          {"temporal_pool_bin_variance",trace.temporal.pool_bin_variance},{"temporal_contextual_pool_bin_variance",trace.temporal.contextual_pool_bin_variance},
          {"temporal_pool_bin_regularized_root",trace.temporal.pool_bin_regularized_root},{"temporal_contextual_pool_bin_regularized_root",trace.temporal.contextual_pool_bin_regularized_root},
          {"temporal_pool_bin_dispersion",trace.temporal.pool_bin_dispersion},{"temporal_contextual_pool_bin_dispersion",trace.temporal.contextual_pool_bin_dispersion}}){
        check(value.defined()&&torch::isfinite(value).all().item<bool>(),"finite same-forward temporal moment/support evidence");
        a.write(key,value.detach().cpu(),true);
      }
      // clip_grad_norm_ rescales parameter gradients only. Retained nonleaf
      // projection-output adjoints are PRECLIP and are scaled once by the
      // recorded factor during independent local-moment adjoint verification.
      for(const auto &[prefix,projected]:std::vector<std::pair<std::string,torch::Tensor>>{
          {"temporal_pool_projected",trace.temporal.pool_projected},
          {"temporal_contextual_pool_projected",trace.temporal.contextual_pool_projected}}){
        check(projected.defined(),"same-forward projected trace exists");const bool defined=projected.grad().defined();
        const auto upstream=defined?projected.grad():torch::zeros_like(projected);
        check(torch::isfinite(upstream).all().item<bool>(),"finite actual preclip upstream or explicit unused-route zero");
        a.write(prefix+"_upstream_defined",torch::tensor(int64_t(defined),torch::kInt64),true);
        a.write(prefix+"_upstream_pre_clip",upstream.detach().cpu(),true);
      }
      publish(a,args.output/"witnesses"/("training-update-"+std::to_string(progress.completed_updates)+"-components.pt"));
    }
    checkpoint_seconds+=elapsed(stage);
    optimizer->zero_grad(true);
    if(progress.completed_updates%128==0){frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);std::cout<<"update="<<progress.completed_updates<<" attempts="<<progress.attempted_batches<<" loss="<<last_loss<<std::endl;}
    if(progress.completed_updates==512||progress.completed_updates==1024||progress.completed_updates==2048){
      stage=Clock::now();check(model->causal_updates()==progress.completed_updates,"live counter agrees before typed training save");log.flush();const auto path=args.output/"checkpoints"/("training-"+std::to_string(progress.completed_updates)+".pt");tr::save_training_state(path.string(),model,scaler,*optimizer,identity,progress);
      if(progress.completed_updates==512){const ResumeSnapshot original(model,*optimizer);auto resumed=tr::load_training_state(path.string(),config,identity);original.compare(ResumeSnapshot(resumed.model,*resumed.optimizer));
        check(resumed.progress.completed_updates==progress.completed_updates&&resumed.progress.attempted_batches==progress.attempted_batches&&resumed.progress.row_seed==row_seed&&resumed.progress.mask_seed==mask_seed&&resumed.progress.context_seed==context_seed,"exact typed resume counters/seed streams");
        model=resumed.model;scaler=resumed.scaler;optimizer=std::move(resumed.optimizer);progress=resumed.progress;check(model->causal_updates()==progress.completed_updates,"resumed live counter agrees with saved progress");model->train();frozen.validate(model,scaler);resume_verified=true;
        write(args.output/"typed-resume-512.json","{\"status\":\"passed\","+objective_json_fields()+",\"gradient_policy\":"+quote(cp::kGradientPolicy)+",\"completed_updates\":512,\"same_parameters_buffers_moments_steps_order\":true,\"same_attempt_counter_and_three_seeds\":true,\"continued_actual_trajectory_from_saved_state\":true,\"repeated_training_updates\":0}");}
      checkpoint_seconds+=elapsed(stage);
    }
  }
  torch::cuda::synchronize();const auto trajectory_seconds=elapsed(trajectory_start);log.close();check(bool(log),"persist training log");check(progress.completed_updates==2048&&resume_verified,"complete one declared causal trajectory with actual typed resume");auto evidence_finish=Clock::now();temporal_learning.finish(model,args.output);evidence_seconds+=elapsed(evidence_finish);optimizer->zero_grad(true);frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);
  check(model->causal_updates()==progress.completed_updates&&progress.completed_updates==2048,"final live counter agrees before inference save and query evidence");
  io_start=Clock::now();model->eval();model->clear_gradients();cp::save_checkpoint((args.output/"checkpoints/inference-2048.pt").string(),model,scaler,2048);checkpoint_seconds+=elapsed(io_start);
  const auto final_train=capture(data,model,scaler,selections.probe_train,query_train,md::train_begin,md::train_end,args.output/"witnesses/final-train-query.pt");
  const auto final_validation=capture(data,model,scaler,selections.probe_validation,query_validation,md::validation_begin,md::validation_end,args.output/"witnesses/final-validation-query.pt");
  check(model->causal_updates()==progress.completed_updates,"final query evidence retains exact live update counter");write(args.output/"query-final.json","{\"train\":"+query_json(final_train)+",\"validation\":"+query_json(final_validation)+"}");frozen.validate(model,scaler);verify(pins);
  struct Group{int64_t tensors{0},values{0},gradient_tensors{0},moved_tensors{0},changed_final_tensors{0};};std::map<std::string,Group> groups;int64_t total_values=0,updated_values=0,fixed_values=0;std::ostringstream parameter_json;parameter_json<<'[';bool first=true;
  for(const auto &[name,e]:evidence){const auto current=model->named_parameters()[name];const bool changed=!cp::boundary::same_bits(current,e.initial);total_values+=e.values;if(e.trainable){auto &g=groups[category(name)];++g.tensors;g.values+=e.values;g.gradient_tensors+=e.gradient_steps>0;g.moved_tensors+=e.moved_steps>0;g.changed_final_tensors+=changed;if(e.moved_steps)updated_values+=e.values;}else{fixed_values+=e.values;check(!changed&&e.gradient_steps==0&&e.moved_steps==0,"fixed prior never learns or drifts");}
    if(!first)parameter_json<<',';first=false;parameter_json<<"{\"name\":"<<quote(name)<<",\"group\":"<<quote(category(name))<<",\"values\":"<<e.values<<",\"requires_grad\":"<<(e.trainable?"true":"false")<<",\"data_gradient_requirement_exempt\":"<<(name==pool_gauge?"true":"false")<<",\"nonzero_CUDA_gradient_steps\":"<<e.gradient_steps<<",\"weight_movement_steps\":"<<e.moved_steps<<",\"changed_from_initial\":"<<(changed?"true":"false")<<'}';}
  parameter_json<<']';write(args.output/"parameter-learning.json",parameter_json.str());
  int64_t identifiable_tensors=0,identifiable_values=0;for(const auto &[name,e]:evidence)if(e.trainable&&name!=pool_gauge){check(e.gradient_steps>0&&e.moved_steps>0,"each mandatory identifiable tensor has actual nonzero CUDA loss gradients and movement: "+name);++identifiable_tensors;identifiable_values+=e.values;}
  check(total_values==298640&&updated_values>=294751&&updated_values<=294752&&fixed_values==3888&&identifiable_tensors==75&&identifiable_values==294751,"fresh registered/fixed counts and75 identifiable tensor data-learning proof; scalar pooling gauge observed separately");
  std::ostringstream group_json;group_json<<'[';first=true;for(const auto &[name,g]:groups){if(!first)group_json<<',';first=false;group_json<<"{\"group\":"<<quote(name)<<",\"tensors\":"<<g.tensors<<",\"values\":"<<g.values<<",\"nonzero_gradient_tensors\":"<<g.gradient_tensors<<",\"moved_tensors\":"<<g.moved_tensors<<",\"changed_final_tensors\":"<<g.changed_final_tensors<<'}';}
  group_json<<']';
  for(const auto &required:{"parent_backbone_projection","parent_content_relation","parent_decoder","parent_global_pool","change_projection","change_wave_decoder","uncertainty_projection","variance_matrix","variance_bias"})check(groups.count(required)&&groups.at(required).gradient_tensors>0&&groups.at(required).moved_tensors>0,"required model category actually learns on CUDA: "+std::string(required));
  std::ostringstream component_witnesses;component_witnesses<<'[';bool first_component=true;
  for(const auto update:{1,2048}){const auto path=args.output/"witnesses"/("training-update-"+std::to_string(update)+"-components.pt");if(!first_component)component_witnesses<<',';first_component=false;
    component_witnesses<<"{\"update\":"<<update<<",\"pre_step_forward\":true,\"archive_fields\":60,\"temporal_trace_fields\":25,\"retained_projection_upstream_stage\":\"preclip-nonleaf;apply-recorded-clip-factor-once-for-postclip-local-parameter-adjoints\",\"extra_encoder_calls\":0,\"path\":"<<quote(path.string())<<",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":"<<quote(cp::boundary::sha256_file(path))<<'}';}
  component_witnesses<<']';write(args.output/"optimization-component-witnesses.json",component_witnesses.str());
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"kind\":\"causal_fresh_v14_training_summary_v1\",\"status\":\"passed\",\"model\":\"CPB-v14.market-01\","<<objective_json_fields()<<",\"gradient_policy\":"<<quote(cp::kGradientPolicy)<<",\"encoder_gradient_kind\":\"Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1\",\"conditional_variance_head_trainable\":true,\"initialization_kind\":\"fresh-seeded-constructor-no-import-v1\",\"initialization_seed\":943001,\"source_updates\":0,\"new_causal_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches<<",\"batch_size\":8,\"TRAIN_windows\":4096,\"probe_TRAIN_windows\":112,\"probe_VALIDATION_windows\":32,\"native_width\":40,\"device\":\"cuda:0\",\"parameters\":"<<total_values<<",\"learned_tensor_value_capacity_with_recorded_movement_events\":"<<updated_values<<",\"recorded_movement_capacity_semantics\":\"sum of numel of learned tensors with at least one recorded movement event; not a count of individually moved scalar values\",\"fixed_prior_parameters\":"<<fixed_values<<",\"parameter_groups\":"<<group_json.str()<<",\"first_training_NLL\":"<<first_loss<<",\"last_training_NLL\":"<<last_loss<<",\"first_mean_Huber_F64\":"<<first_mean<<",\"last_mean_Huber_F64\":"<<last_mean<<",\"first_variance_NLL_F64\":"<<first_variance<<",\"last_variance_NLL_F64\":"<<last_variance<<",\"first_total_objective_F64\":"<<first_total<<",\"last_total_objective_F64\":"<<last_total<<",\"optimization_component_witnesses\":"<<component_witnesses.str()<<",\"temporal_pooling_learning\":{\"path\":"<<quote((args.output/"temporal-projection-learning.json").string())<<",\"bytes\":"<<fs::file_size(args.output/"temporal-projection-learning.json")<<",\"sha256\":"<<quote(cp::boundary::sha256_file(args.output/"temporal-projection-learning.json"))<<"},\"all_eight_temporal_moment_subgroups_learned\":true,\"all_four_temporal_export_row_groups_learned\":true,\"all_four_pool_position_contrasts_learned\":true,\"all_three_between_bin_position_directions_learned\":true,\"final_between_bin_position_projection_changed\":true,\"all_mandatory_identifiable_tensors_learned\":true,\"mandatory_identifiable_tensor_value_capacity\":294751,\"mandatory_identifiable_count_semantics\":\"tensor numel capacity, not every-scalar gradient/movement proof\",\"per_scalar_nonzero_gradient_proof_claimed\":false,\"backward_tensor\":\"objective.total_loss\",\"training_target_cells\":"<<trained_cells
    <<",\"GPU_step_seconds_synchronized\":"<<compute_seconds<<",\"GPU_step_timing_scope\":\"forward_at+F64_Huber+detached-error-variance-NLL+detached-reporting-NLL-copy+backward+joint-clip+AdamW;synchronized_wall;includes_in_step_contract_checks;excludes_loader_masks_checkpoints_evidence_logging\",\"loader_mask_seconds\":"<<loader_mask_seconds<<",\"checkpoint_resume_logging_seconds\":"<<checkpoint_seconds<<",\"parameter_gradient_evidence_seconds\":"<<evidence_seconds<<",\"training_trajectory_wall_seconds\":"<<trajectory_seconds
    <<",\"fixed_query_initial\":{\"train\":"<<query_json(initial_train)<<",\"validation\":"<<query_json(initial_validation)<<"},\"fixed_query_final\":{\"train\":"<<query_json(final_train)<<",\"validation\":"<<query_json(final_validation)<<"},\"raw_load_calls\":"<<data.calls<<",\"direct_F64_record_cells_checked\":"<<data.record_cells<<",\"genuine_F64_differs_from_F32_cells\":"<<data.genuine_f64_cells<<",\"typed_resume_512_exact\":true,\"fixed_prior_scaler_normalizer_unchanged\":true,\"new_scaler_fits\":1,\"new_normalizer_fits\":1,\"calibration_encoder_calls\":0,\"calibration_validation_rows\":0,\"calibration_seconds\":"<<calibration.seconds<<",\"calibration_TRAIN_loader_calls\":"<<calibration.loader_calls<<",\"fresh_scaler_identity\":"<<quote(scaler.identity())<<",\"fresh_bank_identity\":"<<quote(model->normalizer().identity())<<",\"old_model_artifacts_opened\":0,\"old_normalization_artifacts_opened\":0,\"saved_quality_features_reused\":false,\"fixed_query_source\":\"unchanged saved masks; source seed942005; query943005 not consumed\",\"CPU_encoder_calls\":0,\"old_encoder_calls\":0,\"head_fits\":0,\"training_replays\":0}";
  write(args.output/"training-summary.json",summary.str());std::cout<<"CPB_V14_TRAINING="<<args.output<<'\n';return 0;
}
} // namespace
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
