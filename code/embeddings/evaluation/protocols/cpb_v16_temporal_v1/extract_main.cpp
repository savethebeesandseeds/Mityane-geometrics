// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Application data adapter only. Encoder implementation is the independent CPB family.
#include "include/model_binding.h"
#include "cuwacunu/data/record_types.h"
#include <torch/cuda.h>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::CPB_OBJECTIVE_VERSION;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
using Record=cuwacunu::data::kline_cache_t;
constexpr std::array<int64_t,3> masters{941101,941102,941103};
static_assert(sizeof(Record)==88 && std::endian::native==std::endian::little && std::is_trivially_copyable_v<Record>);
void require(bool yes,const std::string &why){TORCH_CHECK(yes,"[CPB-v4 extraction] ",why);}
std::string quote(const std::string &v){std::ostringstream o;o<<'"';for(unsigned char c:v){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c=='\n')o<<"\\n";else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
void write_new(const fs::path &p,const std::string &text){require(!fs::exists(p),"exclusive new text");std::ofstream out(p,std::ios::binary);require(bool(out),"create text");out<<text;out.close();require(bool(out),"save text");}
void publish(torch::serialize::OutputArchive &a,const fs::path &p){require(!fs::exists(p),"exclusive new archive");a.save_to(p.string());}
double elapsed(const Clock::time_point &s){return std::chrono::duration<double>(Clock::now()-s).count();}
struct TensorState{std::string name;torch::Tensor value;bool requires_grad;torch::Device device;};
struct Guard {
  std::vector<TensorState> parameters,buffers;cp::FrozenScaler scaler;decltype(std::declval<cp::Encoder>()->normalizer().clone()) normalizer;
  explicit Guard(const cp::Encoder &m,const cp::FrozenScaler &s):scaler{s.mean.clone(),s.scale.clone(),s.count.clone(),s.channel_ids.clone(),s.scale_floor,s.floor_applied.clone()},normalizer(m->normalizer().clone()){
    for(const auto &p:m->named_parameters()){require(p.value().is_cuda()&&!p.value().grad().defined(),"actual CUDA parameters without gradients");parameters.push_back({p.key(),p.value().detach().cpu().clone(),p.value().requires_grad(),p.value().device()});}
    for(const auto &p:m->named_buffers())buffers.push_back({p.key(),p.value().detach().cpu().clone(),p.value().requires_grad(),p.value().device()});
    require(!parameters.empty(),"nonempty independent model");
  }
  void check(const cp::Encoder &m,const cp::FrozenScaler &s)const{
    require(!m->is_training(),"eval mode");
    const auto compare=[](const auto &now,const auto &before){require(now.size()==before.size(),"state count");for(const auto &p:before){const auto *v=now.find(p.name);require(v && v->sizes()==p.value.sizes() && v->scalar_type()==p.value.scalar_type() && v->device()==p.device && v->requires_grad()==p.requires_grad && !v->grad().defined() && torch::equal(v->detach().cpu(),p.value),"immutable model state: "+p.name);}};
    compare(m->named_parameters(),parameters);compare(m->named_buffers(),buffers);
    const auto &n=m->normalizer();
    for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{n.counts,normalizer.counts},{n.sum,normalizer.sum},{n.sumsq,normalizer.sumsq},{n.mean,normalizer.mean},{n.scale,normalizer.scale},{n.support,normalizer.support}})
      require(a.device()==b.device()&&a.scalar_type()==b.scalar_type()&&torch::equal(a,b)&&!a.requires_grad()&&!a.grad().defined(),"immutable unregistered bank state");
    require(n.identity()==normalizer.identity() && n.source_identity==normalizer.source_identity && n.input_scaler_identity==normalizer.input_scaler_identity && n.input_geometry==normalizer.input_geometry && n.sampling_interval==normalizer.sampling_interval && n.floor==normalizer.floor && n.minimum_support==normalizer.minimum_support && n.frozen==normalizer.frozen && n.bank_family==normalizer.bank_family && n.raw_contract==normalizer.raw_contract && n.increment_recipe==normalizer.increment_recipe && n.layout_identity==normalizer.layout_identity && n.recipe_identity==normalizer.recipe_identity,"immutable normalizer metadata");
    require(s.identity()==scaler.identity()&&s.scale_floor==scaler.scale_floor&&torch::equal(s.mean,scaler.mean)&&torch::equal(s.scale,scaler.scale)&&torch::equal(s.count,scaler.count)&&torch::equal(s.channel_ids,scaler.channel_ids)&&torch::equal(s.floor_applied,scaler.floor_applied),"immutable checkpoint scaler");
  }
};
struct Channel {
  std::vector<char> bytes;int64_t first,last;
  explicit Channel(const fs::path &p){
    std::ifstream in(p,std::ios::binary|std::ios::ate);require(bool(in),"existing original raw cache");const auto size=in.tellg();require(size>0&&static_cast<int64_t>(size)%88==0,"complete original cache88");bytes.resize(static_cast<size_t>(size));in.seekg(0);in.read(bytes.data(),size);require(bool(in),"read original cache");
    first=at(0).close_time;last=at(rows()-1).close_time;for(size_t i=0;i<rows();++i)require(at(i).close_time==first+static_cast<int64_t>(i)*60000,"original exact minute grid");
  }
  size_t rows()const{return bytes.size()/88;}
  Record at(size_t i)const{require(i<rows(),"cache index");Record r;std::memcpy(&r,bytes.data()+i*88,88);return r;}
};
struct Capture {int64_t calls{0},rows{0},valid{0};double api_seconds{0};};
struct Encoded {torch::Tensor native,valid;};
Encoded encode(cp::Encoder &model,const cp::RawHistory &history,const torch::Tensor &cutoffs,const cp::FrozenScaler &scaler,Capture &capture){
  torch::cuda::synchronize();const auto start=Clock::now();const auto e=model->encode_at(history,cutoffs,scaler);torch::cuda::synchronize();capture.api_seconds+=elapsed(start);
  require(e.native.is_cuda()&&e.native.scalar_type()==torch::kFloat32&&!e.native.requires_grad()&&e.native.sizes()==torch::IntArrayRef({history.data.size(0),36})&&torch::isfinite(e.native).all().item<bool>(),"actual finite frozen CUDA native36");
  require(e.valid.is_cuda()&&e.valid.scalar_type()==torch::kBool&&e.valid.sizes()==torch::IntArrayRef({history.data.size(0)})&&torch::equal(e.cutoffs.cpu(),cutoffs),"exact cutoff-labelled sample validity");
  ++capture.calls;capture.rows+=history.data.size(0);capture.valid+=e.valid.sum().item<int64_t>();return{e.native.to(torch::kCPU,torch::kFloat64),e.valid.cpu()};
}
Capture market(const fs::path &shared,const std::string &role,const fs::path &output,std::vector<Channel> &channels,cp::Encoder &model,const cp::FrozenScaler &scaler,const Guard &guard,int64_t &raw_cells_checked){
  torch::serialize::InputArchive target;target.load_from((shared/"features"/(role+"-targets.pt")).string(),torch::kCPU);
  torch::Tensor ids;target.read("row_ids",ids,true);
  const auto count=role=="train"?112:32;require(ids.scalar_type()==torch::kInt64&&ids.sizes()==torch::IntArrayRef({count}),"exact saved raw-data target row IDs");
  const int64_t begin=role=="train"?1704067259999LL:1705881659999LL,end=role=="train"?1705881599999LL:1706486399999LL;
  std::vector<torch::Tensor> values,valid;Capture capture;
  for(int64_t offset=0;offset<count;offset+=8){
    auto raw=torch::zeros({8,3,128,9},torch::kFloat64),mask=torch::zeros({8,3,128,9},torch::kBool),times=torch::zeros({8,128},torch::kFloat64);auto x=raw.accessor<double,4>();auto m=mask.accessor<bool,4>();auto t=times.accessor<double,2>();
    for(int64_t b=0;b<8;++b){const auto anchor=ids[offset+b].item<int64_t>();require(anchor<=end&&anchor-127*60000>=begin,"role-contained H128 past window");for(int64_t h=0;h<128;++h){t[b][h]=static_cast<double>(anchor-(127-h)*60000);for(int64_t c=0;c<3;++c){const auto &file=channels[c];require(anchor>=file.first&&anchor<=file.last&&(anchor-file.first)%60000==0,"exact original anchor");const auto terminal=(anchor-file.first)/60000;require(terminal>=127,"available past");const auto r=file.at(static_cast<size_t>(terminal-127+h));require(r.close_time==static_cast<int64_t>(t[b][h]),"actual record close timestamp");const std::array<double,9> f{r.open_price,r.high_price,r.low_price,r.close_price,r.volume,r.quote_asset_volume,r.number_of_trades,r.taker_buy_base_volume,r.taker_buy_quote_volume};for(int64_t j=0;j<9;++j){const bool visible=r.open_time!=INT64_MIN&&std::isfinite(f[j]);x[b][c][h][j]=visible?f[j]:0.;m[b][c][h][j]=visible;}}}}
    // Raw cache doubles enter the boundary directly; no previous model/scaler bank is opened.
    require(raw.scalar_type()==torch::kFloat64&&mask.scalar_type()==torch::kBool,"genuine raw F64/mask path");raw_cells_checked+=raw.numel();
    const cp::RawHistory history{raw,mask,torch::tensor({0,1,2},torch::kInt64),times,60000.};const auto e=encode(model,history,times.select(1,127),scaler,capture);values.push_back(e.native);valid.push_back(e.valid);
  }
  guard.check(model,scaler);torch::serialize::OutputArchive bank;bank.write("values",torch::cat(values),true);bank.write("valid",torch::cat(valid),true);bank.write("row_ids",ids,true);publish(bank,output/("cpb-objective-"+role+".pt"));return capture;
}
void copy_labels(const fs::path &file,const std::string &role,int64_t rows,torch::serialize::OutputArchive &out){
  torch::serialize::InputArchive a;a.load_from(file.string(),torch::kCPU);for(const auto &key:{"labels_scoring_only","source_ids","source_ids_json"}){torch::Tensor v;a.read(key,v,true);require(v.device().is_cpu()&&((std::string(key)=="source_ids_json"&&v.scalar_type()==torch::kUInt8)||(v.scalar_type()==torch::kInt64&&v.sizes()==torch::IntArrayRef({rows}))),"scoring-only metadata schema");const auto suffix=std::string(key)=="labels_scoring_only"?"labels":key;out.write(role+"_"+suffix,v,true);}
}
Capture synthetic_role(const fs::path &file,const std::string &role,int64_t rows,cp::Encoder &model,const cp::FrozenScaler &scaler,torch::serialize::OutputArchive &out){
  torch::serialize::InputArchive a;a.load_from(file.string(),torch::kCPU);torch::Tensor raw,mask;a.read("observed",raw,true);a.read("feature_mask",mask,true);require(raw.device().is_cpu()&&raw.scalar_type()==torch::kFloat64&&raw.sizes()==torch::IntArrayRef({rows,3,128,9})&&mask.device().is_cpu()&&mask.scalar_type()==torch::kBool&&mask.sizes()==raw.sizes(),"unchanged genuine F64 TEMPO5 values/masks");
  std::vector<torch::Tensor> values,valid;Capture capture;const auto times=torch::arange(128,torch::kFloat64)*60000.;
  for(int64_t offset=0;offset<rows;offset+=8){cp::RawHistory history{raw.narrow(0,offset,8),mask.narrow(0,offset,8),torch::tensor({0,1,2},torch::kInt64),times,60000.};const auto e=encode(model,history,torch::full({8},7620000.,torch::kFloat64),scaler,capture);values.push_back(e.native);valid.push_back(e.valid);}
  out.write(role+"_values",torch::cat(values),true);out.write(role+"_valid",torch::cat(valid),true);copy_labels(file,role,rows,out);return capture;
}
std::string capture_json(const Capture &c){std::ostringstream o;o<<std::setprecision(17)<<"{\"full_CUDA_encode_calls\":"<<c.calls<<",\"rows\":"<<c.rows<<",\"valid\":"<<c.valid<<",\"full_API_seconds\":"<<c.api_seconds<<'}';return o.str();}
cp::FrozenScaler fit_synthetic_scaler(const fs::path &training,const cp::Config &config){
  torch::serialize::InputArchive archive;archive.load_from(training.string(),torch::kCPU);torch::Tensor raw,mask;
  archive.read("observed",raw,true);archive.read("feature_mask",mask,true);
  require(raw.device().is_cpu()&&raw.scalar_type()==torch::kFloat64&&raw.sizes()==torch::IntArrayRef({512,3,128,9})&&
          mask.device().is_cpu()&&mask.scalar_type()==torch::kBool&&mask.sizes()==raw.sizes(),"fresh scaler uses only raw synthetic TRAIN values/masks");
  require(torch::isfinite(raw.masked_select(mask)).all().item<bool>(),"finite observed synthetic TRAIN values");
  const auto rounded=torch::where(mask,raw,torch::zeros_like(raw)).to(torch::kFloat32);
  require(torch::isfinite(rounded.masked_select(mask)).all().item<bool>(),"finite declared F32 parent rounding");
  const cp::implementation::Input input{rounded,mask,torch::tensor({0,1,2},torch::kInt64),torch::full({512},7620000.,torch::kFloat64),60000.};
  auto scaler=cp::implementation::fit_configurable_spectral_scaler(input,config.level);
  require(scaler.mean.device().is_cpu()&&scaler.mean.scalar_type()==torch::kFloat64&&scaler.count.gt(0).all().item<bool>(),"fresh supported CPUF64 TRAIN-only synthetic statistics");
  return scaler;
}
int run(int argc,char **argv){
  torch::set_num_threads(1);torch::set_num_interop_threads(1);require(argc==19,"extract requires initial/final checkpoints, settings, output, shared, synthetic-inputs and ordered cache-btc/cache-eth/cache-sol");std::map<std::string,fs::path> args;for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"unique argument");for(const auto &key:{"--initial-checkpoint","--final-checkpoint","--settings","--output","--shared","--synthetic-inputs","--cache-btc","--cache-eth","--cache-sol"})require(args.count(key),"required argument");
  require(torch::cuda::is_available(),"existing CUDA device required");torch::NoGradGuard no_grad;const auto config=cp::read_settings(args.at("--settings").string());const auto &c=config.level;
  require(c.channel_count==3&&c.history_length==128&&c.input_width==9&&c.patch_length==8&&config.change_width==4&&c.device==torch::Device(torch::kCUDA,0)&&c.sampling_interval==60000.,"declared causal geometry");
  const auto output=args.at("--output");require(!fs::exists(output)&&fs::create_directories(output),"exclusive new extraction leaf");fs::create_directory(output/"synthetic-scalers");
  const auto shared=fs::canonical(args.at("--shared")),inputs=fs::canonical(args.at("--synthetic-inputs"));
  std::vector<Channel> channels;for(const std::string key:{"--cache-btc","--cache-eth","--cache-sol"})channels.emplace_back(args.at(key));
  const auto all_started=Clock::now();const auto fit_started=Clock::now();std::vector<cp::FrozenScaler> synthetic_scalers;
  for(const auto master:masters){auto scaler=fit_synthetic_scaler(inputs/("master-"+std::to_string(master))/"training.pt",config);torch::serialize::OutputArchive archive;scaler.save(archive);publish(archive,output/"synthetic-scalers"/("master-"+std::to_string(master)+".pt"));synthetic_scalers.push_back(std::move(scaler));}
  const auto synthetic_fit_seconds=elapsed(fit_started);int64_t total_calls=0;
  std::string market_scaler_identity,bank_identity;
  for(const std::string stage:{"initial","final"}){
  const int64_t updates=stage=="initial"?0:2048;const auto checkpoint=args.at(stage=="initial"?"--initial-checkpoint":"--final-checkpoint");
  auto loaded=cp::load_checkpoint(checkpoint.string(),c.device,&config);require(loaded.source_updates==0&&loaded.causal_updates==updates,"only fresh initial0/final2048 checkpoints may be quality-exported");loaded.model->validate_frozen();const Guard guard(loaded.model,loaded.scaler);
  if(stage=="initial"){market_scaler_identity=loaded.scaler.identity();bank_identity=loaded.model->normalizer().identity();}
  else require(loaded.scaler.identity()==market_scaler_identity&&loaded.model->normalizer().identity()==bank_identity,"same newly fitted frozen market normalization in initial/final");
  const auto directory=output/stage;fs::create_directory(directory);fs::create_directory(directory/"market");fs::create_directory(directory/"synthetic");
  const auto started=Clock::now();int64_t raw_cells_checked=0;const auto train=market(shared,"train",directory/"market",channels,loaded.model,loaded.scaler,guard,raw_cells_checked);const auto validation=market(shared,"validation",directory/"market",channels,loaded.model,loaded.scaler,guard,raw_cells_checked);
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"protocol\":\"cpb-v16-temporal\",\"model\":\"CPB-v16-temporal-paired\",\"checkpoint_stage\":"<<quote(stage)<<",\"source_updates\":0,\"causal_updates\":"<<updates<<",\"checkpoint\":"<<quote(fs::canonical(checkpoint).string())<<",\"device\":\"cuda:0\",\"batch_size\":8,\"native_width\":36,\"encoder_updates\":0,\"decoder_calls\":0,\"old_encoder_calls\":0,\"new_input_scaler_fits\":0,\"new_bank_normalizer_fits\":0,\"market_scaler_identity\":"<<quote(market_scaler_identity)<<",\"bank_normalizer_identity\":"<<quote(bank_identity)<<",\"market\":{\"train\":"<<capture_json(train)<<",\"validation\":"<<capture_json(validation)<<",\"raw_F64_cells_loaded\":"<<raw_cells_checked<<"},\"synthetic\":[";
  bool first=true;int64_t synthetic_calls=0,synthetic_rows=0;
  for(const std::string mode:{"synthetic_train","market_scaler"})for(size_t index=0;index<masters.size();++index){
    const auto master=masters[index];const auto bank=inputs/("master-"+std::to_string(master)),cohort_output=directory/"synthetic"/mode/("master-"+std::to_string(master));fs::create_directories(cohort_output);cp::FrozenScaler scaler=mode=="synthetic_train"?synthetic_scalers[index]:loaded.scaler;
    const auto scaler_identity=scaler.identity();torch::serialize::OutputArchive archive;std::vector<Capture> captures;const std::array<const char*,3> roles{"training","validation","deleted"},files{"training.pt","validation.pt","validation-deleted.pt"};
    for(size_t j=0;j<roles.size();++j){captures.push_back(synthetic_role(bank/files[j],roles[j],j==0?512:256,loaded.model,scaler,archive));guard.check(loaded.model,loaded.scaler);require(scaler.identity()==scaler_identity,"external scaler unchanged");}
    publish(archive,cohort_output/"features.pt");if(!first){summary<<',';}first=false;summary<<"{\"mode\":"<<quote(mode)<<",\"master\":"<<master<<",\"scaler_identity\":"<<quote(scaler_identity)<<",\"roles\":[";
    for(size_t j=0;j<captures.size();++j){if(j){summary<<',';}summary<<"{\"role\":"<<quote(roles[j])<<",\"capture\":"<<capture_json(captures[j])<<'}';synthetic_calls+=captures[j].calls;synthetic_rows+=captures[j].rows;}summary<<"]}";
  }
  guard.check(loaded.model,loaded.scaler);require(train.calls==14&&validation.calls==4&&synthetic_calls==768&&synthetic_rows==6144,"declared one-pass extraction counts");
  const auto calls=train.calls+validation.calls+synthetic_calls;total_calls+=calls;
  summary<<"],\"full_CUDA_encode_calls\":"<<calls<<",\"synthetic_rows\":"<<synthetic_rows<<",\"all_parameters_buffers_normalizer_scaler_unchanged\":true,\"all_gradients_undefined\":true,\"no_labels_in_encoder\":true,\"all_outputs_at_last_observed_grid_timestamp\":true,\"elapsed_seconds_including_input_io_and_state_checks\":"<<elapsed(started)<<"}\n";write_new(directory/"summary.json",summary.str());
  }
  require(total_calls==1572,"exact initial/final extraction budget");
  std::ostringstream complete;complete<<std::setprecision(17)<<"{\"kind\":\"causal_fresh_v4_initial_final_extraction_v1\",\"status\":\"passed\",\"model\":\"CPB-v16-temporal-paired\",\"checkpoint_stages\":[\"initial\",\"final\"],\"full_CUDA_encode_calls\":1572,\"new_synthetic_TRAIN_input_scaler_fits\":3,\"new_market_input_scaler_fits\":0,\"new_bank_normalizer_fits\":0,\"old_encoder_calls\":0,\"old_model_or_scaler_archives_opened\":0,\"synthetic_scaler_fit_seconds\":"<<synthetic_fit_seconds<<",\"elapsed_seconds_including_all_input_and_state_IO\":"<<elapsed(all_started)<<"}\n";
  write_new(output/"extraction-summary.json",complete.str());std::cout<<"CPB_V4_FEATURES="<<output.string()<<'\n';return 0;
}
} // namespace
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
