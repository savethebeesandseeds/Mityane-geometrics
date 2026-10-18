// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/visible_difference_adapter.h"
#include "embedding/shared/paired_pooling.h"
#include "embedding/shared/fixed_feature_readouts.h"
#include "embedding/shared/data.h"
#include "frozen_role_guard.h"
#include <torch/cuda.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <sys/stat.h>

#ifndef VISIBLE_DIFFERENCE_SOURCE_ID
#define VISIBLE_DIFFERENCE_SOURCE_ID "unrecorded"
#endif
namespace {
namespace fs=std::filesystem;
namespace ev=embedding::evaluation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
using namespace ev::frozen_inputs;
using Clock=std::chrono::steady_clock;
const std::array<uint64_t,5> masters{75272,76373,77474,78575,79676};
const std::string protocol="visible-difference-v1",method="native_visible_difference";
const std::string card_relative="code/evaluation/cards/visible_difference_v1.md";
const std::string card_sha="75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc";
const std::string parent_core="6a8a526087a7cf14fdd9f9f4d0e35218349f766bc24c7a7c58f4f388681bb7b1";
const std::string parent_training="63eae8c8b907885f14a9848d5f203ad6aef09c281489c34bc0b07cf3c7cfd1de";
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::string plan(){return "{\"protocol\":\"visible-difference-v1\",\"dataset_id\":\"TEMPO-3\",\"recipe_id\":\"structured-hard-timing-v1\",\"task\":\"lag_sign\","
  "\"timing_master_seeds\":[75272,76373,77474,78575,79676],\"designed_complexity_level\":4,\"complexity_scale_max\":5,"
  "\"candidate_tag\":\"RPB-v13\",\"matched_control_tag\":\"RPB-v10.alt-05\",\"other_control_tag\":\"RPB-v7.alt-05\","
  "\"parent_inventory_sha256\":\"9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa\","
  "\"reused_payload_roles\":130,\"encoder_trajectories\":5,\"encoder_updates_each\":512,\"batch_size\":8,\"sampled_rows\":20480,\"attempt_limit\":1024,"
  "\"retained_points\":10,\"checkpoint_roles_per_point\":6,\"candidate_parameter_count\":228877,\"common_parameter_count\":225805,\"difference_parameter_count\":3072,"
  "\"train_pairs\":128,\"validation_pairs\":64,\"shape\":[3,32,3],\"native_width\":32,\"temporal_difference_input\":1,"
  "\"head_repetitions\":[2701,2802,2903],\"planned_pipelines\":15,\"planned_heads\":30,\"helper_outer_train_fits\":5,"
  "\"initial_parity_exports\":15,\"unique_quality_native_exports\":15,\"full_native_export_calls\":30,\"query_writer_calls\":10,\"necessary_query_forwards\":40,"
  "\"baseline_or_control_refits\":0,\"parent_model_forwards\":0,\"generator_calls\":0,\"PCA_fits\":0,\"extra_decoder_calibration_updates\":0,"
  "\"selection\":false,\"promotion\":false,\"testing_accessed\":false,\"stress_accessed\":false,\"source_fingerprint\":"+quote(VISIBLE_DIFFERENCE_SOURCE_ID)+",\"human_card_sha256\":"+quote(card_sha)+"}";}
void exact(const torch::Tensor&a,const torch::Tensor&b,const std::string&why){
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  require(x.scalar_type()==y.scalar_type()&&x.sizes()==y.sizes()&&(!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0),why+" exact schema/bytes");
}
torch::Tensor read(torch::serialize::InputArchive&a,const std::string&key){torch::Tensor t;a.read(key,t,true);return t;}
// Retained TEMPO-3 controlled archives use observations/feature_mask, not the
// observed/labels keys in the older shared native-development loader.
ev::ControlledDataset load_retained_controlled(const fs::path&path,int64_t rows,uint64_t master,bool deleted=false){
  require(rows==256||rows==128,"fixed retained controlled row count");
  torch::serialize::InputArchive archive;archive.load_from(path.string(),torch::kCPU);
  const auto keys=archive.keys();const std::set<std::string> actual(keys.begin(),keys.end());
  auto expected=std::set<std::string>{"observations","feature_mask","labels_scoring_only","source_ids_json"};
  if(deleted)expected.insert("requested_erasure");
  require(actual==expected,"exact retained controlled archive keys");
  ev::ControlledDataset result;result.observed.data=read(archive,"observations");
  result.observed.feature_mask=read(archive,"feature_mask");result.labels=read(archive,"labels_scoring_only");
  const std::vector<int64_t> dimensions{rows,3,32,3};const torch::IntArrayRef geometry(dimensions);
  require(result.observed.data.device()==torch::kCPU&&result.observed.data.scalar_type()==torch::kFloat64&&
    result.observed.data.sizes()==geometry&&result.observed.data.is_contiguous(),"retained observations CPU F64 B3H32F3");
  require(result.observed.feature_mask.device()==torch::kCPU&&result.observed.feature_mask.scalar_type()==torch::kBool&&
    result.observed.feature_mask.sizes()==geometry&&result.observed.feature_mask.is_contiguous(),"retained feature mask CPU Bool B3H32F3");
  require(result.labels.device()==torch::kCPU&&result.labels.scalar_type()==torch::kInt64&&
    result.labels.sizes()==torch::IntArrayRef({rows})&&result.labels.is_contiguous(),"retained labels CPU Long B");
  require(torch::isfinite(result.observed.data).all().item<bool>()&&
    result.observed.data.masked_select(result.observed.feature_mask.logical_not()).eq(0).all().item<bool>(),"finite observations and exact hidden zeros");
  const auto ids=read(archive,"source_ids_json");
  require(ids.device()==torch::kCPU&&ids.scalar_type()==torch::kUInt8&&ids.dim()==1&&ids.is_contiguous(),"retained source JSON CPU UInt8 vector");
  const auto text=embedding::archive::tensor_text(ids);const std::regex quoted("\"([^\"\\\\]+)\"");
  for(auto it=std::sregex_iterator(text.begin(),text.end(),quoted);it!=std::sregex_iterator();++it)result.source_ids.push_back((*it)[1].str());
  require(result.source_ids.size()==size_t(rows)&&strings(result.source_ids)==text,"canonical retained source array and row count");
  const auto prefix="structured-hard-timing-comparison-v2/lag_sign/structured-hard-timing-v1/seed-"+std::to_string(master)+"/lag_sign/source-";
  int64_t previous_source=-1;
  for(int64_t i=0;i<rows;i+=2){
    require(result.source_ids[i].rfind(prefix,0)==0&&result.source_ids[i+1]==result.source_ids[i],"literal original paired source namespace");
    const auto suffix=result.source_ids[i].substr(prefix.size());
    require(!suffix.empty()&&suffix.find_first_not_of("0123456789")==std::string::npos,"canonical original source integer");
    const auto number=std::stoll(suffix);
    require(number>=0&&number<192&&number>previous_source&&std::to_string(number)==suffix,"ascending unique original192 source universe");previous_source=number;
    const auto first=result.labels[i].item<int64_t>(),second=result.labels[i+1].item<int64_t>();
    require((first==0&&second==1)||(first==1&&second==0),"randomized opposite original pair labels");
    require(torch::equal(result.observed.feature_mask[i],result.observed.feature_mask[i+1]),"paired retained feature masks");
  }
  if(deleted){const auto requested=read(archive,"requested_erasure");
    require(requested.device()==torch::kCPU&&requested.scalar_type()==torch::kBool&&requested.sizes()==geometry&&requested.is_contiguous(),"retained requested erasure CPU Bool B3H32F3");
    require(!requested.logical_and(result.observed.feature_mask).any().item<bool>(),"requested deleted coordinates remain hidden");
  }
  result.clean={result.observed.data.clone(),result.observed.feature_mask.clone()}; // legal placeholder only
  return result;
}
void retained_split_pairing(const ev::ControlledDataset&train,const ev::ControlledDataset&val,const ev::ControlledDataset&deleted){
  require(train.labels.size(0)==256&&val.labels.size(0)==128&&deleted.source_ids==val.source_ids,"fixed original split populations");exact(deleted.labels,val.labels,"same validation labels");
  const std::set<std::string> train_ids(train.source_ids.begin(),train.source_ids.end()),val_ids(val.source_ids.begin(),val.source_ids.end());
  auto universe=train_ids;universe.insert(val_ids.begin(),val_ids.end());
  require(train_ids.size()==128&&val_ids.size()==64&&universe.size()==192,"disjoint retained TRAIN/VAL covers original192 sources");
  require(!deleted.observed.feature_mask.logical_and(val.observed.feature_mask.logical_not()).any().item<bool>(),"retained deletion support subset");
}

void input_loader_test(const fs::path&output){
  require(output.is_absolute()&&fs::canonical(output.parent_path())==output.parent_path()&&
    !fs::exists(output)&&fs::create_directory(output),"exclusive artificial input-loader evidence");
  torch::set_num_threads(1);
  using Fields=std::map<std::string,torch::Tensor>;
  auto artificial=[](int64_t rows,bool deleted){Fields fields;
    fields["observations"]=torch::arange(rows*288,torch::TensorOptions().dtype(torch::kFloat64)).reshape({rows,3,32,3})*.001;
    fields["feature_mask"]=torch::ones({rows,3,32,3},torch::TensorOptions().dtype(torch::kBool));
    std::vector<int64_t> labels;std::vector<std::string> ids;for(int64_t i=0;i<rows;++i){labels.push_back((i/2)%2?1-i%2:i%2);
      const auto source=rows==128?3*(i/2)+2:i/2+(i/2)/2; // ascending, disjoint, gappy original192 universe
      ids.push_back("structured-hard-timing-comparison-v2/lag_sign/structured-hard-timing-v1/seed-75272/lag_sign/source-"+std::to_string(source));}
    fields["labels_scoring_only"]=torch::tensor(labels,torch::kInt64);
    fields["source_ids_json"]=embedding::archive::text_tensor(strings(ids));
    if(deleted)fields["requested_erasure"]=torch::zeros({rows,3,32,3},torch::TensorOptions().dtype(torch::kBool));
    return fields;};
  // These are engineering evidence, separate from the closed quality .pt role
  // matrix. InputArchive reads the same serialized format regardless of suffix.
  auto save=[&](const std::string&name,const Fields&fields){const auto path=output/(name+".serialized-archive");require(!fs::exists(path),"new artificial archive");
    torch::serialize::OutputArchive a;for(const auto&[key,value]:fields)a.write(key,value,true);a.save_to(path.string());return path;};
  std::map<std::pair<int64_t,bool>,ev::ControlledDataset> positives;
  for(const auto rows:{256,128})for(const auto deleted:{false,true}){auto fields=artificial(rows,deleted);
    if(deleted)for(const auto row:{0,1}){fields["requested_erasure"].index_put_({row,0,1,0},true);fields["feature_mask"].index_put_({row,0,1,0},false);fields["observations"].index_put_({row,0,1,0},0);}
    const auto loaded=load_retained_controlled(save("valid-"+std::to_string(rows)+(deleted?"-deleted":""),fields),rows,75272,deleted);
    exact(loaded.observed.data,fields.at("observations"),"serialized observations");exact(loaded.observed.feature_mask,fields.at("feature_mask"),"serialized feature mask");
    exact(loaded.labels,fields.at("labels_scoring_only"),"serialized labels");exact(loaded.clean.data,loaded.observed.data,"legal clean placeholder");
    positives.emplace(std::make_pair(rows,deleted),loaded);}
  retained_split_pairing(positives.at({256,false}),positives.at({128,false}),positives.at({128,true}));
  int negatives=0;
  auto reject=[&](const std::string&name,const std::function<void(Fields&)>&mutate,bool deleted=false){auto fields=artificial(128,deleted);mutate(fields);
    const auto path=save(name,fields);bool failed=false;try{(void)load_retained_controlled(path,128,75272,deleted);}catch(const std::exception&){failed=true;}
    require(failed,"artificial loader must reject "+name);++negatives;};
  reject("wrong-data-dtype",[](Fields&f){f["observations"]=f.at("observations").to(torch::kFloat32);});
  reject("wrong-data-shape",[](Fields&f){f["observations"]=f.at("observations").narrow(1,0,2).contiguous();});
  reject("wrong-mask-dtype",[](Fields&f){f["feature_mask"]=f.at("feature_mask").to(torch::kInt64);});
  reject("wrong-mask-shape",[](Fields&f){f["feature_mask"]=f.at("feature_mask").narrow(2,0,31).contiguous();});
  reject("wrong-label-dtype",[](Fields&f){f["labels_scoring_only"]=f.at("labels_scoring_only").to(torch::kFloat64);});
  reject("wrong-label-shape",[](Fields&f){f["labels_scoring_only"]=f.at("labels_scoring_only").unsqueeze(1);});
  reject("wrong-paired-label",[](Fields&f){f["labels_scoring_only"].index_put_({1},0);});
  reject("wrong-source-dtype",[](Fields&f){f["source_ids_json"]=f.at("source_ids_json").to(torch::kInt64);});
  reject("wrong-source-shape",[](Fields&f){f["source_ids_json"]=f.at("source_ids_json").unsqueeze(0);});
  reject("wrong-source-count",[](Fields&f){f["source_ids_json"]=embedding::archive::text_tensor("[\"wrong\"]");});
  reject("wrong-source-namespace",[](Fields&f){auto text=embedding::archive::tensor_text(f.at("source_ids_json"));text.replace(text.find("structured-hard"),15,"unrelated-hard");f["source_ids_json"]=embedding::archive::text_tensor(text);});
  reject("extra-key",[](Fields&f){f["observed"]=f.at("observations");});
  reject("missing-key",[](Fields&f){f.erase("observations");});
  reject("hidden-nonzero",[](Fields&f){f["feature_mask"].index_put_({0,0,1,0},false);f["feature_mask"].index_put_({1,0,1,0},false);});
  reject("nonfinite-observed",[](Fields&f){f["observations"].index_put_({0,0,0,0},std::numeric_limits<double>::quiet_NaN());});
  reject("mismatched-pair-mask",[](Fields&f){f["feature_mask"].index_put_({0,0,0,0},false);});
  reject("wrong-requested-dtype",[](Fields&f){f["requested_erasure"]=f.at("requested_erasure").to(torch::kInt64);},true);
  reject("wrong-requested-shape",[](Fields&f){f["requested_erasure"]=f.at("requested_erasure").narrow(2,0,31).contiguous();},true);
  reject("requested-still-visible",[](Fields&f){f["requested_erasure"].index_put_({0,0,0,0},true);},true);
  require(negatives==19,"all serialized input-loader negatives executed");
  std::cout<<"Visible difference retained input loader fixtures passed: 4 serialized positives; 19 negatives; no encoder construction, forward or fitting"<<'\n';
}
std::vector<std::string> split(const std::string&text,char delimiter){std::vector<std::string> out;std::string part;std::istringstream in(text);while(std::getline(in,part,delimiter))out.push_back(part);return out;}
std::vector<std::string> parent_roles(){std::vector<std::string> roles;
  for(const auto master:masters){const auto base="results/seed-"+std::to_string(master)+"-lag_sign/";
    for(const std::string name:{"controlled-training.pt","controlled-validation.pt","controlled-validation-deleted.pt","initial-pair.pt"})roles.push_back(base+name);
    for(const std::string suffix:{"",".audit.pt",".scaler.pt",".training-raw.pt",".structured-timing.pt"})roles.push_back(base+"early/point-0/checkpoint.pt"+suffix);
    for(const std::string view:{"training","validation-intact","validation-deleted"})roles.push_back(base+"lag_sign/readouts/untrained_early/"+view+"-features.pt");
    for(const std::string control:{"native_late","native_early"})for(const auto rep:{2701,2802,2903})for(const std::string view:{"validation-intact","validation-deleted"})
      roles.push_back(base+"lag_sign/readouts/"+control+"/rep-"+std::to_string(rep)+"/"+view+"-predictions.pt");
    for(const std::string view:{"training","validation"})roles.push_back(base+"early/point-512/"+view+"-reconstruction.pt");}
  std::sort(roles.begin(),roles.end());require(roles.size()==130&&std::set<std::string>(roles.begin(),roles.end()).size()==130,"closed retained130 names");return roles;
}
void whole_admit(const std::vector<fs::path>&paths){std::set<fs::path> names;std::set<std::pair<dev_t,ino_t>> nodes;
  for(const auto&p:paths){require(p.is_absolute()&&fs::canonical(p)==p&&fs::is_regular_file(p)&&!fs::is_symlink(p),"direct canonical regular input");
    for(auto node=p;!node.empty()&&node!=node.root_path();node=node.parent_path())require(!fs::is_symlink(node),"no input symlink ancestor");
    struct stat s{};require(::stat(p.c_str(),&s)==0&&s.st_nlink==1&&names.insert(p).second&&nodes.emplace(s.st_dev,s.st_ino).second,"whole nonhardlinked unique inode matrix before bytes");}}
std::string progress_json(const ev::CurveProgress&p){std::ostringstream s;s<<std::setprecision(17)<<"{\"attempted\":"<<p.attempted<<",\"completed\":"<<p.completed<<",\"sampled_rows\":"<<p.sampled_rows
  <<",\"parameter_count\":"<<p.parameter_count<<",\"cuda_parameter_count\":"<<p.cuda_parameter_count<<",\"training_seconds\":"<<p.training_seconds<<",\"training_device\":"<<quote(p.training_device)
  <<",\"training_dataset_id\":"<<quote(p.training_dataset_id)<<",\"preprocessing_id\":"<<quote(p.preprocessing_id)<<",\"last_input_cuda\":"<<(p.last_input_cuda?"true":"false")
  <<",\"last_loss_cuda\":"<<(p.last_loss_cuda?"true":"false")<<",\"finite_gradients\":"<<(p.finite_gradients?"true":"false")<<",\"weights_changed\":"<<(p.weights_changed?"true":"false")<<",\"losses\":[";
  bool first=true;for(const auto&l:p.losses){if(!first)s<<',';first=false;s<<'['<<l.attempted<<','<<l.completed<<','<<l.target_cells<<','<<l.loss<<','<<l.gradient_norm<<']';}return s.str()+"]}";}
struct ParentProgress{std::array<int64_t,6> counters;std::vector<int64_t> targets;};
std::map<uint64_t,ParentProgress> parent_progress(const fs::path&path){std::istringstream in(bytes(path));std::string line;
  require(std::getline(in,line)&&line=="master\tattempted\tcompleted\tsampled_rows\trequested\tactual\trestored\ttarget_cells","exact parent progress metadata header");std::map<uint64_t,ParentProgress> rows;
  while(std::getline(in,line)){const auto fields=split(line,'\t');require(fields.size()==8,"exact parent progress metadata fields");const auto master=std::stoull(fields[0]);ParentProgress p;
    for(size_t i=0;i<6;++i){require(!fields[i+1].empty()&&fields[i+1].find_first_not_of("0123456789")==std::string::npos,"unsigned parent counters");p.counters[i]=std::stoll(fields[i+1]);}
    for(const auto&v:split(fields[7],',')){require(!v.empty()&&v.find_first_not_of("0123456789")==std::string::npos,"closed parent target counts");p.targets.push_back(std::stoll(v));}
    require(p.counters[0]==512&&p.counters[1]==512&&p.counters[2]==4096&&p.targets.size()==512&&rows.emplace(master,p).second,"complete parent update prefix");}
  require(rows.size()==5,"all five parent progress records");for(auto m:masters)require(rows.count(m),"fixed parent master");return rows;}
void initial_feature_parity(const ev::FeatureSurface&current,const fs::path&saved,const ev::ControlledDataset&data){torch::serialize::InputArchive a;a.load_from(saved.string(),torch::kCPU);
  exact(current.values,read(a,"features"),"initial native F32");exact(current.valid,read(a,"valid"),"initial native support");exact(data.labels,read(a,"labels_scoring_only"),"initial scoring labels");
  require(embedding::archive::tensor_text(read(a,"source_ids_json"))==strings(data.source_ids),"initial source row order");}
void query_parity(const fs::path&candidate,const fs::path&parent){torch::serialize::InputArchive a,b;a.load_from(candidate.string(),torch::kCPU);b.load_from(parent.string(),torch::kCPU);
  for(const std::string key:{"standardized_target","target_mask","requested_observed_target_mask","visible_mask","trial_channel_eligible","channel_target_counts","channel_valid","example_valid","source_ids_json"})exact(read(a,key),read(b,key),"original query association "+key);}
std::string interval_json(const ev::GroupedInterval&v){std::ostringstream s;s<<std::setprecision(17)<<"{\"method\":\"source-group percentile bootstrap; within-master conditional on fitted readout\",\"replicates\":1000,\"confidence\":0.95,\"source_groups\":"<<v.source_groups<<",\"estimate\":";
  if(v.source_groups)s<<v.estimate;else s<<"null";s<<",\"lower\":";if(v.supported)s<<v.lower;else s<<"null";s<<",\"upper\":";if(v.supported)s<<v.upper;else s<<"null";return s.str()+"}";}
uint64_t named(const std::string&s){uint64_t h=14695981039346656037ULL;for(const unsigned char c:s){h^=c;h*=1099511628211ULL;}return h;}
std::string retained_pairs(const fs::path&readouts,const Guard&guard,const std::string&base,uint64_t master,const ev::ControlledDataset&val){std::ostringstream s;s<<'[';bool first=true;
  for(const std::string control:{"native_early","native_late"})for(const auto rep:{2701,2802,2903})for(const std::string view:{"validation-intact","validation-deleted"}){
    if(!first)s<<',';first=false;const auto repid="rep-"+std::to_string(rep);const std::string view_id=view=="validation-intact"?"validation_intact":"validation_deleted";const auto id=method+"_minus_"+control;
    const auto newpath=readouts/method/repid/(view+"-predictions.pt");s<<"{\"id\":"<<quote(id)<<",\"reference\":"<<quote(control)<<",\"candidate\":"<<quote(method)<<",\"repetition\":"<<quote(repid)<<",\"view\":"<<quote(view_id);
    if(!fs::exists(newpath)){s<<",\"status\":\"unsupported_fit\"}";continue;}
    torch::serialize::InputArchive a,b;a.load_from(newpath.string(),torch::kCPU);b.load_from(guard.bind(base+"lag_sign/readouts/"+control+"/"+repid+"/"+view+"-predictions.pt"),torch::kCPU);
    for(auto*p:{&a,&b}){exact(read(*p,"labels_scoring_only"),val.labels,"paired scoring labels");require(embedding::archive::tensor_text(read(*p,"source_ids_json"))==strings(val.source_ids),"paired source order");
      for(const std::string head:{"ridge","tiny_secondary"}){const auto logits=read(*p,head=="ridge"?"ridge_logits":"tiny_logits");exact(read(*p,head),logits.argmax(1),"saved own-logit class");}}
    const auto common=read(a,"valid").logical_and(read(b,"valid"));const auto seed=ev::stream_seed(master,named(id+"/"+repid+"/"+view_id));
    s<<",\"status\":"<<quote(common.any().item<bool>()?"measured":"unsupported_zero_common")<<",\"total_rows\":"<<val.labels.size(0)<<",\"common_valid_rows\":"<<common.sum().item<int64_t>()<<",\"bootstrap_seed_decimal\":"<<quote(std::to_string(seed));
    for(const std::string head:{"ridge","tiny_secondary"})s<<','<<quote(head)<<':'<<interval_json(ev::grouped_accuracy_interval(read(a,head),val.labels,common,val.source_ids,seed,1000,read(b,head)));s<<'}';}
  return s.str()+']';}
int64_t fit_count(const std::string&s,const std::string&key){const auto pos=s.rfind("\"fit_counts\"");require(pos!=std::string::npos,"shared fit counts");std::smatch match;const auto tail=s.substr(pos);
  require(std::regex_search(tail,match,std::regex("\""+key+"\":([0-9]+)")),"shared named fit count");return std::stoll(match[1]);}
} // namespace

int main(int argc,char**argv){try{
  if(argc==3&&std::string(argv[1])=="--input-loader-test"){input_loader_test(argv[2]);return 0;}
  if(argc==2&&std::string(argv[1])=="--source-id"){std::cout<<VISIBLE_DIFFERENCE_SOURCE_ID<<'\n';return 0;}
  if(argc==2&&std::string(argv[1])=="--plan"){std::cout<<plan()<<'\n';return 0;}
  std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2){require(i+1<argc&&args.emplace(argv[i],argv[i+1]).second,"unique options with values");}
  require(args.size()==5&&args.count("--output")&&args.count("--admission-log")&&args.count("--admission-sha256")&&args.count("--card")&&args.count("--card-sha256"),"five closed measured options");
  const fs::path output=args.at("--output"),capsule=output.parent_path();
  require(output.is_absolute()&&output.filename()=="results"&&capsule.parent_path()==fs::path("/embedding/output/runs/rpb-visible-difference")&&fs::canonical(capsule)==capsule&&capsule.filename().string().rfind("visible-difference-",0)==0,"exclusive named capsule");
  require(fs::path(args.at("--card"))==capsule/"source"/card_relative&&fs::path(args.at("--admission-log"))==capsule/"admission/build-and-tests.log","captured card/log paths");
  std::map<std::string,fs::path> allowed;for(const auto&name:parent_roles())allowed.emplace(name,capsule/"retained-inputs"/name);
  std::vector<fs::path> inputs{args.at("--card"),args.at("--admission-log"),capsule/"retained-inputs.sha256",capsule/"parent-progress.tsv"};for(const auto&[_,p]:allowed)inputs.push_back(p);whole_admit(inputs);
  require(is_sha(VISIBLE_DIFFERENCE_SOURCE_ID)&&args.at("--card-sha256")==card_sha&&sha256(bytes(args.at("--card")))==card_sha,"frozen card/current source");
  const auto log=bytes(args.at("--admission-log"));require(sha256(log)==args.at("--admission-sha256")&&log.find(VISIBLE_DIFFERENCE_SOURCE_ID)!=std::string::npos&&log.find("Visible difference model CUDA admission passed")!=std::string::npos&&
    log.find("Visible difference adapter CUDA admission passed")!=std::string::npos&&log.find("Visible difference retained input loader fixtures passed")!=std::string::npos&&log.find("Fixed feature readout tests passed")!=std::string::npos&&log.find("Frozen role guard checks passed")!=std::string::npos,"source-bound actual CUDA admission");
  Guard guard(capsule/"retained-inputs.sha256",std::move(allowed));const auto parent=parent_progress(capsule/"parent-progress.tsv");const auto progress_sha=sha256(bytes(capsule/"parent-progress.tsv"));
  require(torch::cuda::is_available()&&!fs::exists(output)&&fs::create_directory(output),"new output and CUDA required");torch::set_num_threads(1);write_new(output/"recipe-plan.json",plan()+"\n");write_new(output/"input-manifest.json",guard.json()+"\n");
  ev::NativeCurveRun recipe;recipe.card.shape={3,32,3,torch::kFloat64,torch::kCPU};recipe.card.channel_ids={0,1,2};recipe.card.feature_units="unitless,unitless,unitless";recipe.card.sampling_interval=1;
  recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.development_only=true;recipe.stress_sweep=false;
  auto settings=rpb::default_settings();settings.steps=512;settings.attempt_limit=1024;settings.log_every=1;settings.batch_size=8;settings.threads=1;settings.learning_rate=.001;settings.weight_decay=.0001;settings.gradient_clip_norm=1;
  settings.model.device=torch::kCUDA;settings.model.channel_mixer_layers=1;settings.model.channel_mixer_placement=1;settings.model.global_bottleneck_mode=2;settings.model.global_pool_input_source=0;settings.model.temporal_difference_input=1;settings.model.export_width=32;
  std::ostringstream report;report<<std::setprecision(17)<<"{\"protocol\":"<<quote(protocol)<<",\"dataset_id\":\"TEMPO-3\",\"recipe_id\":\"structured-hard-timing-v1\",\"task\":\"lag_sign\",\"source_fingerprint\":"<<quote(VISIBLE_DIFFERENCE_SOURCE_ID)<<",\"cohorts\":[";
  bool first=true;int64_t pipelines=0,outer_fits=0;
  for(auto master:masters){const auto base="results/seed-"+std::to_string(master)+"-lag_sign/";const auto root=output/("seed-"+std::to_string(master)+"-lag_sign");require(fs::create_directory(root)&&fs::create_directory(root/"candidate"),"new cohort/candidate");
    auto start=Clock::now();const auto train=load_retained_controlled(guard.bind(base+"controlled-training.pt"),256,master),val=load_retained_controlled(guard.bind(base+"controlled-validation.pt"),128,master),deleted=load_retained_controlled(guard.bind(base+"controlled-validation-deleted.pt"),128,master,true);
    retained_split_pairing(train,val,deleted);const double parent_io=elapsed(start);
    ev::ProviderFitInput fit{train.observed,recipe.card.shape,master,train.source_ids,recipe.card.channel_ids,recipe.card.feature_units,rpb::kVisibleDifferenceFitProtocol,1,31};
    rpb::VisibleDifferenceOptions options{true,guard.bind(base+"early/point-0/checkpoint.pt"),parent_core,parent_training,"early-mixer-reliability-v1/lag_sign"};
    start=Clock::now();auto trainer=rpb::make_visible_difference_trainer(settings,options)(fit);double binding=elapsed(start);
    auto snapshot=[&](int64_t point){return rpb::make_visible_difference_snapshot({(root/"candidate"/("point-"+std::to_string(point))/"checkpoint.pt").string(),point,rpb::VisibleDifferenceScope::quality,
      trainer.audit_fields.at("core_writer_source_fingerprint"),trainer.audit_fields.at("training_producer_source_fingerprint")},fit);};
    require(fs::create_directory(root/"candidate/point-0"),"new point0");start=Clock::now();trainer.save_checkpoint((root/"candidate/point-0/checkpoint.pt").string());auto initial=snapshot(0);
    require(fs::create_directory(root/"candidate/point-0/snapshot-assets"),"new initial assets");initial.features.save_assets((root/"candidate/point-0/snapshot-assets").string());binding+=elapsed(start);
    double initial_seconds=0;start=Clock::now();const std::array<ev::ControlledDataset const*,3> splits{&train,&val,&deleted};const std::array<std::string,3> views{"training","validation-intact","validation-deleted"};
    require(fs::create_directory(root/"initial-parity"),"new parity evidence");for(size_t v=0;v<3;++v){auto current=ev::extract_native_global(initial,splits[v]->observed,recipe);initial_feature_parity(current,guard.bind(base+"lag_sign/readouts/untrained_early/"+views[v]+"-features.pt"),*splits[v]);
      ev::save_native_feature_archive((root/"initial-parity"/(views[v]+"-features.pt")).string(),current,*splits[v],recipe);}initial_seconds+=elapsed(start);
    const auto p=trainer.train_to(512);require(p.attempted==512&&p.completed==512&&p.sampled_rows==4096&&p.parameter_count==228877&&p.cuda_parameter_count==228877&&p.last_input_cuda&&p.last_loss_cuda&&p.finite_gradients&&p.weights_changed&&p.losses.size()==512,"complete candidate CUDA update budget");
    for(size_t i=0;i<512;++i)require(p.losses[i].attempted==int64_t(i+1)&&p.losses[i].completed==int64_t(i+1)&&p.losses[i].target_cells==parent.at(master).targets[i],"paired actual target-cell prefix before head fitting");
    require(fs::create_directory(root/"candidate/point-512"),"new point512");start=Clock::now();trainer.save_checkpoint((root/"candidate/point-512/checkpoint.pt").string());auto trained=snapshot(512);
    require(fs::create_directory(root/"candidate/point-512/snapshot-assets"),"new trained assets");trained.features.save_assets((root/"candidate/point-512/snapshot-assets").string());binding+=elapsed(start);
    torch::serialize::InputArchive audit;audit.load_from((root/"candidate/point-512/checkpoint.pt.audit.pt").string(),torch::kCPU);size_t n=3;
    for(const std::string key:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"})require(read(audit,key).item<int64_t>()==parent.at(master).counters[n++],"paired cumulative erasure counts");
    write_new(root/"candidate/encoder-progress.json",progress_json(p)+"\n");write_new(root/"candidate/trainer-audit.json",fields(trainer.audit_fields)+"\n");
    start=Clock::now();const auto tq=ev::write_native_patch_reconstruction((root/"candidate/point-512/training-reconstruction.pt").string(),train,trained,recipe),vq=ev::write_native_patch_reconstruction((root/"candidate/point-512/validation-reconstruction.pt").string(),val,trained,recipe);
    query_parity(root/"candidate/point-512/training-reconstruction.pt",guard.bind(base+"early/point-512/training-reconstruction.pt"));query_parity(root/"candidate/point-512/validation-reconstruction.pt",guard.bind(base+"early/point-512/validation-reconstruction.pt"));const double query_seconds=elapsed(start);
    start=Clock::now();ev::FixedFeatureMethod m{method,ev::extract_native_global(trained,train.observed,recipe),ev::extract_native_global(trained,val.observed,recipe),ev::extract_native_global(trained,deleted.observed,recipe),false,{}};const double native_seconds=elapsed(start);
    require(fs::create_directory(root/"lag_sign"),"new task");ev::FixedFeatureReadoutRun heads;heads.output_directory=(root/"lag_sign/readouts").string();heads.master_seed=master;heads.training_labels=train.labels;heads.validation_labels=val.labels;
    heads.training_source_ids=train.source_ids;heads.validation_source_ids=val.source_ids;heads.methods.push_back(std::move(m));start=Clock::now();const auto scores=ev::run_fixed_feature_readouts(heads);
    const auto nf=fit_count(scores,"ridge_fits");require(nf==fit_count(scores,"tiny_fits")&&nf<=3,"one native method three fixed fits");pipelines+=nf;outer_fits+=fit_count(scores,"outer_train_normalizer_fits");
    const auto pairs=retained_pairs(root/"lag_sign/readouts",guard,base,master,val);write_new(root/"lag_sign/reused-paired-comparisons.json",pairs+"\n");const double head_seconds=elapsed(start);
    guard.verify();if(!first)report<<',';first=false;report<<"{\"timing_master\":"<<master<<",\"encoder_points\":[{\"model_tag\":\"RPB-v13\",\"placement\":1,\"temporal_difference_input\":1,\"encoder_progress\":"<<progress_json(p)
      <<",\"training_reconstruction\":"<<tq<<",\"validation_reconstruction\":"<<vq<<"}],\"tasks\":[{\"task\":\"lag_sign\",\"data_master\":"<<master<<",\"readouts\":"<<scores<<",\"reused_paired_comparisons\":"<<pairs
      <<"}],\"initial_parity\":{\"exports\":3,\"exact_saved_untrained_early_features\":true,\"before_heads\":true},\"costs\":{\"parent_archive_verification_and_load_seconds\":"<<parent_io<<",\"binding_checkpoint_and_snapshot_IO_seconds\":"<<binding
      <<",\"CUDA_initial_parity_transfer_and_verification_seconds\":"<<initial_seconds<<",\"CUDA_query_transfer_verification_IO_seconds\":"<<query_seconds<<",\"CUDA_quality_native_transfer_seconds\":"<<native_seconds<<",\"CPU_candidate_heads_and_paired_bootstrap_IO_seconds\":"<<head_seconds<<"}}";
    std::cout<<"RPB-v13 master="<<master<<" CUDA_updates=512 candidate_head_pipelines="<<nf<<'\n'<<std::flush;}
  guard.verify();require(sha256(bytes(capsule/"parent-progress.tsv"))==progress_sha,"parent progress metadata preserved");write_new(output/"report.json",report.str()+"],\"parent_summary_sha256\":\"1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4\",\"selection\":false,\"promotion\":false,\"testing_accessed\":false,\"stress_accessed\":false}\n");
  write_new(output/"complete.json","{\"protocol\":\"visible-difference-v1\",\"status\":\"complete\",\"cohorts\":5,\"encoder_trajectories\":5,\"encoder_updates_each\":512,\"skipped_attempts\":0,\"sampled_rows\":20480,\"retained_points\":10,\"checkpoint_roles_per_point\":6,\"retained_payload_roles\":130,\"head_pipelines\":"+std::to_string(pipelines)+",\"individual_heads\":"+std::to_string(2*pipelines)+",\"planned_pipelines\":15,\"planned_heads\":30,\"helper_outer_train_fits\":"+std::to_string(outer_fits)+",\"quality_native_exports\":15,\"initial_parity_exports\":15,\"full_native_export_calls\":30,\"query_writer_calls\":10,\"necessary_query_forwards\":40,\"baseline_or_control_refits\":0,\"parent_model_forwards\":0,\"generator_calls\":0,\"PCA_fits\":0,\"CPU_encoder_forward\":false,\"CPU_encoder_training\":false,\"extra_decoder_calibration_updates\":0,\"selection\":false,\"promotion\":false,\"testing_accessed\":false,\"stress_accessed\":false}\n");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
