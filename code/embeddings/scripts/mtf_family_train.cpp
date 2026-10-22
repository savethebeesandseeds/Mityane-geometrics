// SPDX-License-Identifier: MIT
// Portable family example. Application populations, admission and budgets remain caller-owned.
#include "embedding/encoders/mtf_jepa_mae_vicreg/family.h"
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;
namespace fam = mtf::family;
namespace fs = std::filesystem;
namespace {
using Args=std::map<std::string,std::string>;
std::string quoted(const std::string &s) {
  std::ostringstream out;out<<'"';for(unsigned char c:s) {
    if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";
    else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else {TORCH_CHECK(c>=32,"control character in identity");out<<c;}
  }out<<'"';return out.str();
}
Args arguments(int argc,char **argv,const std::set<std::string> &allowed) {
  Args out;for(int i=2;i<argc;i+=2) {
    const std::string key(argv[i]);TORCH_CHECK(allowed.count(key),"unknown option: ",key);
    TORCH_CHECK(i+1<argc&&out.emplace(key,argv[i+1]).second,"missing value or duplicate: ",key);
  }return out;
}
std::string required(const Args &a,const std::string &key) {
  const auto it=a.find(key);TORCH_CHECK(it!=a.end()&&!it->second.empty(),"required option: ",key);return it->second;
}
std::string optional(const Args &a,const std::string &key,const std::string &fallback) {
  const auto it=a.find(key);return it==a.end()?fallback:it->second;
}
int64_t integer(const std::string &text) {
  int64_t value=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
  TORCH_CHECK(parsed.ec==std::errc()&&parsed.ptr==text.data()+text.size(),"invalid integer: ",text);return value;
}
double real(const std::string &text) {
  size_t used=0;const double value=std::stod(text,&used);
  TORCH_CHECK(used==text.size()&&std::isfinite(value),"invalid real: ",text);return value;
}
bool boolean(const std::string &text) {
  TORCH_CHECK(text=="true"||text=="false","expected true/false");return text=="true";
}
void output_path(const std::string &output,const std::vector<std::string> &inputs) {
  TORCH_CHECK(!output.empty()&&!fs::exists(output),"output must be a new path: ",output);
  for(const auto &input:inputs) if(!input.empty())
    TORCH_CHECK(fs::weakly_canonical(output)!=fs::weakly_canonical(input),"output aliases input: ",input);
}
embedding::Batch read_raw(const std::string &path) {
  torch::serialize::InputArchive archive;archive.load_from(path,torch::kCPU);
  embedding::Batch raw;archive.read("data",raw.data,true);archive.read("feature_mask",raw.feature_mask,true);return raw;
}
embedding::Batch artificial(const mtf::Config &c,int64_t rows) {
  TORCH_CHECK(rows>0&&rows<=1024,"artificial example rows must be in [1,1024]");
  auto x=torch::empty({rows,c.channel_count,c.history_length,c.input_width},torch::kFloat64);
  auto mask=torch::ones(x.sizes(),torch::kBool);
  auto a=x.accessor<double,4>();auto m=mask.accessor<bool,4>();
  for(int64_t b=0;b<rows;++b)for(int64_t ch=0;ch<c.channel_count;++ch)
    for(int64_t t=0;t<c.history_length;++t)for(int64_t f=0;f<c.input_width;++f) {
      a[b][ch][t][f]=.1*b+.17*ch+.08*f+.03*t+std::sin(.27*t+.31*f+.13*ch+.22*b);
      m[b][ch][t][f]=(t+f+ch+b)%29!=0;
    }
  return{x,mask};
}
fam::Recipe recipe(const Args &args,const embedding::Batch &raw) {
  fam::Recipe r;r.settings=mtf::read_settings(required(args,"--config"));
  r.settings.steps=integer(required(args,"--budget"));
  r.source_sha256=required(args,"--source-sha256");r.input_source_identity=required(args,"--input-source");
  r.split_identity=required(args,"--split");r.recipe_identity=required(args,"--recipe");
  r.sampling=required(args,"--sampling");
  r.adam_beta1=real(optional(args,"--beta1","0.9"));r.adam_beta2=real(optional(args,"--beta2","0.999"));
  r.adam_epsilon=real(optional(args,"--epsilon","1e-8"));r.adam_amsgrad=boolean(optional(args,"--amsgrad","false"));
  r.dataset_sha256=fam::dataset_identity(fam::prepare(raw,r.settings.model));fam::validate_recipe(r);return r;
}
void usage() {
  std::cout<<R"(MTF portable CPU/F32/single-thread family example
  mtf_family_train synthetic --config FILE --output NEW.pt [--rows 5]
  mtf_family_train prepare --config FILE --input RAW.pt --output NEW.pt
  mtf_family_train train --config FILE --input RAW.pt --checkpoint NEW.pt
      --source-sha256 SHA256 --input-source ID --split ID --recipe ID
      --sampling replacement-step-seeded-v1|full-partition-step-seeded-v1
      --budget TOTAL [--steps INVOCATION_UPDATES] [--resume OLD.pt]
      [--inference NEW.pt] [--beta1 0.9] [--beta2 0.999] [--epsilon 1e-8] [--amsgrad false]
  mtf_family_train encode --config FILE --input RAW.pt --model INFERENCE.pt
      --source-sha256 SHA256 --state-sha256 SHA256 --output NEW.pt

Raw input archives contain floating CPU data [B,C,H,F] and matching bool feature_mask.
Preparation is explicit identity F32 conversion/observed-finite validation; no scaler is fitted.
synthetic produces artificial F64 observations for any supported F, including F5.
--budget fixes the recipe's total update budget; --steps controls this invocation.
Resume requires exactly the same recipe, data bytes after preparation, and runtime.
The source SHA256 must identify the delivered source inventory; it is a caller declaration.
Checkpoint/inference outputs must be new paths. Artifacts are raw family state, not application admission.
)";
}
}
int main(int argc,char **argv) {
  try {
    if(argc<2||std::string(argv[1])=="--help"||(argc==3&&std::string(argv[2])=="--help")) {usage();return argc<2?1:0;}
    const std::string command(argv[1]);
    if(command=="synthetic"||command=="prepare") {
      const auto args=arguments(argc,argv,command=="synthetic"?std::set<std::string>{"--config","--output","--rows"}:
        std::set<std::string>{"--config","--input","--output"});
      const auto config=mtf::read_settings(required(args,"--config")).model;
      const auto output=required(args,"--output");output_path(output,{required(args,"--config"),optional(args,"--input","")});
      const auto raw=command=="synthetic"?artificial(config,integer(optional(args,"--rows","5"))):read_raw(required(args,"--input"));
      const auto prepared=fam::prepare(raw,config);
      embedding::save_batch(output,command=="synthetic"?raw:prepared);
      std::cout<<"{\"command\":"<<quoted(command)<<",\"preparation\":"<<quoted(fam::kPreparation)
        <<",\"dataset_sha256\":"<<quoted(fam::dataset_identity(prepared))<<",\"output\":"<<quoted(output)<<"}\n";return 0;
    }
    if(command=="train") {
      const auto args=arguments(argc,argv,{"--config","--input","--checkpoint","--source-sha256","--input-source",
        "--split","--recipe","--sampling","--budget","--steps","--resume","--inference","--beta1","--beta2","--epsilon","--amsgrad"});
      const auto input=required(args,"--input"), output=required(args,"--checkpoint"),resume=optional(args,"--resume","");
      const auto inference=optional(args,"--inference","");
      const std::vector<std::string> inputs{input,required(args,"--config"),resume};output_path(output,inputs);
      if(!inference.empty()) {auto aliases=inputs;aliases.push_back(output);output_path(inference,aliases);}
      const auto raw=read_raw(input);const auto r=recipe(args,raw);
      auto session=resume.empty()?fam::Session::fresh(r,raw):fam::Session::resume(resume,r,raw);
      const auto start=session.completed_updates();
      const auto steps=integer(optional(args,"--steps",std::to_string(r.settings.steps-start)));
      TORCH_CHECK(steps>0&&steps<=r.settings.steps-start,"invocation updates exceed remaining fixed budget");
      for(int64_t i=0;i<steps;++i)session.update();
      session.save_training(output);if(!inference.empty())session.export_inference(inference);
      const auto inventory=fam::parameter_inventory(session.model());
      std::cout<<"{\"command\":\"train\",\"physical_updates_this_invocation\":"<<steps
        <<",\"completed_updates\":"<<session.completed_updates()<<",\"fixed_total_budget\":"<<r.settings.steps
        <<",\"source_sha256\":"<<quoted(r.source_sha256)<<",\"dataset_sha256\":"<<quoted(r.dataset_sha256)
        <<",\"recipe_sha256\":"<<quoted(fam::recipe_sha256(r))<<",\"state_sha256\":"<<quoted(session.state_sha256())
        <<",\"optimizer_sha256\":"<<quoted(session.optimizer_sha256())<<",\"runtime\":"<<quoted(fam::runtime_identity())
        <<",\"parameters\":{\"total\":"<<inventory.total<<",\"trainable\":"<<inventory.trainable<<",\"frozen\":"<<inventory.frozen
        <<"},\"checkpoint\":"<<quoted(output)<<",\"inference\":"<<quoted(inference)<<"}\n";return 0;
    }
    if(command=="encode") {
      const auto args=arguments(argc,argv,{"--config","--input","--model","--source-sha256","--state-sha256","--output"});
      const auto input=required(args,"--input"),model=required(args,"--model"),output=required(args,"--output");
      output_path(output,{input,model,required(args,"--config")});
      fam::InferenceExpectation expectation{mtf::read_settings(required(args,"--config")).model,
        required(args,"--source-sha256"),required(args,"--state-sha256")};
      auto inference=fam::load_inference(model,expectation);const auto result=inference.encode(read_raw(input));
      torch::serialize::OutputArchive archive;
      archive.write("pooled_embedding",result.pooled_embedding,true);archive.write("pooled_by_channel",result.pooled_by_channel,true);
      archive.write("pooled_time",result.pooled_time,true);archive.write("pooled_frequency",result.pooled_frequency,true);
      archive.write("sample_valid_mask",result.sample_valid_mask,true);archive.write("channel_valid_mask",result.channel_valid_mask,true);
      embedding::archive::save_archive(output,archive);
      std::cout<<"{\"command\":\"encode\",\"state_sha256\":"<<quoted(inference.state_sha256)
        <<",\"rows\":"<<result.pooled_embedding.size(0)<<",\"output\":"<<quoted(output)<<"}\n";return 0;
    }
    TORCH_CHECK(false,"unknown command: ",command);
  } catch(const std::exception &e) {std::cerr<<"mtf_family_train: "<<e.what()<<'\n';return 1;}
}
