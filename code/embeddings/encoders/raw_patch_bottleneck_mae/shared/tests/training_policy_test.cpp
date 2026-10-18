// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
namespace fs=std::filesystem;
using namespace rpb_test;
int invoke(std::vector<std::string> arguments) {
  std::vector<char *> values;for(auto &argument:arguments)values.push_back(argument.data());
  return rpb::run_cli(static_cast<int>(values.size()),values.data());
}
std::string bytes(const fs::path &path) {
  std::ifstream input(path,std::ios::binary);check(bool(input),"preserved input readable");
  return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
void contract() {
  auto c=config();c.global_bottleneck_mode=2;c.channel_mixer_layers=1;c.export_width=32;
  auto settings=rpb::default_settings();settings.model=c;
  const auto raw=input(c,4);const auto dataset=rpb::describe_dataset(raw,c,"unitless,unitless");
  rpb::Checkpoint original;original.settings=settings;original.model=rpb::Model(c);
  original.scaler=rpb::fit_scaler(raw,c);original.schema_id=dataset.schema_id;
  original.dataset_id=dataset.dataset_id;original.scaler_fit_dataset_id=dataset.dataset_id;
  torch::optim::AdamW optimizer(original.model->parameters(),torch::optim::AdamWOptions(settings.learning_rate));
  const fs::path directory=fs::path(std::getenv("TMPDIR")?std::getenv("TMPDIR"):"/tmp")/
      ("rpb-training-policy-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(directory);
  const auto ordinary=(directory/"ordinary.pt").string(),tagged=(directory/"context.pt").string();
  rpb::save_checkpoint(ordinary,original,optimizer);
  auto loaded=rpb::load_checkpoint(ordinary);check(loaded.training_policy_id.empty(),"legacy/ordinary policy defaults empty");
  torch::serialize::InputArchive ordinary_archive;ordinary_archive.load_from(ordinary,torch::kCPU);torch::Tensor absent;
  check(!ordinary_archive.try_read("training_policy_id",absent,true),"ordinary archive omits optional policy tag");
  original.training_policy_id="rpb-training-context-deletion-v1";rpb::save_checkpoint(tagged,original,optimizer);
  auto contextual=rpb::load_checkpoint(tagged);check(contextual.training_policy_id==original.training_policy_id,"policy tag roundtrip");
  const auto a=loaded.model->named_parameters(),b=contextual.model->named_parameters();check(a.size()==b.size(),"policy changes no modules");
  for(const auto &parameter:a)close(parameter.value(),b[parameter.key()],"policy preserves exact weights",0,0);
  const auto raw_path=(directory/"raw.pt").string();rpb::save_dataset(raw_path,dataset);
  const auto preserved_cp=bytes(tagged),preserved_raw=bytes(raw_path);
  const auto output=(directory/"forbidden-resume.pt").string();
  rejects([&]{invoke({"rpb","train","--resume",tagged,"--input",raw_path,"--checkpoint",output,"--steps","1"});},
      "ordinary resume rejects unsupported view policy");
  check(!fs::exists(output) && bytes(tagged)==preserved_cp && bytes(raw_path)==preserved_raw,
      "rejected resume preserves inputs and creates no training output");
  const auto export_path=(directory/"served.pt").string();
  check(invoke({"rpb","embed","--checkpoint",tagged,"--input",raw_path,"--output",export_path})==0,"tagged weights still serve ordinary embedding");
  torch::serialize::InputArchive served;served.load_from(export_path,torch::kCPU);torch::Tensor tag,global;
  served.read("training_policy_id",tag,true);check(embedding::archive::tensor_text(tag)==original.training_policy_id,"served provenance retains training policy");
  served.read("z_contextual_global",global,true);contextual.model->eval();torch::NoGradGuard no_grad;
  const auto encoded=contextual.model->encode(contextual.scaler.transform(raw,c));
  close(global,encoded.z_contextual_global,"same exact native32 inference",0,0);
  original.training_policy_id="invalid policy id";
  rejects([&]{rpb::save_checkpoint((directory/"invalid.pt").string(),original,optimizer);},"malformed policy ID");
  check(!fs::exists(directory/"invalid.pt"),"invalid policy rejected before writing");
  std::cout << "RPB training policy compatibility and resume guard tests passed; artifacts=" << directory << '\n';
}
}
int main() {
  try {torch::set_num_threads(1);contract();return 0;}
  catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
