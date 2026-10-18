// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
namespace {

void check(bool condition,const std::string &message) {
  if(!condition) throw std::runtime_error(message);
}
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &message) {
  check(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() && torch::equal(a,b),message);
}
template<typename Function> void rejects(Function fn,const std::string &message) {
  bool rejected=false;try{fn();}catch(const std::exception &){rejected=true;}
  check(rejected,message+" was accepted");
}
struct TemporaryFiles {
  std::filesystem::path directory;
  TemporaryFiles() {
    directory=std::filesystem::temp_directory_path()/
        ("rpb-mae-workflow-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(directory),"cannot create unique test directory");
  }
  ~TemporaryFiles(){std::error_code error;std::filesystem::remove_all(directory,error);}
  std::string file(const std::string &name)const{return(directory/name).string();}
};
int invoke(std::vector<std::string> args) {
  std::vector<char *> argv;for(auto &arg:args)argv.push_back(arg.data());
  return rpb::run_cli(static_cast<int>(argv.size()),argv.data());
}
rpb::Settings small_settings(int64_t mixer_layers=0,int64_t global_mode=0) {
  auto s=rpb::default_settings();s.model.channel_count=2;s.model.input_width=1;
  s.model.channel_mixer_layers=mixer_layers;
  s.model.global_bottleneck_mode=global_mode;
  s.model.channel_ids={41,7};s.model.encoder_width=8;s.model.export_width=4;
  s.model.num_heads=2;s.model.num_layers=1;s.model.feedforward_width=16;s.model.decoder_hidden_width=8;
  s.model.dropout=0.1;s.seed=71;s.batch_size=3;s.log_every=2;s.steps=3;s.attempt_limit=30;
  s.learning_rate=0.0023;s.weight_decay=0.007;s.gradient_clip_norm=0.75;return s;
}
void write_config(const std::string &path,const rpb::Settings &settings) {
  std::ofstream file(path);check(file.good(),"cannot write test config");file<<rpb::settings_text(settings);
}
void compare_models(rpb::Model &a,rpb::Model &b,const std::string &label) {
  const auto expected=a->named_parameters();
  check(expected.size()==b->named_parameters().size(),label+" parameter count");
  for(const auto &p:b->named_parameters())exact(expected[p.key()],p.value(),label+" "+p.key());
  const auto expected_buffers=a->named_buffers();
  check(expected_buffers.size()==b->named_buffers().size(),label+" buffer count");
  for(const auto &p:b->named_buffers())exact(expected_buffers[p.key()],p.value(),label+" buffer "+p.key());
}

void test_settings(const TemporaryFiles &files,int64_t mixer_layers) {
  const auto settings=small_settings(mixer_layers);const auto text=rpb::settings_text(settings);
  check(rpb::settings_text(rpb::parse_settings(text))==text,"config round-trip changed settings");
  check(rpb::parse_settings(text).model.channel_mixer_layers==mixer_layers,"mixer configuration was not persisted");
  check(text.find("global_bottleneck_mode=")==std::string::npos &&
      rpb::parse_settings(text).model.global_bottleneck_mode==0,
      "legacy settings must omit/default global bottleneck mode");
  check(rpb::settings_text(rpb::parse_settings(text+"global_bottleneck_mode=0\n"))==text,
      "explicit mode0 changed archived canonical settings text");
  for(const int64_t global_mode:{1,2,3}) {
    auto global=settings;global.model.global_bottleneck_mode=global_mode;
    const auto encoded=rpb::settings_text(global);
    check(encoded.find("global_bottleneck_mode="+std::to_string(global_mode)+"\n")!=std::string::npos &&
        rpb::parse_settings(encoded).model.global_bottleneck_mode==global_mode &&
        rpb::settings_text(rpb::parse_settings(encoded))==encoded,
        "global mode settings persistence");
  }
  // A historical settings string has no mixer key and must retain independent behavior.
  std::istringstream source(text);std::ostringstream legacy;std::string line;
  while(std::getline(source,line))
    if(line.rfind("channel_mixer_layers=",0)!=0)legacy<<line<<'\n';
  auto restored_legacy=rpb::parse_settings(legacy.str());
  check(restored_legacy.model.channel_mixer_layers==0 &&
      std::string(rpb::output_semantics(restored_legacy.model))==rpb::kOutputSemantics,
      "absent legacy mixer key changed independent semantics");
  const auto path=files.file("small.conf");write_config(path,settings);
  check(rpb::settings_text(rpb::read_settings(path))==text,"config file differs");
  for(const auto &bad:{"unknown=1\n","steps=0\n","attempt_limit=0\n","checkpoint_every=-1\n",
      "learning_rate=nan\n","dropout=inf\n","sampling_interval=0\n","history_length=31\n",
      "encoder_width=7\n","channel_count=2\nchannel_ids=1,1\n","seed=3\nseed=4\n",
      "channel_mixer_layers=-1\n","channel_mixer_layers=0\nchannel_mixer_layers=1\n"})
    rejects([&]{rpb::parse_settings(bad);},std::string("invalid config ")+bad);
  for(const auto *bad:{"global_bottleneck_mode=-1\n","global_bottleneck_mode=4\n",
      "global_bottleneck_mode=1\nglobal_bottleneck_mode=2\n"})
    rejects([&]{rpb::parse_settings(bad);},std::string("invalid global mode config ")+bad);
}

void test_raw_archive(const TemporaryFiles &files,int64_t mixer_layers) {
  const auto settings=small_settings(mixer_layers);auto dataset=rpb::synthetic_dataset(settings.model,5,93);
  dataset.input.data.index_put_({0,0,0,0},1e12);
  dataset.input.data.index_put_({0,0,1,0},1e12+0.01);
  dataset.input.observed.index_put_({0,0,0,0},true);dataset.input.observed.index_put_({0,0,1,0},true);
  const auto path=files.file("raw.pt");rpb::save_dataset(path,dataset);
  const auto restored=rpb::load_dataset(path,settings.model);
  exact(restored.input.data,dataset.input.data,"raw float64 values lost precision");
  check(restored.input.data[0][0][1][0].item<double>()!=restored.input.data[0][0][0][0].item<double>(),
        "large-offset variation was quantized during archive round-trip");
  exact(restored.input.observed,dataset.input.observed,"raw observed support changed");
  exact(restored.input.channel_ids,dataset.input.channel_ids,"semantic IDs changed");
  exact(restored.input.endpoints,dataset.input.endpoints,"endpoints changed");
  auto wrong=settings.model;wrong.sampling_interval=2;
  rejects([&]{rpb::load_dataset(path,wrong);},"wrong uniform sampling interval");
  auto scaler=rpb::fit_scaler(restored.input,settings.model);
  const auto scaler_path=files.file("scaler.pt");rpb::save_scaler(scaler_path,scaler,settings.model,restored.schema_id);
  auto saved=rpb::load_scaler(scaler_path,settings.model,restored.schema_id);
  check(saved.identity()==scaler.identity(),"frozen scaler identity changed");
  exact(saved.mean,scaler.mean,"frozen scaler mean changed");
  auto alternate=settings.model;alternate.channel_mixer_layers=1-mixer_layers;
  alternate.global_bottleneck_mode=2;
  const auto same_raw=rpb::load_dataset(path,alternate);
  const auto same_scaler=rpb::load_scaler(scaler_path,alternate,same_raw.schema_id);
  check(same_raw.schema_id==restored.schema_id && same_raw.dataset_id==restored.dataset_id &&
      same_scaler.identity()==saved.identity(),"mixer/global architecture changed raw/scaler provenance");
  exact(same_scaler.transform(same_raw.input,alternate).data,
      saved.transform(restored.input,settings.model).data,"mixer changed frozen preprocessing");
  rejects([&]{rpb::load_scaler(scaler_path,settings.model,"wrong-schema");},"incompatible scaler schema");
  rejects([&]{rpb::load_checkpoint(path);},"raw archive cross-loaded as checkpoint");
  embedding::Batch legacy{dataset.input.data.to(torch::kFloat32),dataset.input.observed};
  const auto legacy_path=files.file("legacy.pt");embedding::save_batch(legacy_path,legacy);
  rejects([&]{rpb::load_dataset(legacy_path,settings.model);},"untagged legacy data archive");
  torch::serialize::OutputArchive baseline;
  baseline.write("format_version",torch::tensor(int64_t{1}),true);
  baseline.write("settings",embedding::archive::text_tensor("channel_count=2\n"),true);
  const auto baseline_path=files.file("baseline.pt");embedding::archive::save_archive(baseline_path,baseline);
  rejects([&]{rpb::load_checkpoint(baseline_path);},"baseline checkpoint lacking encoder ID");
}

void test_training_resume_export(const TemporaryFiles &files,int64_t mixer_layers,int64_t global_mode=0) {
  const auto settings=small_settings(mixer_layers,global_mode);const auto config=files.file("train.conf");write_config(config,settings);
  const auto input=files.file("train.pt");rpb::save_dataset(input,rpb::synthetic_dataset(settings.model,7,97));
  const auto whole_path=files.file("whole.pt"),split_path=files.file("split.pt");
  check(invoke({"rpb","train","--config",config,"--input",input,"--checkpoint",whole_path,
      "--steps","3","--checkpoint-every","1"})==0,"whole training failed");
  check(invoke({"rpb","train","--config",config,"--input",input,"--checkpoint",split_path,
      "--steps","1","--checkpoint-every","1"})==0,"initial training failed");
  check(invoke({"rpb","train","--resume",split_path,"--input",input,"--checkpoint",split_path,
      "--steps","2"})==0,"resumed training failed");
  auto whole=rpb::load_checkpoint(whole_path),continued=rpb::load_checkpoint(split_path);
  check(whole.completed_steps==3 && continued.completed_steps==3 &&
      whole.attempted_steps==continued.attempted_steps,"resume counters differ");
  check(whole.settings.model.channel_mixer_layers==mixer_layers &&
      continued.settings.model.channel_mixer_layers==mixer_layers &&
      whole.settings.model.global_bottleneck_mode==global_mode &&
      continued.settings.model.global_bottleneck_mode==global_mode &&
      whole.settings.seed==settings.seed && continued.settings.seed==settings.seed,
      "resume lost mixer configuration or actual seed");
  torch::serialize::InputArchive saved; saved.load_from(split_path,torch::kCPU);
  torch::Tensor mode_tag,reconstruction_tag;
  const bool has_mode=saved.try_read("global_bottleneck_mode",mode_tag,true);
  const bool has_reconstruction=saved.try_read("reconstruction_export_semantics",reconstruction_tag,true);
  check(has_mode==(global_mode>0) && has_reconstruction==(global_mode>0),
      "legacy/new checkpoint global metadata presence");
  if(global_mode>0) {
    check(mode_tag.item<int64_t>()==global_mode &&
        embedding::archive::tensor_text(reconstruction_tag)==rpb::reconstruction_output_semantics(settings.model),
        "new checkpoint exact global export semantics");
    // An intentionally incomplete envelope proves this check precedes all model,
    // scaler and optimizer loading: failure must name the mismatched mode tag.
    torch::serialize::OutputArchive wrong;
    for(const auto *key:{"encoder_id","format_version","artifact_kind","rng_policy","settings",
                        "configuration_id","output_semantics","reconstruction_export_semantics"}) {
      torch::Tensor field;saved.read(key,field,true);wrong.write(key,field,true);
    }
    wrong.write("global_bottleneck_mode",torch::tensor(global_mode==1?int64_t{2}:int64_t{1}),true);
    const auto wrong_path=files.file("wrong-global-tag.pt");embedding::archive::save_archive(wrong_path,wrong);
    bool rejected_before_weights=false;
    try{rpb::load_checkpoint(wrong_path);}
    catch(const std::exception &error) {
      rejected_before_weights=std::string(error.what()).find("global bottleneck metadata mismatch")!=std::string::npos;
    }
    check(rejected_before_weights,"global checkpoint mismatch was not rejected before weight loading");
  }
  compare_models(whole.model,continued.model,"exact CPU continuation");
  check(whole.scaler.identity()==continued.scaler.identity() && whole.dataset_id==continued.dataset_id,
      "resume preprocessing/dataset provenance changed");
  auto dataset=rpb::load_dataset(input,settings.model);dataset.input.observed.index_put_({0,1},false);
  dataset.input.observed.index_put_({1},false);const auto missing=files.file("missing.pt");rpb::save_dataset(missing,dataset);
  const auto output=files.file("embeddings.pt");
  check(invoke({"rpb","embed","--checkpoint",split_path,"--input",missing,"--output",output,
      "--batch-size","2"})==0,"embedding export failed");
  torch::serialize::InputArchive archive;archive.load_from(output,torch::kCPU);
  torch::Tensor local,global,cv,sv,counts,ids;
  archive.read("z_local",local,true);archive.read("z_global",global,true);
  archive.read("channel_valid_mask",cv,true);archive.read("sample_valid_mask",sv,true);
  archive.read("visible_observation_counts",counts,true);archive.read("channel_ids",ids,true);
  torch::Tensor semantics;archive.read("output_semantics",semantics,true);
  check(embedding::archive::tensor_text(semantics)==rpb::output_semantics(settings.model),
      "export semantics differ from persisted architecture");
  torch::Tensor exported_mode,exported_reconstruction;
  const bool export_has_mode=archive.try_read("global_bottleneck_mode",exported_mode,true);
  const bool export_has_reconstruction=archive.try_read("reconstruction_export_semantics",exported_reconstruction,true);
  check(export_has_mode==(global_mode>0) && export_has_reconstruction==(global_mode>0),
      "legacy/new embedding global metadata presence");
  if(global_mode>0)
    check(exported_mode.item<int64_t>()==global_mode &&
        embedding::archive::tensor_text(exported_reconstruction)==rpb::reconstruction_output_semantics(settings.model),
        "global embedding served export metadata");
  continued.model->eval();torch::NoGradGuard no_grad;
  const auto expected=continued.model->encode(continued.scaler.transform(dataset.input,settings.model));
  check(torch::allclose(local,expected.z_local,1e-6,1e-6),"batched local export differs");
  check(torch::allclose(global,expected.z_global,1e-6,1e-6),"batched global export differs");
  exact(cv,expected.channel_valid_mask,"export channel validity differs");
  exact(sv,expected.sample_valid_mask,"export sample validity differs");
  exact(counts,expected.visible_observation_counts,"export support counts differ");
  exact(ids,expected.channel_ids,"export semantic IDs differ");
  check(!cv[0][1].item<bool>() && local[0][1].eq(0).all().item<bool>() &&
        !sv[1].item<bool>() && global[1].eq(0).all().item<bool>(),"absent output laundered into validity");
  torch::Tensor contextual,contextual_global;
  const bool has_contextual=archive.try_read("z_contextual",contextual,true);
  const bool has_contextual_global=archive.try_read("z_contextual_global",contextual_global,true);
  check(has_contextual==(mixer_layers>0) && has_contextual_global==(mixer_layers>0),
      "contextual archive fields disagree with enabled architecture");
  if(mixer_layers>0) {
    check(expected.z_contextual.defined() && expected.z_contextual_global.defined() &&
        torch::allclose(contextual,expected.z_contextual,1e-6,1e-6) &&
        torch::allclose(contextual_global,expected.z_contextual_global,1e-6,1e-6),
        "batched contextual exports differ from saved model");
    check(contextual[0][1].eq(0).all().item<bool>() &&
        contextual_global[1].eq(0).all().item<bool>(),"context peers laundered absent output validity");
  } else check(!expected.z_contextual.defined() && !expected.z_contextual_global.defined(),
      "disabled mixer created contextual outputs");
  rejects([&]{invoke({"rpb","train","--resume",split_path,"--input",missing,"--checkpoint",files.file("wrong.pt"),"--steps","1"});},
      "changed resume dataset");
  rejects([&]{invoke({"rpb","train","--resume",split_path,"--input",input,"--checkpoint",files.file("wrong-size.pt"),"--batch-size","2"});},
      "changed resume training batch size");
}

void test_skips_and_aliases(const TemporaryFiles &files,int64_t mixer_layers) {
  auto settings=small_settings(mixer_layers);settings.model.dropout=0;
  const auto config=files.file("skip.conf");write_config(config,settings);
  auto full=rpb::synthetic_dataset(settings.model,5,19);
  const auto full_path=files.file("full.pt");rpb::save_dataset(full_path,full);
  const auto loaded=rpb::load_dataset(full_path,settings.model);
  const auto scaler_path=files.file("skip-scaler.pt");
  rpb::save_scaler(scaler_path,rpb::fit_scaler(loaded.input,settings.model),settings.model,loaded.schema_id);
  auto sparse=loaded;sparse.input.observed=torch::zeros_like(sparse.input.observed);
  sparse.input.observed.slice(2,0,8).fill_(true); // One observed patch: inference-valid, training-ineligible.
  const auto input=files.file("sparse.pt");rpb::save_dataset(input,sparse);
  const auto checkpoint_path=files.file("skipped.pt");
  torch::manual_seed(settings.seed);auto initial=rpb::Model(settings.model);
  rejects([&]{invoke({"rpb","train","--config",config,"--input",input,"--scaler",scaler_path,
      "--checkpoint",checkpoint_path,"--steps","1","--attempt-limit","3","--checkpoint-every","1"});},"all-ineligible attempt exhaustion");
  auto saved=rpb::load_checkpoint(checkpoint_path);
  check(saved.attempted_steps==3 && saved.completed_steps==0,"skip counters were not saved");
  compare_models(initial,saved.model,"skip must not update weights or decay");
  torch::optim::AdamW optimizer(saved.model->parameters(),torch::optim::AdamWOptions(settings.learning_rate));
  rpb::load_optimizer(checkpoint_path,optimizer);
  check(optimizer.state().empty(),"skips initialized/mutated optimizer state");
  rejects([&]{invoke({"rpb","train","--resume",checkpoint_path,"--input",input,
      "--checkpoint",checkpoint_path,"--steps","1","--attempt-limit","2"});},"resumed ineligible attempt exhaustion");
  auto resumed=rpb::load_checkpoint(checkpoint_path);
  check(resumed.attempted_steps==5 && resumed.completed_steps==0,"skip resume lost attempted counter");
  compare_models(initial,resumed.model,"skip resume must preserve weights");
  rejects([&]{invoke({"rpb","prepare","--config",config,"--input",input,"--output",input});},"input/output alias");
  const auto alias=files.file("hardlink.pt");std::filesystem::create_hard_link(input,alias);
  rejects([&]{invoke({"rpb","train","--config",config,"--input",input,"--checkpoint",alias});},"hardlink input/output alias");
  const auto unchanged=rpb::load_dataset(input,settings.model);
  check(unchanged.dataset_id==rpb::load_dataset(alias,settings.model).dataset_id,"alias rejection damaged source archive");
}

std::vector<char> file_bytes(const std::filesystem::path &path) {
  std::ifstream file(path,std::ios::binary);check(file.good(),"cannot read preserved fixture "+path.string());
  return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
void test_preserved_legacy_fixture() {
  const auto fixture=std::getenv("RPB_LEGACY_FIXTURE_DIR");
  if(!fixture || !*fixture)return;
  const std::filesystem::path directory(fixture);
  const auto checkpoint_path=directory/"resumed.pt",input_path=directory/"data.pt",old_export_path=directory/"embeddings.pt";
  const auto original_checkpoint=file_bytes(checkpoint_path),original_input=file_bytes(input_path),
      original_export=file_bytes(old_export_path);
  auto original=rpb::load_checkpoint(checkpoint_path.string());
  check(original.settings.model.channel_mixer_layers==0,"preserved legacy checkpoint did not default to independent mode");
  check(original.settings.model.global_bottleneck_mode==0 &&
      rpb::settings_text(original.settings).find("global_bottleneck_mode=")==std::string::npos,
      "preserved legacy checkpoint changed canonical global mode/settings");
  const auto dataset=rpb::load_dataset(input_path.string(),original.settings.model);
  check(dataset.schema_id==original.schema_id,"legacy checkpoint/raw schema association");
  TemporaryFiles files;
  const auto new_export=files.file("legacy-reexport.pt");
  check(invoke({"rpb","embed","--checkpoint",checkpoint_path.string(),"--input",input_path.string(),
      "--output",new_export,"--batch-size","4","--device","cpu"})==0,"legacy checkpoint embedding failed");
  torch::serialize::InputArchive before,after;
  before.load_from(old_export_path.string(),torch::kCPU);after.load_from(new_export,torch::kCPU);
  for(const auto *key:{"z_local","z_global"}) {
    torch::Tensor expected,actual;before.read(key,expected,true);after.read(key,actual,true);
    check(expected.sizes()==actual.sizes() && expected.scalar_type()==actual.scalar_type() &&
        torch::allclose(expected,actual,1e-6,1e-6),std::string("legacy served export changed ")+key);
  }
  for(const auto *key:{"channel_valid_mask","sample_valid_mask","visible_observation_counts",
                      "visible_patch_counts","channel_ids","endpoints","sampling_interval"}) {
    torch::Tensor expected,actual;before.read(key,expected,true);after.read(key,actual,true);
    exact(expected,actual,std::string("legacy export metadata changed ")+key);
  }
  for(const auto *key:{"schema_id","dataset_id","preprocessing_id","scaler_fit_dataset_id","output_semantics"}) {
    torch::Tensor expected,actual;before.read(key,expected,true);after.read(key,actual,true);
    exact(expected,actual,std::string("legacy export provenance changed ")+key);
  }
  const auto continued_path=files.file("legacy-continued.pt");
  check(invoke({"rpb","train","--resume",checkpoint_path.string(),"--input",input_path.string(),
      "--checkpoint",continued_path,"--steps","1","--device","cpu"})==0,"legacy checkpoint continuation failed");
  auto continued=rpb::load_checkpoint(continued_path);
  check(continued.completed_steps==original.completed_steps+1 &&
      continued.attempted_steps>=original.attempted_steps+1 &&
      continued.settings.model.channel_mixer_layers==0 && continued.settings.seed==original.settings.seed &&
      continued.settings.model.global_bottleneck_mode==0 &&
      continued.dataset_id==original.dataset_id && continued.schema_id==original.schema_id &&
      continued.scaler_fit_dataset_id==original.scaler_fit_dataset_id &&
      continued.scaler.identity()==original.scaler.identity(),"legacy continuation lost independent state/provenance");
  exact(continued.scaler.mean,original.scaler.mean,"legacy continuation changed scaler mean");
  exact(continued.scaler.scale,original.scaler.scale,"legacy continuation changed scaler scale");
  auto preserved=rpb::load_checkpoint(checkpoint_path.string());
  check(preserved.attempted_steps==original.attempted_steps && preserved.completed_steps==original.completed_steps,
      "legacy checkpoint counters were overwritten");
  compare_models(original.model,preserved.model,"preserved legacy weights");
  check(file_bytes(checkpoint_path)==original_checkpoint && file_bytes(input_path)==original_input &&
      file_bytes(old_export_path)==original_export,"legacy fixture archives were modified");
  std::cout<<"PASS: preserved pre-mixer checkpoint export and one-update continuation "<<directory<<'\n';
}

} // namespace
int main() {
  try {
    torch::set_num_threads(1);
    for(const int64_t mixer_layers:{0,1}) {
      TemporaryFiles files;
      test_settings(files,mixer_layers);test_raw_archive(files,mixer_layers);
      test_training_resume_export(files,mixer_layers);test_skips_and_aliases(files,mixer_layers);
      for(const int64_t global_mode:{1,2,3}) {
        TemporaryFiles global_files;test_training_resume_export(global_files,mixer_layers,global_mode);
      }
    }
    test_preserved_legacy_fixture();
    std::cout<<"PASS: RPB-MAE independent/contextual raw precision/schema, scaler, checkpoint, exact CPU resume, export, skip recovery and path preservation\n";
    return 0;
  }catch(const std::exception &error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
