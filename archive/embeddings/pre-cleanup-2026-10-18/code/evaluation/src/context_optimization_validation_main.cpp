// SPDX-License-Identifier: MIT
#include "embedding/shared/paired_pooling.h"
#include "embedding/shared/archive_readout.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replay_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
#ifndef EVALUATION_GIT_HEAD
#define EVALUATION_GIT_HEAD "unrecorded"
#endif
#ifndef EVALUATION_GIT_DIRTY
#define EVALUATION_GIT_DIRTY "unrecorded"
#endif
namespace {
namespace fs = std::filesystem;
namespace ev = embedding::evaluation;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
const std::string protocol = "context-optimization-validation-v1";
const std::string retained = "output/runs/rpb-context-deletion/context-deletion-KTbte3";
const std::map<uint64_t,std::string> parent_hashes{
  {3101,"f995739e4ec90f411ec153f2382c9e5c3afc8104675bc1d1f018960cde644c18"},
  {3202,"8eec0ccdc28f36c79e0692acbd03ce862f3d02307a86925fc3e762d8bddebc38"},
  {3303,"2c81b7a728cf7ea11bbd1a6b4d3b98feba170d8bafb35ceb609874ea364926bd"}};
constexpr uint64_t deletion_stream = 0x63766c3164726f70ULL;
const std::string view_id = "validation-dropout-030";
void require(bool value,const std::string &message) {
  if(!value) throw std::runtime_error("[context optimization validation] " + message);
}
bool sha(const std::string &s) {
  return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\') out << '\\' << c;
    else if(c=='\n') out << "\\n";
    else { require(c>=32,"unsupported control character"); out << c; }
  }
  out << '"'; return out.str();
}
void write_new(const fs::path &path,const std::string &value) {
  const int file=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  require(file>=0,"output must be new: " + path.string());
  size_t written=0;
  while(written<value.size()) {
    const auto count=::write(file,value.data()+written,value.size()-written);
    if(count<=0){::close(file);throw std::runtime_error("output write failed");}
    written+=static_cast<size_t>(count);
  }
  const int synced=::fsync(file),closed=::close(file);
  require(synced==0 && closed==0,"output fsync failed");
  const int directory=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
  require(directory>=0,"output directory unavailable");
  const int ds=::fsync(directory),dc=::close(directory);
  require(ds==0 && dc==0,"directory fsync failed");
}
std::map<std::string,std::string> hashes(const fs::path &path) {
  std::ifstream in(path);require(bool(in),"explicit frozen retained manifest required");
  std::map<std::string,std::string> out;std::string line;
  while(std::getline(in,line)) {
    require(line.size()>66 && line.substr(64,2)=="  " && sha(line.substr(0,64)),"bad frozen hash record");
    const fs::path relative(line.substr(66));
    require(!relative.is_absolute() && relative.lexically_normal()==relative &&
        relative.generic_string().find("..") == std::string::npos,"retained path outside capsule");
    require(out.emplace(fs::absolute(fs::path(retained)/relative).string(),line.substr(0,64)).second,
        "duplicate retained input");
  }
  require(in.eof() && !out.empty(),"empty/unreadable frozen manifest");return out;
}
torch::Tensor tensor(torch::serialize::InputArchive &archive,const std::string &key) {
  torch::Tensor out;archive.read(key,out,true);return out;
}
void feature_parity(const ev::FeatureSurface &actual,const std::string &old) {
  torch::serialize::InputArchive archive;archive.load_from(old,torch::kCPU);
  require(torch::equal(actual.values,tensor(archive,"features")) &&
      torch::equal(actual.valid,tensor(archive,"valid")),"original512 native export differs");
}
void query_parity(const std::string &actual,const std::string &old) {
  torch::serialize::InputArchive a,b;a.load_from(actual,torch::kCPU);b.load_from(old,torch::kCPU);
  require(a.keys()==b.keys(),"original512 reconstruction schema differs");
  for(const auto &key:a.keys()) require(torch::equal(tensor(a,key),tensor(b,key)),
      "original512 reconstruction witness differs: " + key);
}
ev::ProviderFitInput metadata(const ev::ControlledDataset &train,uint64_t seed,const ev::NativeCurveRun &recipe) {
  auto legal=train.observed;
  legal.data=torch::where(legal.feature_mask,legal.data,torch::zeros_like(legal.data)).detach().clone();
  legal.feature_mask=legal.feature_mask.clone();
  return {legal,recipe.card.shape,seed,train.source_ids,recipe.card.channel_ids,recipe.card.feature_units,
      "native-curve-v1/lag_sign",recipe.card.sampling_interval,31*recipe.card.sampling_interval};
}
std::string progress_json(const ev::CurveProgress &p) {
  std::ostringstream out;out << std::setprecision(17)
    << "{\"attempted\":" << p.attempted << ",\"completed\":" << p.completed
    << ",\"parameter_count\":" << p.parameter_count << ",\"cuda_parameter_count\":" << p.cuda_parameter_count
    << ",\"sampled_rows\":" << p.sampled_rows << ",\"cumulative_training_seconds\":" << p.training_seconds
    << ",\"training_device\":" << quote(p.training_device)
    << ",\"preprocessing_id\":" << quote(p.preprocessing_id)
    << ",\"training_dataset_id\":" << quote(p.training_dataset_id)
    << ",\"last_input_cuda\":" << (p.last_input_cuda?"true":"false")
    << ",\"last_loss_cuda\":" << (p.last_loss_cuda?"true":"false")
    << ",\"finite_gradients\":" << (p.finite_gradients?"true":"false")
    << ",\"weights_changed\":" << (p.weights_changed?"true":"false") << ",\"losses\":[";
  for(size_t i=0;i<p.losses.size();++i) {
    const auto &l=p.losses[i];if(i)out << ',';
    out << "{\"attempted\":" << l.attempted << ",\"completed\":" << l.completed
      << ",\"target_cells\":" << l.target_cells << ",\"loss\":" << l.loss
      << ",\"gradient_norm\":" << l.gradient_norm << '}';
  }
  out << "]}";return out.str();
}
void text(torch::serialize::OutputArchive &a,const std::string &key,const std::string &s) {
  a.write(key,torch::tensor(std::vector<uint8_t>(s.begin(),s.end()),torch::kUInt8),true);
}
void save_deleted_validation(const std::string &path,const ev::ControlledDataset &val,
    const ev::CoordinateDeletionView &view,uint64_t seed) {
  require(!fs::exists(path),"deletion archive already exists");
  torch::serialize::OutputArchive a;
  a.write("observed",view.observations.data,true);
  a.write("feature_mask",view.observations.feature_mask,true);
  a.write("labels_scoring_only",val.labels,true);
  a.write("requested_erasure",view.requested_erasure,true);
  a.write("base_observed",val.observed.data,true);
  a.write("base_feature_mask",val.observed.feature_mask,true);
  const auto oracle=ev::raw_oracle(ev::Task::lag_sign,view.observations);
  a.write("raw_oracle_predictions",oracle.predictions,true);
  a.write("raw_oracle_valid",oracle.valid,true);
  std::ostringstream ids;ids << '[';
  for(size_t i=0;i<val.source_ids.size();++i){if(i)ids << ',';ids << quote(val.source_ids[i]);}ids << ']';
  text(a,"source_ids_json",ids.str());text(a,"artifact_kind","validation_coordinate_deletion_v1");
  text(a,"rng_namespace",protocol);text(a,"view_id",view_id);
  text(a,"actual_corruption_seed",std::to_string(seed));
  text(a,"rate","0.30");text(a,"corruption_stream",std::to_string(deletion_stream));
  a.save_to(path);
}
} // namespace
int main(int argc,char **argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--source-id"){std::cout << EVALUATION_SOURCE_ID << '\n';return 0;}
    if(argc==1 || (argc==2 && std::string(argv[1])=="--help")) {
      std::cout << "embedding_context_optimization_validation --output NEW_DIRECTORY --retained-hashes FROZEN_MANIFEST\n"
          "Fixed context-optimization-validation-v1; exact RPB-v6 replay512, then live1024/2048; TRAIN/VALIDATION only.\n";
      return argc==1?1:0;
    }
    std::map<std::string,std::string> options;
    for(int i=1;i<argc;i+=2)require(i+1<argc &&
        (std::string(argv[i])=="--output" || std::string(argv[i])=="--retained-hashes") &&
        options.emplace(argv[i],argv[i+1]).second,"unknown/duplicate option or missing value");
    require(options.size()==2 && !options.at("--output").empty(),"new output and frozen manifest required");
    require(sha(EVALUATION_SOURCE_ID) && torch::cuda::is_available(),"source fingerprint and actual CUDA required");
    const fs::path output=fs::absolute(options.at("--output")).lexically_normal();
    require(!fs::exists(output),"output already exists");fs::create_directories(output.parent_path());
    const auto expected=hashes(options.at("--retained-hashes"));
    const auto bind=[&](const fs::path &p) {
      const auto full=fs::absolute(p).lexically_normal().string();
      require(expected.count(full),"undeclared retained input: " + full);return full;
    };
    ev::NativeCurveRun recipe;recipe.source_fingerprint=EVALUATION_SOURCE_ID;recipe.model_tag="RPB-v6";
    recipe.card.seeds={3101,3202,3303};recipe.card.tasks={ev::Task::lag_sign};
    recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.stress_sweep=false;
    recipe.card.id=protocol;recipe.milestones={512,1024,2048};
    auto settings=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf");
    settings.model.device=torch::kCUDA;
    std::ostringstream plan;plan << "{\"protocol\":" << quote(protocol)
      << ",\"policy_version\":\"1.2\",\"stage\":\"development\",\"model_tag\":\"RPB-v6\","
      "\"masters\":[3101,3202,3303],\"absolute_budgets\":[512,1024,2048],\"source_pairs\":[128,64],"
      "\"batch_size\":8,\"device\":\"cuda\",\"native_size\":32,\"post_encoder_pca\":false,"
      "\"test_access\":false,\"stress_access\":false,\"selection\":false,\"promotion\":false,"
      "\"continuation\":\"fresh deterministic replay512; exact model/buffers/AdamW/scaler/counters gate; same live optimizer thereafter\","
      "\"readouts\":\"fixed ridge1; tanh16 Adam0.01x100; base seeds2701,2802,2903; TRAIN refit at each point; same head for intact and deleted VALIDATION\","
      "\"validation_view\":{\"id\":" << quote(view_id) << ",\"rate\":0.30,\"stream\":" << deletion_stream
      << ",\"seed_policy\":\"stream_seed(master,stream)\",\"rng_namespace\":" << quote(protocol)
      << ",\"pair_shared\":true,\"fixed_across_budgets\":true,\"repair\":false},"
      "\"historical_v4_reference\":\"fixed512 mean TRAIN MAE0.061601474; VALIDATION MAE0.063486190; known intact VALIDATION ridge94.7916667; no v4 score on new deletion view\","
      "\"retained_inventory_sha256\":\"62415443de5db9ebe7509d31d79b3bbab8d95efa3bf72b7afde3466b1ae021dd\","
      "\"source_fingerprint\":" << quote(EVALUATION_SOURCE_ID) << ",\"git_head\":" << quote(EVALUATION_GIT_HEAD)
      << ",\"git_dirty\":" << quote(EVALUATION_GIT_DIRTY) << ",\"retained_hash_manifest\":"
      << quote(fs::absolute(options.at("--retained-hashes")).string())
      << ",\"actual_parent_settings\":\"pinned per-master checkpoint producer audit; only completed steps ceiling extended\",\"parents\":[";
    bool first=true;
    for(const auto &[master,checksum]:parent_hashes) {
      const auto cp=bind(fs::path(retained)/"results"/("seed-"+std::to_string(master)+"-lag_sign")/
          "candidate-milestone-512/checkpoint.pt");
      require(expected.at(cp)==checksum,"wrong exact512 parent");if(!first)plan << ',';first=false;
      plan << "{\"master_seed\":" << master << ",\"checkpoint\":" << quote(cp)
        << ",\"sha256\":" << quote(checksum) << '}';
    }
    plan << "]}\n";write_new(output.string()+".launch-plan.json",plan.str());
    require(fs::create_directory(output),"cannot claim exclusive output");
    torch::set_num_threads(1);rpb::run_native_curve_cuda_gate(settings,output.string()+"-cuda-gate");
    std::cout << "Actual CUDA inference gate passed; replay state gate precedes continuation.\n" << std::flush;
    ev::ArchiveReadoutRun readouts;readouts.output_directory=(output/"readouts").string();
    readouts.source_fingerprint=EVALUATION_SOURCE_ID;readouts.git_head=EVALUATION_GIT_HEAD;
    readouts.git_dirty=EVALUATION_GIT_DIRTY;
    std::ostringstream report;report << "{\"protocol\":" << quote(protocol)
      << ",\"model_tag\":\"RPB-v6\",\"test_access\":false,\"selection\":false,\"promotion\":false,\"points\":[";
    first=true;std::set<std::string> all_sources;
    for(const auto &[master,checksum]:parent_hashes) {
      (void)checksum;const auto old=fs::path(retained)/"results"/("seed-"+std::to_string(master)+"-lag_sign");
      const auto directory=output/("seed-"+std::to_string(master)+"-lag_sign");
      require(fs::create_directory(directory),"cohort output exists");
      const auto train_path=bind(old/"controlled-training.pt"),val_path=bind(old/"controlled-validation.pt");
      const auto train=ev::load_native_development_observations(train_path),val=ev::load_native_development_observations(val_path);
      require(train.labels.numel()==256 && val.labels.numel()==128,"wrong frozen split size");
      for(const auto *split:{&train,&val}) {
        const std::set<std::string> sources(split->source_ids.begin(),split->source_ids.end());
        for(const auto &id:sources)require(all_sources.insert(id).second,"TRAIN/VALIDATION/master source overlap");
      }
      const auto parent=bind(old/"candidate-milestone-512/checkpoint.pt");
      const auto raw=bind(parent+".training-raw.pt");bind(parent+".audit.pt");bind(parent+".scaler.pt");
      for(const std::string repetition:{"rep-1","rep-2","rep-3"}) {
        bind(old/"candidate-milestone-512"/repetition/"fit.pt");
        bind(old/"candidate-milestone-512"/repetition/"validation-predictions.pt");
      }
      const auto replay_point=directory/"milestone-512";
      require(fs::create_directory(replay_point),"replay point output exists");
      const auto replay_cp=(replay_point/"checkpoint.pt").string();
      const auto fit=metadata(train,master,recipe);
      auto trainer=rpb::make_context_replay_trainer({parent,raw,replay_cp,512,2048})(fit);
      const auto replay_progress=trainer.train_to(512);
      require(replay_progress.completed==512 && replay_progress.attempted==512,"replay skipped updates");
      const auto initial=trainer.snapshot(replay_cp);
      const auto original_train=ev::extract_native_global(initial,train.observed,recipe);
      const auto original_val=ev::extract_native_global(initial,val.observed,recipe);
      feature_parity(original_train,bind(old/"candidate-milestone-512/native-training.pt"));
      feature_parity(original_val,bind(old/"candidate-milestone-512/native-validation.pt"));
      const auto corruption_seed=ev::stream_seed(master,deletion_stream);
      const auto deletion=ev::make_coordinate_deletion_view(val.observed,val.source_ids,ev::Task::lag_sign,
          corruption_seed,0.30,protocol);
      auto deleted=val;deleted.observed=deletion.observations;deleted.clean=deletion.observations;
      const auto deleted_path=(directory/"validation-dropout-030-observations.pt").string();
      save_deleted_validation(deleted_path,val,deletion,corruption_seed);
      const auto oracle=ev::raw_oracle(ev::Task::lag_sign,deleted.observed);
      const auto oracle_score=ev::score(oracle.predictions,val.labels,oracle.valid);
      std::ostringstream view_record;view_record << std::setprecision(17)
        << "{\"view_id\":" << quote(view_id) << ",\"master_seed\":" << master
        << ",\"actual_seed\":" << quote(std::to_string(corruption_seed))
        << ",\"requested_rate\":0.30,\"coordinate_count\":" << val.observed.feature_mask.numel()
        << ",\"original_observed\":" << val.observed.feature_mask.sum().item<int64_t>()
        << ",\"requested_erasure\":" << deletion.requested_erasure.sum().item<int64_t>()
        << ",\"remaining_observed\":" << deleted.observed.feature_mask.sum().item<int64_t>()
        << ",\"oracle\":{\"supported\":" << (oracle_score.supported?"true":"false")
        << ",\"total\":" << oracle_score.total << ",\"valid\":" << oracle_score.valid
        << ",\"correct\":" << oracle_score.correct << ",\"accuracy\":" << oracle_score.accuracy
        << ",\"coverage\":" << oracle_score.coverage << "}}\n";
      write_new(directory/"validation-dropout-030.json",view_record.str());
      for(const int64_t budget:{512,1024,2048}) {
        const auto point=directory/("milestone-"+std::to_string(budget));
        if(budget!=512)require(fs::create_directory(point),"point output exists");
        const auto progress=trainer.train_to(budget);
        require(progress.completed==budget && progress.attempted==budget &&
            progress.parameter_count==225805 && progress.cuda_parameter_count==225805 &&
            progress.training_device=="cuda" && progress.last_input_cuda && progress.last_loss_cuda &&
            progress.finite_gradients && progress.sampled_rows==budget*8,"budget or actual CUDA contract differs");
        const auto cp=(point/"checkpoint.pt").string();trainer.save_checkpoint(cp);
        const auto snapshot=trainer.snapshot(cp);
        const auto training=ev::extract_native_global(snapshot,train.observed,recipe);
        const auto validation=ev::extract_native_global(snapshot,val.observed,recipe);
        const auto drop_features=ev::extract_native_global(snapshot,deleted.observed,recipe);
        require(training.provenance==validation.provenance && training.provenance==drop_features.provenance,
            "feature producer lineage differs");
        const auto tf=(point/"native-training.pt").string(),vf=(point/"native-validation.pt").string();
        const auto df=(point/"native-validation-dropout-030.pt").string();
        ev::save_native_feature_archive(tf,training,train,recipe);
        ev::save_native_feature_archive(vf,validation,val,recipe);
        ev::save_native_feature_archive(df,drop_features,deleted,recipe);
        const auto te=ev::write_native_patch_reconstruction((point/"training-reconstruction.pt").string(),train,snapshot,recipe);
        const auto ve=ev::write_native_patch_reconstruction((point/"validation-reconstruction.pt").string(),val,snapshot,recipe);
        if(budget==512) {
          feature_parity(training,bind(old/"candidate-milestone-512/native-training.pt"));
          feature_parity(validation,bind(old/"candidate-milestone-512/native-validation.pt"));
          query_parity((point/"training-reconstruction.pt").string(),bind(old/"candidate-training-reconstruction.pt"));
          query_parity((point/"validation-reconstruction.pt").string(),bind(old/"candidate-validation-reconstruction.pt"));
          write_new(point/"original512-parity.json","{\"native_training_exact\":true,\"native_validation_exact\":true,\"fixed_queries_exact\":true,\"planned_readout_repetitions\":3}\n");
        }
        const auto json="{\"master_seed\":"+std::to_string(master)+",\"completed_updates\":"+std::to_string(budget)+
            ",\"progress\":"+progress_json(progress)+",\"training_reconstruction\":"+te+
            ",\"validation_reconstruction\":"+ve+'}';
        write_new(point/"point.json",json+'\n');if(!first)report << ',';first=false;report << json;
        ev::ArchiveReadoutInput input;input.id="seed-"+std::to_string(master)+"-updates-"+std::to_string(budget);
        input.tag="RPB-v6";input.task="lag_sign";input.master_seed=master;input.checkpoint_steps=budget;
        input.producer_source_fingerprint=EVALUATION_SOURCE_ID;
        input.cohort_provenance="context-deletion-v1/KTbte3 retained TRAIN/VALIDATION; " + protocol;
        input.training_observations=train_path;input.validation_observations=val_path;
        input.training_features=tf;input.validation_features=vf;
        input.training_observations_sha256=expected.at(train_path);
        input.validation_observations_sha256=expected.at(val_path);input.expected_feature_provenance=training.provenance;
        ev::ArchiveReadoutValidationView view;view.id=view_id;view.observations=deleted_path;view.features=df;
        view.expected_feature_provenance=training.provenance;
        view.corruption_provenance=protocol+"; rate=0.30; stream="+std::to_string(deletion_stream)+
            "; actual_seed="+std::to_string(corruption_seed)+"; pair_shared=true; repair=false";
        input.validation_views.push_back(view);readouts.inputs.push_back(input);
        std::cout << "Master " << master << " completed " << budget
          << " updates; intact/deleted native32 and original fixed-query measurements saved.\n" << std::flush;
      }
      const auto after_train=ev::extract_native_global(initial,train.observed,recipe);
      const auto after_val=ev::extract_native_global(initial,val.observed,recipe);
      feature_parity(after_train,bind(old/"candidate-milestone-512/native-training.pt"));
      feature_parity(after_val,bind(old/"candidate-milestone-512/native-validation.pt"));
      ev::save_native_feature_archive((directory/"after-continuation-original512-native-training.pt").string(),after_train,train,recipe);
      ev::save_native_feature_archive((directory/"after-continuation-original512-native-validation.pt").string(),after_val,val,recipe);
      for(const auto &[name,split]:std::vector<std::pair<std::string,const ev::ControlledDataset *>>{{"training",&train},{"validation",&val}}) {
        const auto witness=directory/("after-continuation-original512-"+name+"-reconstruction.pt");
        write_new(directory/("after-continuation-original512-"+name+"-reconstruction.json"),
            ev::write_native_patch_reconstruction(witness.string(),*split,initial,recipe));
        query_parity(witness.string(),bind(old/("candidate-"+name+"-reconstruction.pt")));
      }
      write_new(directory/"after-continuation-original512-parity.json",
          "{\"native_training_exact\":true,\"native_validation_exact\":true,\"fixed_queries_exact\":true,\"captured_snapshot_immutable\":true}\n");
    }
    report << "]}\n";write_new(output/"continuation-report.json",report.str());
    ev::run_archive_readout(readouts);
    std::cout << "All declared TRAIN/VALIDATION points measured; no TEST, selection or promotion.\n";
    return 0;
  } catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
