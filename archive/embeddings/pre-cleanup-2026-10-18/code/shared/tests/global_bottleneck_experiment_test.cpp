// SPDX-License-Identifier: MIT
#include "embedding/shared/global_bottleneck_experiment.h"
#include "shared_test_support.h"
#include <ATen/Context.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>

namespace ev=embedding::evaluation;
namespace fs=std::filesystem;
namespace {
template<class F> void rejects(F &&function,const std::string &label) {
  bool failed=false;
  try {function();} catch(const std::exception &) {failed=true;}
  test::check(failed,label);
}
std::string read(const fs::path &path) {
  std::ifstream input(path);test::check(bool(input),"missing artifact "+path.string());
  std::ostringstream out;out << input.rdbuf();return out.str();
}
std::string object(const std::string &json,const std::string &marker) {
  const auto location=json.find(marker);test::check(location!=std::string::npos,"missing JSON marker "+marker);
  const auto begin=json.rfind('{',location);
  int depth=0;bool quoted=false,escaped=false;
  for(size_t i=begin;i<json.size();++i) {
    const auto c=json[i];
    if(quoted) {if(escaped) escaped=false;else if(c=='\\') escaped=true;else if(c=='"') quoted=false;}
    else if(c=='"') quoted=true;else if(c=='{') ++depth;
    else if(c=='}' && --depth==0) return json.substr(begin,i-begin+1);
  }
  throw std::runtime_error("unterminated report object");
}
torch::Tensor tensor(const fs::path &path,const std::string &name) {
  torch::serialize::InputArchive input;input.load_from(path.string(),torch::kCPU);
  torch::Tensor out;input.read(name,out,true);return out;
}
ev::GlobalBottleneckRun configuration(const fs::path &output) {
  ev::GlobalBottleneckRun run;
  run.output_directory=output.string();run.source_fingerprint="dummy-global-source";
  run.card.shape={2,32,1,torch::kFloat64,torch::kCPU};run.card.channel_ids={4,9};run.card.feature_units="unitless";
  run.card.train_pairs=4;run.card.validation_pairs=6;run.card.test_pairs=5;
  run.card.matched_global_width=2;run.card.matched_channel_width=2;
  run.card.seeds={41,52};run.card.tasks={ev::Task::level,ev::Task::lag_sign};run.milestones={0,2,4};
  return run;
}
enum class Mode { ordinary,unsupported_global_last,unsupported_global_all,mutable_snapshot,dropped_prefix,nonfinite_reconstruction,
                  lower_validation_late,intact_mismatch };
struct Audit {
  int factories{0},saves{0},snapshots{0},testing_calls{0};
  std::vector<int64_t> requests;
  std::vector<double> draws;
  torch::Tensor expected_rng;
};
using Audits=std::map<std::string,std::shared_ptr<Audit>>;
struct State {ev::CurveProgress progress;std::shared_ptr<Audit> audit;};
ev::NamedCurveFactory factory(const std::string &variant,const ev::GlobalBottleneckRun &run,Mode mode,
                             const std::shared_ptr<Audits> &audits) {
  return {variant,"dummy continuous train-only frozen snapshot recipe",[variant,run,mode,audits](const ev::ProviderFitInput &fit) {
    const fs::path output(run.output_directory);
    test::check(fs::exists(output/"global-bottleneck-card.json") && !fs::exists(output/"selection.json"),
                "factory called without preregistered card or after test selection");
    ev::Task task=ev::Task::lag_sign;
    for(const auto candidate:run.card.tasks)
      if(fit.protocol_id=="global-bottleneck-comparison-v1/"+ev::task_name(candidate)) task=candidate;
    auto expected=ev::make_controlled_development_protocol(task,run.card.shape,
        run.training_reservoir_pairs?run.training_reservoir_pairs:run.card.train_pairs,
        run.training_reservoir_pairs?1:run.card.validation_pairs,fit.seed);
    expected.training.source_ids.resize(2*run.card.train_pairs);
    test::check(fit.training_source_ids==expected.training.source_ids && fit.channel_ids==run.card.channel_ids &&
        torch::equal(fit.training_observations.data,expected.training.observed.data.slice(0,0,2*run.card.train_pairs)) &&
        torch::equal(fit.training_observations.feature_mask,expected.training.observed.feature_mask.slice(0,0,2*run.card.train_pairs)),
        "factory fit received another task, altered data or held-out observations");
    const auto audit_key=std::to_string(fit.seed)+"/"+ev::task_name(task);
    auto audit=std::make_shared<Audit>();++audit->factories;
    test::check(audits->emplace(audit_key,audit).second,"factory reinitialized a continuous path");
    const ev::ObservationScaler scaler(fit.training_observations);
    auto state=std::make_shared<State>();state->audit=audit;
    state->progress.parameter_count=1;state->progress.training_device="cpu";
    state->progress.preprocessing_id="dummy-scaler-"+audit_key;state->progress.training_dataset_id="dummy-rows-"+audit_key;
    torch::manual_seed(fit.seed);
    ev::CurveTrainer trainer;
    trainer.audit_fields={{"train_only","true"},{"task",ev::task_name(task)}};
    trainer.train_to=[state,mode](int64_t budget) {
      auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));
      if(state->audit->expected_rng.defined())
        test::check(torch::equal(generator.get_state(),state->audit->expected_rng),"measurement changed the continuing trainer RNG");
      test::check(budget>=state->progress.completed,"completed update budget regressed");
      state->audit->requests.push_back(budget);
      while(state->progress.completed<budget) {
        state->audit->draws.push_back(torch::rand({1},torch::kFloat64).item<double>());
        ++state->progress.completed;++state->progress.attempted;state->progress.sampled_rows+=2;
        state->progress.losses.push_back({state->progress.attempted,state->progress.completed,48,
            1.0/(state->progress.completed+1),.5});
      }
      state->progress.training_seconds=.001*budget;
      state->progress.finite_gradients=budget>0;state->progress.weights_changed=budget>0;
      state->audit->expected_rng=generator.get_state().clone();
      auto out=state->progress;
      if(mode==Mode::dropped_prefix && budget==4) out.losses.erase(out.losses.begin());
      return out;
    };
    trainer.save_checkpoint=[state,scaler](const std::string &path) {
      ++state->audit->saves;test::check(!fs::exists(path),"checkpoint overwritten");
      torch::serialize::OutputArchive archive;
      archive.write("completed",torch::tensor(state->progress.completed),true);
      archive.write("optimizer_counter",torch::tensor(state->progress.completed),true);
      archive.write("train_mean",scaler.mean,true);archive.write("train_scale",scaler.scale,true);
      embedding::archive::save_archive(path,archive);
    };
    trainer.snapshot=[state,scaler,task,run,mode,seed=fit.seed](const std::string &path) {
      ++state->audit->snapshots;torch::manual_seed(665544);
      const auto budget=tensor(path,"completed").item<int64_t>();
      auto frozen=scaler;frozen.mean=tensor(path,"train_mean");frozen.scale=tensor(path,"train_scale");frozen.counts=scaler.counts.clone();
      ev::CurveSnapshot snapshot;
      snapshot.features.provenance="dummy immutable saved state";
      snapshot.features.surfaces={{"odd_global",{ev::SurfaceKind::global,"any semantic channel observed",{}}},
          {"odd_channels",{ev::SurfaceKind::channel_concatenation,"all semantic channels observed",run.card.channel_ids}}};
      snapshot.features.extract=[state,task,run,mode,budget,seed](const embedding::Batch &batch) {
        const auto current=mode==Mode::mutable_snapshot?state->progress.completed:budget;
        if(batch.data.size(0)==2*run.card.test_pairs) {
          ++state->audit->testing_calls;
          test::check(fs::exists(fs::path(run.output_directory)/"selection.json") &&
              read(fs::path(run.output_directory)/"selection.json").find("\"selected_budget\":"+
                  std::to_string(run.fixed_test_budget?run.fixed_test_budget:2))!=std::string::npos,
              "fresh test data reached provider before frozen validation selection");
        }
        test::check(batch.data.masked_select(batch.feature_mask.logical_not()).eq(0).all().item<bool>(),
                    "hidden values reached feature provider");
        const auto oracle=ev::raw_oracle(task,batch);
        auto sign=oracle.predictions.to(torch::kFloat64)*2-1;
        if((mode==Mode::lower_validation_late && budget==4 && batch.data.size(0)==2*run.card.validation_pairs) ||
           (mode==Mode::intact_mismatch && batch.data.size(0)==2*run.card.test_pairs && state->audit->testing_calls>1))
          sign=-sign;
        const auto channel=batch.data.select(1,0),mask=batch.feature_mask.select(1,0);
        const auto nuisance=channel.flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1).to(torch::kFloat64);
        const auto values=torch::stack({sign*double(current),nuisance},1);
        auto channel_support=batch.feature_mask.flatten(2).any(2);
        auto global_valid=channel_support.any(1),concat_valid=channel_support.all(1);
        // Only a GLOBAL fit fails: a perfect concatenation diagnostic must not
        // rescue the primary budget, nor can a failed master be averaged away.
        if(mode==Mode::unsupported_global_all ||
            (mode==Mode::unsupported_global_last && current==4 && seed==run.card.seeds.back()))
          global_valid=torch::zeros_like(global_valid);
        ev::FeatureMap out{{"odd_global",{torch::where(global_valid.unsqueeze(-1),values,torch::zeros_like(values)),global_valid,"global signal"}},
            {"odd_channels",{torch::where(concat_valid.unsqueeze(-1),values,torch::zeros_like(values)),concat_valid,"channel diagnostics"}}};
        batch.data.fill_(std::numeric_limits<double>::quiet_NaN());batch.feature_mask.fill_(false);torch::manual_seed(443322);
        return out;
      };
      snapshot.features.save_assets=[budget](const std::string &directory) {
        torch::serialize::OutputArchive assets;assets.write("snapshot_budget",torch::tensor(budget),true);
        embedding::archive::save_archive((fs::path(directory)/"dummy.pt").string(),assets);
      };
      snapshot.reconstruct=[run,frozen,budget,mode](const embedding::Batch &batch,const torch::Tensor &hidden) {
        const auto visible=batch.feature_mask.logical_and(hidden.logical_not());
        const auto B=batch.data.size(0),C=run.card.shape.channel_count,K=run.card.shape.history_length/run.patch_length;
        const auto eligible=visible.reshape({B,C,K,-1}).any(-1).sum(-1).ge(2)
            .logical_and(batch.feature_mask.logical_and(hidden).flatten(2).any(2));
        const auto target=frozen.transform(batch).data;
        auto prediction=torch::full_like(target,.1*budget);
        if(mode==Mode::nonfinite_reconstruction) prediction.fill_(std::numeric_limits<double>::quiet_NaN());
        batch.data.fill_(123456);batch.feature_mask.fill_(false);hidden.fill_(true);torch::rand({1});
        return ev::CurveReconstruction{prediction,target,eligible};
      };
      return snapshot;
    };
    // Deliberately attempt to mutate the canonical training cohort through fit.
    fit.training_observations.data.fill_(98765);fit.training_observations.feature_mask.fill_(false);
    return trainer;
  }};
}
void ordinary(const fs::path &temporary) {
  const auto run=configuration(temporary/"ordinary");
  std::vector<std::shared_ptr<Audits>> audits(3);
  std::vector<ev::NamedCurveFactory> factories;
  const std::vector<std::string> names{"current_mixer","mean_global","learned_global"};
  for(size_t i=0;i<names.size();++i) {audits[i]=std::make_shared<Audits>();factories.push_back(factory(names[i],run,Mode::ordinary,audits[i]));}
  ev::run_global_bottleneck_experiment(run,factories);
  const fs::path output(run.output_directory);
  const auto selection=read(output/"selection.json"),report=read(output/"report.json"),validation=read(output/"validation-report.json");
  test::check(selection.find("\"selected_budget\":2")!=std::string::npos &&
      selection.find("matched-global")!=std::string::npos && selection.find("\"budget\":0")==std::string::npos &&
      object(selection,"\"budget\":2").find("\"utility\":1")!=std::string::npos &&
      object(selection,"\"budget\":4").find("\"utility\":1")!=std::string::npos,
      "global primary/equal weights/exact tie/zero exclusion changed");
  test::check(report.find("\"task\":\"level\"")!=std::string::npos && report.find("\"task\":\"lag_sign\"")!=std::string::npos &&
      report.find("\"fitted_readouts_reused\":true")!=std::string::npos &&
      report.find("ridge_candidate_minus_comparator_grouped_interval")!=std::string::npos &&
      report.find("full_population_correctness")!=std::string::npos && validation.find("selected-test")==std::string::npos,
      "multitask report lost selection/test separation, support or paired evidence");
  const auto stress=read(output/"stress-report.json");
  test::check(stress.find("random_dropout_030")!=std::string::npos && stress.find("all_absent")!=std::string::npos &&
      stress.find("channel_id_4_absent")!=std::string::npos && stress.find("conditional_accuracy")!=std::string::npos,
      "selected fixed-readout stress was omitted");
  for(const auto seed:run.card.seeds) for(const auto task:run.card.tasks) {
    const auto key=std::to_string(seed)+"/"+ev::task_name(task);
    for(size_t i=0;i<audits.size();++i) {
      const auto a=audits[i]->at(key);
      test::check(a->requests==std::vector<int64_t>({0,2,4}) && a->factories==1 && a->saves==3 && a->snapshots==3 &&
          a->testing_calls>1 && a->draws==audits[0]->at(key)->draws,
          "continuous trainer was reset, measurement RNG changed or stress did not use selected provider");
      const auto directory=output/("seed-"+std::to_string(seed)+"-"+ev::task_name(task));
      const auto point=directory/names[i]/"milestone-2";
      test::check(tensor(point/"checkpoint.pt","completed").item<int64_t>()==2 &&
          tensor(point/"checkpoint.pt","optimizer_counter").item<int64_t>()==2 &&
          tensor(directory/names[i]/"milestone-4"/"checkpoint.pt","completed").item<int64_t>()==4,
          "chosen checkpoint was replaced with final path");
      const auto prefix=names[i]+"_global";
      const auto intact=directory/"stress"/("intact-"+prefix+"-matched_global-predictions.pt");
      const auto ordinary_prediction=point/(prefix+"-matched_global-selected-test-predictions.pt");
      test::check(torch::equal(tensor(intact,"prediction_valid"),tensor(intact,"ordinary_prediction_valid")) &&
          torch::equal(tensor(intact,"ridge"),tensor(intact,"ordinary_ridge")) &&
          torch::equal(tensor(intact,"tiny"),tensor(intact,"ordinary_tiny")) &&
          torch::equal(tensor(intact,"ridge"),tensor(ordinary_prediction,"ridge")) &&
          torch::equal(tensor(intact,"tiny"),tensor(ordinary_prediction,"tiny_secondary")),
          "intact stress readout differs from exact ordinary selected predictions");
      test::check(tensor(point/(prefix+"-selected-testing.pt"),"features").select(1,0).abs().eq(2).all().item<bool>(),
          "final test used a later mutable snapshot");
      const auto fitted=tensor(point/(prefix+"-training.pt"),"features");
      test::close(tensor(point/(prefix+"-matched_global-fit.pt"),"feature_mean"),fitted.mean(0),
          "normalizer included held-out rows",0,0);
      test::check(tensor(point/(prefix+"-matched_global-fit.pt"),"fitted_rows").item<int64_t>()==2*run.card.train_pairs,
          "readout refitted on validation/test observations");
      const auto query=tensor(point/"validation-reconstruction.pt","target_mask");
      test::check(torch::equal(query.sum(0),tensor(directory/"controlled-validation.pt","feature_mask").to(torch::kInt64)),
          "fixed patch queries changed observed target coverage");
    }
    const auto directory=output/("seed-"+std::to_string(seed)+"-"+ev::task_name(task));
    const auto expected=ev::make_controlled_development_protocol(task,run.card.shape,4,6,seed);
    test::close(tensor(directory/"controlled-training.pt","observed"),expected.training.observed.data,"mutated canonical training cohort",0,0);
    const auto fresh=ev::make_controlled_test_dataset(task,run.card.shape,5,ev::stream_seed(seed,run.fresh_test_stream));
    test::close(tensor(directory/"controlled-selected-testing.pt","observed"),fresh.observed.data,"wrong fresh final stream",0,0);
    const auto absent=object(read(directory/"stress"/"report.json"),"\"case\":{\"id\":\"all_absent\"");
    test::check(absent.find("\"coverage\":0")!=std::string::npos && absent.find("\"conditional_accuracy\":null")!=std::string::npos,
        "allmissing signal support fabricated");
  }
  rejects([&]{ev::run_global_bottleneck_experiment(run,factories);},"existing output overwritten");
}
void failures(const fs::path &temporary) {
  for(const auto mode:{Mode::unsupported_global_last,Mode::unsupported_global_all,Mode::mutable_snapshot,Mode::dropped_prefix,Mode::nonfinite_reconstruction}) {
    auto run=configuration(temporary/("failure-"+std::to_string(static_cast<int>(mode))));run.stress_sweep=false;run.card.tasks={ev::Task::lag_sign};
    std::vector<ev::NamedCurveFactory> factories;
    for(const auto &name:std::vector<std::string>{"current_mixer","mean_global","learned_global"})
      factories.push_back(factory(name,run,name=="learned_global"?mode:Mode::ordinary,std::make_shared<Audits>()));
    if(mode==Mode::unsupported_global_last) {
      ev::run_global_bottleneck_experiment(run,factories);
      const auto late=object(read(fs::path(run.output_directory)/"selection.json"),"\"budget\":4");
      test::check(late.find("\"status\":\"unsupported_common_budget\"")!=std::string::npos && late.find("\"utility\":null")!=std::string::npos,
          "one global failure was rescued by concatenation/another variant/master");
    } else {
      rejects([&]{ev::run_global_bottleneck_experiment(run,factories);},"failed contract was measured");
      for(const auto &entry:fs::recursive_directory_iterator(run.output_directory))
        test::check(entry.path().filename()!="controlled-selected-testing.pt","test generated before failure/selection validation");
    }
  }
  auto invalid=configuration(temporary/"invalid");invalid.card.tasks={ev::Task::level};
  std::vector<ev::NamedCurveFactory> factories;
  for(const auto &name:std::vector<std::string>{"current_mixer","mean_global","learned_global"})
    factories.push_back(factory(name,invalid,Mode::ordinary,std::make_shared<Audits>()));
  rejects([&]{ev::run_global_bottleneck_experiment(invalid,factories);},"missing selection task accepted");
  test::check(!fs::exists(invalid.output_directory),"invalid frozen card generated observations");
}
void named_validation_and_fixed_budget(const fs::path &temporary) {
  std::vector<fs::path> directories;
  for(const auto pairs:{int64_t(4),int64_t(8)}) {
    auto run=configuration(temporary/("named-validation-"+std::to_string(pairs)));
    run.card.tasks={ev::Task::lag_sign};run.card.seeds={41};run.card.train_pairs=pairs;
    run.validation_seed_stream=929292;run.training_reservoir_pairs=8;run.fixed_test_budget=4;run.stress_sweep=false;
    std::vector<ev::NamedCurveFactory> factories;
    for(const auto &name:std::vector<std::string>{"current_mixer","mean_global","learned_global"})
      factories.push_back(factory(name,run,Mode::lower_validation_late,std::make_shared<Audits>()));
    ev::run_global_bottleneck_experiment(run,factories);
    const auto output=fs::path(run.output_directory);
    directories.push_back(output/"seed-41-lag_sign");
    const auto selection=read(output/"selection.json");
    test::check(selection.find("\"selected_budget\":4")!=std::string::npos &&
        selection.find("declared fixed supported nonzero budget; no validation budget search")!=std::string::npos &&
        object(selection,"\"budget\":2").find("\"utility\":1")!=std::string::npos &&
        object(selection,"\"budget\":4").find("\"utility\":0")!=std::string::npos,
        "fixed budget was replaced by higher validation utility");
    const auto expected=ev::make_controlled_development_protocol(ev::Task::lag_sign,run.card.shape,1,
        run.card.validation_pairs,ev::stream_seed(41,run.validation_seed_stream));
    test::close(tensor(directories.back()/"controlled-validation.pt","observed"),expected.validation.observed.data,
        "named validation drew another cohort",0,0);
    test::check(torch::equal(tensor(directories.back()/"controlled-validation.pt","feature_mask"),expected.validation.observed.feature_mask),
        "named validation masks differ");
    test::check(read(output/"global-bottleneck-card.json").find("\"validation_seed_stream\":\"929292\"")!=std::string::npos &&
        read(directories.back()/"development-manifest.json").find("\"validation_seed\":\""+
            std::to_string(ev::stream_seed(41,run.validation_seed_stream))+"\"")!=std::string::npos,
        "named validation or fixed-budget policy absent from frozen provenance");
  }
  for(const auto &name:{"observed","feature_mask","labels_scoring_only","source_ids_json"})
    test::check(torch::equal(tensor(directories[0]/"controlled-validation.pt",name),tensor(directories[1]/"controlled-validation.pt",name)),
        "validation changed with training-pair count");
  test::check(object(read(directories[0]/"development-manifest.json"),"\"split\":\"validation\"")==
      object(read(directories[1]/"development-manifest.json"),"\"split\":\"validation\""),
      "named validation source manifest changed with training-pair count");
  for(const auto &name:{"observed","feature_mask","labels_scoring_only"})
    test::check(torch::equal(tensor(directories[0]/"controlled-training.pt",name),
        tensor(directories[1]/"controlled-training.pt",name).slice(0,0,8)),"training reservoir cohorts are not exact nested prefixes");
  const auto reservoir=ev::make_controlled_development_protocol(ev::Task::lag_sign,
      configuration(temporary/"unused").card.shape,8,1,41);
  auto small_ids=reservoir.training.source_ids;small_ids.resize(8);
  std::ostringstream expected_ids;expected_ids << '[';
  for(size_t i=0;i<small_ids.size();++i) {if(i)expected_ids << ',';expected_ids << '"' << small_ids[i] << '"';}
  expected_ids << ']';
  test::check(torch::equal(tensor(directories[0]/"controlled-training.pt","source_ids_json"),embedding::archive::text_tensor(expected_ids.str())),
      "nested source identities differ from exact reservoir prefix");
  std::set<std::string> all_roles;
  const auto named=ev::make_controlled_development_protocol(ev::Task::lag_sign,
      configuration(temporary/"unused").card.shape,1,6,ev::stream_seed(41,929292));
  const auto testing=ev::make_controlled_test_dataset(ev::Task::lag_sign,
      configuration(temporary/"unused").card.shape,5,ev::stream_seed(41,configuration(temporary/"unused").fresh_test_stream));
  for(const auto *ids:{&reservoir.training.source_ids,&named.validation.source_ids,&testing.source_ids}) {
    const std::set<std::string> unique(ids->begin(),ids->end());
    for(const auto &id:unique)test::check(all_roles.insert(id).second,"training/validation/testing roles overlap across diversity matrix");
  }
  auto unsupported=configuration(temporary/"fixed-budget-unsupported");
  unsupported.card.tasks={ev::Task::lag_sign};unsupported.fixed_test_budget=4;unsupported.stress_sweep=false;
  std::vector<ev::NamedCurveFactory> factories;
  for(const auto &name:std::vector<std::string>{"current_mixer","mean_global","learned_global"})
    factories.push_back(factory(name,unsupported,name=="learned_global"?Mode::unsupported_global_last:Mode::ordinary,std::make_shared<Audits>()));
  rejects([&]{ev::run_global_bottleneck_experiment(unsupported,factories);},"unsupported fixed budget fell back to another milestone");
  test::check(!fs::exists(fs::path(unsupported.output_directory)/"selection.json"),"unsupported fixed budget persisted selection");
  auto invalid=configuration(temporary/"fixed-budget-undeclared");invalid.fixed_test_budget=3;
  rejects([&]{ev::run_global_bottleneck_experiment(invalid,{});},"undeclared fixed budget accepted");
  test::check(!fs::exists(invalid.output_directory),"invalid fixed budget generated data");
  invalid=configuration(temporary/"reservoir-without-validation");invalid.training_reservoir_pairs=8;
  rejects([&]{ev::run_global_bottleneck_experiment(invalid,{});},"reservoir without independent validation accepted");
  test::check(!fs::exists(invalid.output_directory),"invalid reservoir generated data");
}
void intact_guard(const fs::path &temporary) {
  auto run=configuration(temporary/"intact-mismatch");run.card.tasks={ev::Task::lag_sign};run.card.seeds={41};
  std::vector<ev::NamedCurveFactory> factories;
  for(const auto &name:std::vector<std::string>{"current_mixer","mean_global","learned_global"})
    factories.push_back(factory(name,run,name=="learned_global"?Mode::intact_mismatch:Mode::ordinary,std::make_shared<Audits>()));
  bool detected=false;
  try {ev::run_global_bottleneck_experiment(run,factories);}
  catch(const std::exception &error) {detected=std::string(error.what()).find("intact stress predictions differ")!=std::string::npos;}
  test::check(detected,"stochastic or test-specific intact provider mutation was accepted");
}
} // namespace
int main() {
  try {
    const auto directory=fs::temp_directory_path()/("embedding-global-bottleneck-test-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);ordinary(directory);failures(directory);
    named_validation_and_fixed_budget(directory);intact_guard(directory);
    std::cout << "shared global bottleneck tests passed; artifacts=" << directory << '\n';
    return 0;
  } catch(const std::exception &error) {std::cerr << error.what() << '\n';return 1;}
}
