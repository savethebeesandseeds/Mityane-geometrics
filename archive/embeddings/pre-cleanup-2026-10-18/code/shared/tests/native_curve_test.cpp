// SPDX-License-Identifier: MIT
#include "embedding/shared/native_curve.h"
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
ev::NativeCurveRun configuration(const fs::path &output) {
  ev::NativeCurveRun run;
  run.output_directory=output.string();run.source_fingerprint="dummy-global-source";
  run.card.shape={2,32,1,torch::kFloat64,torch::kCPU};run.card.channel_ids={4,9};run.card.feature_units="unitless";
  run.card.train_pairs=4;run.card.validation_pairs=6;run.card.test_pairs=5;
  run.card.matched_global_width=2;run.card.matched_channel_width=2;
  run.card.seeds={41,52};run.card.tasks={ev::Task::level,ev::Task::lag_sign};run.milestones={0,2,4};
  run.export_width=2;run.sanity_budget=2;run.repetitions={{"rep-1",2701},{"rep-2",2802}};
  run.tiny_hidden=4;run.tiny_steps=5;run.bootstrap_replicates=100;
  return run;
}
enum class Mode { ordinary,unsupported_global_last,unsupported_global_all,mutable_snapshot,dropped_prefix,nonfinite_reconstruction,
                  lower_validation_late,intact_mismatch,mutable_reconstruction,missing_endpoint,unsupported_train_last,
                  unsupported_validation_all,mutable_middle_snapshot,mutable_middle_reconstruction };
struct Audit {
  int factories{0},saves{0},snapshots{0},testing_calls{0};
  std::vector<int64_t> requests;
  std::vector<double> draws;
  torch::Tensor expected_rng;
  std::string protocol_id;
};
using Audits=std::map<std::string,std::shared_ptr<Audit>>;
struct State {ev::CurveProgress progress;std::shared_ptr<Audit> audit;};
ev::NamedCurveFactory factory(const std::string &variant,const ev::NativeCurveRun &run,Mode mode,
                             const std::shared_ptr<Audits> &audits) {
  return {variant,"dummy continuous train-only frozen snapshot recipe",[variant,run,mode,audits](const ev::ProviderFitInput &fit) {
    const fs::path output(run.output_directory);
    test::check(fs::exists(output/(run.development_only?"native-development-card.json":"native-curve-card.json")) && !fs::exists(output/"selection.json"),
                "factory called without preregistered card or after test selection");
    ev::Task task=ev::Task::lag_sign;
    const std::string protocol=run.development_only?"native-development-v1":"native-curve-v1";
    bool matched=false;
    for(const auto candidate:run.card.tasks)
      if(fit.protocol_id==protocol+"/"+ev::task_name(candidate)){task=candidate;matched=true;}
    test::check(matched,"fit metadata used the wrong training protocol namespace");
    auto expected=ev::make_controlled_development_protocol(task,run.card.shape,
        run.card.train_pairs,
        run.card.validation_pairs,fit.seed);
    expected.training.source_ids.resize(2*run.card.train_pairs);
    test::check(fit.training_source_ids==expected.training.source_ids && fit.channel_ids==run.card.channel_ids &&
        torch::equal(fit.training_observations.data,expected.training.observed.data.slice(0,0,2*run.card.train_pairs)) &&
        torch::equal(fit.training_observations.feature_mask,expected.training.observed.feature_mask.slice(0,0,2*run.card.train_pairs)),
        "factory fit received another task, altered data or held-out observations");
    const auto audit_key=std::to_string(fit.seed)+"/"+ev::task_name(task);
    auto audit=std::make_shared<Audit>();++audit->factories;audit->protocol_id=fit.protocol_id;
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
      if(mode==Mode::missing_endpoint && budget==4) out.losses.pop_back();
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
        const auto current=mode==Mode::mutable_snapshot || (mode==Mode::mutable_middle_snapshot && budget==2)?state->progress.completed:budget;
        if(batch.data.size(0)==2*run.card.test_pairs) {
          ++state->audit->testing_calls;
          test::check(state->audit->requests.back()==(task==run.selection_task?run.milestones.back():run.sanity_budget),"TEST reached a path before its training finished");
          test::check(fs::exists(fs::path(run.output_directory)/"selection.json") &&
              read(fs::path(run.output_directory)/"selection.json").find("\"selected_budget\":"+
                  std::to_string(2))!=std::string::npos,
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
        auto values=torch::stack({sign*double(current),nuisance},1);
        if(run.export_width>2)values=torch::cat({values,torch::zeros({values.size(0),run.export_width-2},values.options())},1);
        auto channel_support=batch.feature_mask.flatten(2).any(2);
        auto global_valid=channel_support.any(1),concat_valid=channel_support.all(1);
        // Only a GLOBAL fit fails: a perfect concatenation diagnostic must not
        // rescue the primary budget, nor can a failed master be averaged away.
        if(mode==Mode::unsupported_global_all ||
            (mode==Mode::unsupported_validation_all && batch.data.size(0)==2*run.card.validation_pairs) ||
            (mode==Mode::unsupported_global_last && current==4 && seed==run.card.seeds.back()) ||
            (mode==Mode::unsupported_train_last && current==4 && seed==run.card.seeds.back() && batch.data.size(0)==2*run.card.train_pairs))
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
      snapshot.reconstruct=[run,frozen,budget,mode,state](const embedding::Batch &batch,const torch::Tensor &hidden) {
        const auto visible=batch.feature_mask.logical_and(hidden.logical_not());
        const auto B=batch.data.size(0),C=run.card.shape.channel_count,K=run.card.shape.history_length/run.patch_length;
        const auto eligible=visible.reshape({B,C,K,-1}).any(-1).sum(-1).ge(2)
            .logical_and(batch.feature_mask.logical_and(hidden).flatten(2).any(2));
        const auto target=frozen.transform(batch).data;
        const auto current=mode==Mode::mutable_reconstruction || (mode==Mode::mutable_middle_reconstruction && budget==2)?state->progress.completed:budget;
        auto prediction=torch::full_like(target,.1*current);
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
  const auto run=configuration(temporary/"ordinary");auto audits=std::make_shared<Audits>();
  const auto trainer=factory("rpb_v4",run,Mode::ordinary,audits);
  torch::manual_seed(8675309);const auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto ambient=generator.get_state().clone();const auto threads=at::get_num_threads();
  ev::run_native_curve(run,trainer);
  test::check(torch::equal(generator.get_state(),ambient) && at::get_num_threads()==threads,"ambient RNG/threads not restored");
  const fs::path output(run.output_directory);const auto selection=read(output/"selection.json");
  test::check(selection.find("\"selected_budget\":2")!=std::string::npos && selection.find("fsync")!=std::string::npos &&
      selection.find("\"budget\":0")==std::string::npos && object(selection,"\"budget\":2").find("\"utility\":1")!=std::string::npos &&
      object(selection,"\"budget\":4").find("\"utility\":1")!=std::string::npos,"native ridge selection, smaller exact tie or zero exclusion changed");
  const auto card=read(output/"native-curve-card.json"),report=read(output/"report.json"),validation=read(output/"validation-report.json");
  test::check(!run.development_only && card.find("\"protocol\":\"native-curve-v1\"")!=std::string::npos &&
      card.find("development_only")==std::string::npos && !fs::exists(output/"development-complete.json") &&
      !fs::exists(output/"native-development-card.json"),"default full-curve output acquired development-only schema");
  test::check(card.find("\"policy_version\":\"1.2\"")!=std::string::npos && card.find("\"post_encoder_pca\":false")!=std::string::npos &&
      report.find("\"fitted_readouts_reused\":true")!=std::string::npos && report.find("ridge_candidate_minus_comparator_grouped_interval")!=std::string::npos &&
      report.find("full_population_correctness")!=std::string::npos && validation.find("selected-testing")==std::string::npos,
      "native protocol/report lost noPCA, retained fits, paired evidence or split separation");
  for(const auto seed:run.card.seeds)for(const auto task:run.card.tasks) {
    const auto audit=audits->at(std::to_string(seed)+"/"+ev::task_name(task));
    const auto budgets=task==run.selection_task?std::vector<int64_t>{0,2,4}:std::vector<int64_t>{0,2};
    test::check(audit->requests==budgets && audit->factories==1 && audit->saves==int(budgets.size()) &&
        audit->snapshots==int(budgets.size()) && audit->testing_calls>2,"trainer reinitialized or selected/init snapshots were not stressed");
    const auto directory=output/("seed-"+std::to_string(seed)+"-"+ev::task_name(task));const auto point=directory/"milestone-2";
    test::check(tensor(point/"checkpoint.pt","completed").item<int64_t>()==2 &&
        tensor(point/"checkpoint.pt","optimizer_counter").item<int64_t>()==2,"selected ordinary checkpoint was replaced by final weights");
    if(task==run.selection_task)test::check(tensor(directory/"milestone-4"/"checkpoint.pt","completed").item<int64_t>()==4,"timing path stopped after selected budget");
    const auto expected=ev::make_controlled_development_protocol(task,run.card.shape,run.card.train_pairs,run.card.validation_pairs,seed);
    test::close(tensor(directory/"controlled-training.pt","observed"),expected.training.observed.data,"provider mutated canonical TRAIN",0,0);
    test::check(!expected.testing.observed.data.defined(),"development generation created TEST");
    const auto fresh=ev::make_controlled_test_dataset(task,run.card.shape,run.card.test_pairs,ev::stream_seed(seed,run.fresh_test_stream));
    test::close(tensor(directory/"controlled-selected-testing.pt","observed"),fresh.observed.data,"fresh TEST stream changed",0,0);
    const auto witness=tensor(point/"validation-reconstruction-witness.pt","standardized_prediction");
    test::check(torch::equal(witness,tensor(point/"validation-reconstruction.pt","standardized_prediction")) &&
        read(point/"witness-audit.json").find("\"performed_before_all_testing\":true")!=std::string::npos,"selected reconstruction snapshot changed during later training");
    test::check(torch::equal(tensor(directory/"milestone-0"/"validation-reconstruction-witness.pt","standardized_prediction"),
        tensor(directory/"milestone-0"/"validation-reconstruction.pt","standardized_prediction")),"point0 reconstruction no longer same initial path");
    const auto query=tensor(point/"validation-reconstruction.pt","target_mask");
    test::check(torch::equal(query.sum(0),tensor(directory/"controlled-validation.pt","feature_mask").to(torch::kInt64)),"fixed reconstruction patch queries lost observed targets");
    const auto controls=read(directory/"controls"/"controls.json");
    test::check(controls.find("\"observation_scaler_train_fits\":1")!=std::string::npos,"raw scaler was refitted per checkpoint");
    for(const auto &name:{"raw","pca_only","mask_metadata"}) {
      const auto count=read(directory/"controls"/name/"fit-counts.json");
      test::check(count.find("\"outer_train_normalizer_fits\":1")!=std::string::npos && count.find("\"ridge_fits\":2")!=std::string::npos &&
          count.find("\"tiny_fits\":2")!=std::string::npos && count.find("\"validation_test_stress_fits\":0")!=std::string::npos,
          "fixed control transforms/heads were not fitted once per recipe");
      test::check(count.find(std::string("\"standalone_pca_fits\":")+(std::string(name)=="pca_only"?"1":"0"))!=std::string::npos,"PCA control map refitted across head repetitions");
    }
    for(const auto &rep:run.repetitions) {
      const auto fit=point/rep.id/"fit.pt";
      test::close(tensor(fit,"feature_mean"),tensor(point/"native-training.pt","features").mean(0),"native normalization included heldout rows",0,0);
      test::check(tensor(fit,"fitted_rows").item<int64_t>()==2*run.card.train_pairs,"native head fit used heldout rows");
      torch::serialize::InputArchive archive;archive.load_from(fit.string(),torch::kCPU);torch::Tensor ignored;
      test::check(!archive.try_read("pca_components",ignored,true),"native checkpoint readout has postencoder PCA");
      test::check(torch::equal(tensor(fit,"actual_probe_seed_decimal"),tensor(directory/"controls"/"pca_only"/rep.id/"fit.pt","actual_probe_seed_decimal")),"equal width PCA/native head seeds not paired");
      const auto stress=directory/("testing-"+rep.id)/"stress";
      const auto intact=stress/"intact-native-native-predictions.pt",ordinary_predictions=directory/("testing-"+rep.id)/"native-predictions.pt";
      test::check(torch::equal(tensor(intact,"prediction_valid"),tensor(ordinary_predictions,"valid")) &&
          torch::equal(tensor(intact,"ridge"),tensor(ordinary_predictions,"ridge")) &&
          torch::equal(tensor(intact,"tiny"),tensor(ordinary_predictions,"tiny_secondary")),"intact stress differs from frozen native TEST prediction");
      for(const auto &name:{"raw","pca_only","native","untrained_native"})
        test::check(!tensor(stress/(std::string("all_absent-")+name+"-native-predictions.pt"),"prediction_valid").any().item<bool>(),"allmissing signal coverage fabricated");
      test::check(tensor(stress/"all_absent-mask_metadata-native-predictions.pt","prediction_valid").all().item<bool>(),"mask control wrongly abstains on allmissing");
      const auto stress_text=read(stress/"report.json");
      test::check(stress_text.find("random_dropout_030")!=std::string::npos && stress_text.find("channel_id_4_absent")!=std::string::npos &&
          stress_text.find("\"conditional_accuracy\":null")!=std::string::npos,"fixed stress cases or zero-support scoring missing");
    }
  }
  rejects([&]{ev::run_native_curve(run,trainer);},"existing output root overwritten");
}
void failures(const fs::path &temporary) {
  const std::vector<std::pair<std::string,Mode>> cases{{"live-native",Mode::mutable_snapshot},{"live-reconstruction",Mode::mutable_reconstruction},
      {"changed-support",Mode::unsupported_global_last},{"unsupported-all",Mode::unsupported_global_all},
      {"changed-prefix",Mode::dropped_prefix},{"missing-endpoint",Mode::missing_endpoint},{"nonfinite-query",Mode::nonfinite_reconstruction}};
  for(const auto &[name,mode]:cases) {
    auto run=configuration(temporary/name);run.card.tasks={ev::Task::lag_sign};run.card.seeds={41};run.repetitions={{"rep-1",2701}};run.stress_sweep=false;
    auto audits=std::make_shared<Audits>();const auto trainer=factory("rpb_v4",run,mode,audits);
    torch::manual_seed(4422);const auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));const auto ambient=generator.get_state().clone();
    rejects([&]{ev::run_native_curve(run,trainer);},name+" snapshot/progress/support violation accepted");
    test::check(torch::equal(generator.get_state(),ambient),"exception did not restore ambient RNG");
    for(const auto &[key,audit]:*audits){(void)key;test::check(audit->testing_calls==0,"invalid checkpoint reached TEST before witness rejection");}
    test::check(!fs::exists(fs::path(run.output_directory)/"seed-41-lag_sign"/"controlled-selected-testing.pt"),"invalid retained checkpoint generated TEST");
  }
  auto invalid=configuration(temporary/"invalid-recipe");invalid.milestones={0,4,2};
  rejects([&]{ev::run_native_curve(invalid,factory("rpb_v4",invalid,Mode::ordinary,std::make_shared<Audits>()));},"unordered selection budgets accepted");
  test::check(!fs::exists(invalid.output_directory),"invalid card generated observations/artifacts");
}
void validation_and_intact(const fs::path &temporary) {
  auto run=configuration(temporary/"validation-choice");run.card.tasks={ev::Task::lag_sign};run.card.seeds={41};run.repetitions={{"rep-1",2701}};run.stress_sweep=false;
  ev::run_native_curve(run,factory("rpb_v4",run,Mode::lower_validation_late,std::make_shared<Audits>()));
  const auto selection=read(fs::path(run.output_directory)/"selection.json");
  test::check(selection.find("\"selected_budget\":2")!=std::string::npos && object(selection,"\"budget\":4").find("\"utility\":0")!=std::string::npos,"selection used final checkpoint or non-validation objective");
  run.output_directory=(temporary/"intact-mismatch").string();run.stress_sweep=true;
  bool detected=false;
  try{ev::run_native_curve(run,factory("rpb_v4",run,Mode::intact_mismatch,std::make_shared<Audits>()));}
  catch(const std::exception &error){detected=std::string(error.what()).find("intact stress predictions differ")!=std::string::npos;}
  test::check(detected,"stochastic/test-specific stress provider changed intact predictions without rejection");
}
void unsupported_fits(const fs::path &temporary) {
  auto run=configuration(temporary/"unsupported-one-master-fit");run.card.tasks={ev::Task::lag_sign};run.repetitions={{"rep-1",2701}};run.stress_sweep=false;
  ev::run_native_curve(run,factory("rpb_v4",run,Mode::unsupported_train_last,std::make_shared<Audits>()));
  const auto selection=read(fs::path(run.output_directory)/"selection.json");
  test::check(selection.find("\"selected_budget\":2")!=std::string::npos &&
      object(selection,"\"budget\":4").find("\"status\":\"unsupported_common_budget\"")!=std::string::npos &&
      object(selection,"\"budget\":4").find("\"utility\":null")!=std::string::npos,
      "unsupported TRAIN fit for one master was averaged away or rescued by another head");
  const auto last=fs::path(run.output_directory)/"seed-52-lag_sign";
  test::check(torch::equal(tensor(last/"milestone-0"/"native-validation.pt","valid"),tensor(last/"milestone-4"/"native-validation.pt","valid")),
      "unsupported TRAIN-fit test changed VALIDATION support");
  run=configuration(temporary/"unsupported-standalone-pca");run.card.tasks={ev::Task::lag_sign};run.card.seeds={41};run.repetitions={{"rep-1",2701}};
  run.export_width=16;run.stress_sweep=false;
  ev::run_native_curve(run,factory("rpb_v4",run,Mode::ordinary,std::make_shared<Audits>()));
  const auto directory=fs::path(run.output_directory)/"seed-41-lag_sign";
  const auto counts=read(directory/"controls"/"pca_only"/"fit-counts.json");
  test::check(counts.find("\"standalone_pca_attempts\":1")!=std::string::npos && counts.find("\"standalone_pca_fits\":0")!=std::string::npos &&
      counts.find("\"ridge_fits\":0")!=std::string::npos && !fs::exists(directory/"controls"/"pca_only"/"rep-1"/"fit.pt"),
      "insufficient standalone PCA rank silently changed dimensions or fitted a head");
  const auto report=read(fs::path(run.output_directory)/"report.json");
  test::check(object(report,"\"method\":\"pca_only\"").find("\"status\":\"unsupported_fit\"")!=std::string::npos &&
      object(report,"\"method\":\"native\"").find("\"status\":\"measured\"")!=std::string::npos &&
      tensor(directory/"milestone-2"/"rep-1"/"fit.pt","ridge_weights").size(0)==16,
      "standalone PCA failure disabled legal native selection/TEST or projected the native width");
}
void development_only(const fs::path &temporary) {
  auto run=configuration(temporary/"development-only");run.development_only=true;run.card.test_pairs=0;run.stress_sweep=false;
  // Neither a selection task nor a sanity budget is used by asset preparation.
  run.selection_task=ev::Task::amplitude;run.sanity_budget=0;
  const auto audits=std::make_shared<Audits>();
  torch::manual_seed(34821);const auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto rng=generator.get_state().clone();const auto threads=at::get_num_threads();
  ev::run_native_curve(run,factory("rpb_v4",run,Mode::ordinary,audits));
  test::check(torch::equal(generator.get_state(),rng) && at::get_num_threads()==threads,"development-only changed ambient RNG/threads");
  const fs::path output(run.output_directory);
  const auto card=read(output/"native-development-card.json"),validation=read(output/"validation-report.json"),completion=read(output/"development-complete.json");
  test::check(card.find("\"protocol\":\"native-development-v1\"")!=std::string::npos &&
      card.find("\"development_only\":true")!=std::string::npos && card.find("\"milestones\":[0,2,4]")!=std::string::npos &&
      card.find("selection_policy")==std::string::npos && card.find("selection_task")==std::string::npos &&
      card.find("fresh_test_namespace")==std::string::npos && card.find("fresh_test_stream")==std::string::npos &&
      card.find("stress_sweep")==std::string::npos && validation.find("\"protocol\":\"native-development-v1\"")!=std::string::npos &&
      validation.find("\"card\":\"native-development-card.json\"")!=std::string::npos,
      "development card/report claimed a selection or TEST/stress protocol");
  test::check(completion.find("\"checked_points\":12")!=std::string::npos &&
      completion.find("\"all_retained_witnesses_exact\":true")!=std::string::npos &&
      completion.find("\"selection_performed\":false")!=std::string::npos &&
      completion.find("\"testing_generated\":false")!=std::string::npos && completion.find("fsync")!=std::string::npos,
      "development completion omitted all-point witnesses or durable boundary");
  for(const auto &name:{"selection.json","report.json","stress-card.json","stress-report.json","native-curve-card.json"})
    test::check(!fs::exists(output/name),"development-only emitted a full-curve selection/TEST artifact");
  for(const auto &entry:fs::recursive_directory_iterator(output)) {
    const auto name=entry.path().filename().string();
    test::check(name.find("testing")==std::string::npos && name.find("stress")==std::string::npos,
        "development-only wrote a TEST/stress artifact");
  }
  for(const auto seed:run.card.seeds)for(const auto task:run.card.tasks) {
    const auto audit=audits->at(std::to_string(seed)+"/"+ev::task_name(task));
    test::check(audit->requests==run.milestones && audit->saves==3 && audit->snapshots==3 && audit->testing_calls==0 &&
        audit->protocol_id=="native-development-v1/"+ev::task_name(task),"development task omitted milestones or used another fit namespace");
    const auto cohort=output/("seed-"+std::to_string(seed)+"-"+ev::task_name(task));
    test::check(read(cohort/"development-manifest.json").find("\"testing_generated\":false")!=std::string::npos,"development cohort includes TEST");
    test::check(read(cohort/"controls"/"controls.json").find("\"observation_scaler_train_fits\":1")!=std::string::npos,"development controls did not retain fit-once assets");
    for(const auto budget:run.milestones) {
      const auto point=cohort/("milestone-"+std::to_string(budget));
      test::check(tensor(point/"checkpoint.pt","completed").item<int64_t>()==budget,"development checkpoint budget changed");
      for(const auto &split:{"training","validation"}) {
        const auto prefix=std::string(split)+"-reconstruction";
        for(const auto &key:{"standardized_prediction","standardized_target","target_mask","requested_observed_target_mask","visible_mask","trial_channel_eligible"})
          test::check(torch::equal(tensor(point/(prefix+".pt"),key),tensor(point/(prefix+"-witness.pt"),key)),"development omitted an immutable TRAIN/VAL decoder witness");
      }
      test::check(read(point/"witness-audit.json").find("\"performed_before_development_complete\":true")!=std::string::npos,
          "development completion preceded a retained point witness");
      test::check(fs::exists(point/"rep-1"/"fit.pt"),"development native TRAIN fit missing");
    }
  }
  rejects([&]{ev::run_native_curve(run,factory("rpb_v4",run,Mode::ordinary,std::make_shared<Audits>()));},"development-only overwrote output");
  for(const auto &[name,mode]:std::vector<std::pair<std::string,Mode>>{{"middle-features",Mode::mutable_middle_snapshot},{"middle-decoder",Mode::mutable_middle_reconstruction}}) {
    auto bad=run;bad.output_directory=(temporary/name).string();bad.card.tasks={ev::Task::lag_sign};bad.card.seeds={41};bad.repetitions={{"rep-1",2701}};
    auto failed_audits=std::make_shared<Audits>();
    rejects([&]{ev::run_native_curve(bad,factory("rpb_v4",bad,mode,failed_audits));},"development accepted a mutable intermediate snapshot");
    test::check(!fs::exists(fs::path(bad.output_directory)/"development-complete.json") &&
        !fs::exists(fs::path(bad.output_directory)/"selection.json"),"failed intermediate witness declared development complete");
    test::check(failed_audits->at("41/lag_sign")->requests==run.milestones,"intermediate mutation test did not reach later training");
  }
  auto unsupported=run;unsupported.output_directory=(temporary/"development-unsupported-validation").string();
  unsupported.card.tasks={ev::Task::lag_sign};unsupported.card.seeds={41};unsupported.repetitions={{"rep-1",2701}};
  ev::run_native_curve(unsupported,factory("rpb_v4",unsupported,Mode::unsupported_validation_all,std::make_shared<Audits>()));
  const fs::path unsupported_output(unsupported.output_directory);
  test::check(fs::exists(unsupported_output/"development-complete.json") &&
      !tensor(unsupported_output/"seed-41-lag_sign"/"milestone-4"/"native-validation.pt","valid").any().item<bool>() &&
      fs::exists(unsupported_output/"seed-41-lag_sign"/"milestone-4"/"rep-1"/"fit.pt"),
      "unsupported VALIDATION support blocked legal development asset preparation or discarded TRAIN fits");
  for(const auto name:{"nonzero-test","enabled-stress"}) {
    auto bad=run;bad.output_directory=(temporary/name).string();
    if(std::string(name)=="nonzero-test")bad.card.test_pairs=1;else bad.stress_sweep=true;
    const auto failed_audits=std::make_shared<Audits>();
    rejects([&]{ev::run_native_curve(bad,factory("rpb_v4",bad,Mode::ordinary,failed_audits));},"development accepted TEST or stress access");
    test::check(!fs::exists(bad.output_directory) && failed_audits->empty(),"invalid development boundary created output or invoked trainer");
  }
}
} // namespace
int main() {
  try {
    const auto directory=fs::temp_directory_path()/("embedding-native-curve-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);ordinary(directory);failures(directory);validation_and_intact(directory);unsupported_fits(directory);development_only(directory);
    std::cout << "shared native curve tests passed; artifacts=" << directory << '\n';return 0;
  }catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
