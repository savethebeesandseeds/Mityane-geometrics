// SPDX-License-Identifier: MIT
#include "embedding/shared/paired_pooling.h"
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
                  lower_validation_late,intact_mismatch,mutable_reconstruction,missing_endpoint,unsupported_train_last,skipped_attempt };
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
                             const std::shared_ptr<Audits> &audits,const std::string &training_namespace="") {
  const auto protocol=training_namespace.empty()?(run.development_only?"native-development-v1":"native-curve-v1"):training_namespace;
  return {variant,"dummy continuous train-only frozen snapshot recipe",[variant,run,mode,audits,protocol](const ev::ProviderFitInput &fit) {
    const fs::path output(run.output_directory);
    test::check((fs::exists(output/"native-curve-card.json") || fs::exists(output/"native-development-card.json") || fs::exists(output/"paired-pooling-card.json")) && !fs::exists(output/"comparison-manifest.json") && !fs::exists(output/"selection.json"),
                "factory called without preregistered card or after test selection");
    ev::Task task=ev::Task::lag_sign;
    bool matched=false;
    for(const auto candidate:run.card.tasks)
      if(fit.protocol_id==protocol+"/"+ev::task_name(candidate)){task=candidate;matched=true;}
    test::check(matched,"candidate fit received the wrong explicit training namespace");
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
      if(mode==Mode::skipped_attempt && budget>0 && state->progress.completed==0) {
        ++state->progress.attempted;state->progress.sampled_rows+=2;
      }
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
        const auto current=mode==Mode::mutable_snapshot?state->progress.completed:budget;
        if(batch.data.size(0)==2*run.card.test_pairs) {
          ++state->audit->testing_calls;
          test::check(state->audit->requests.back()==(task==run.selection_task?run.milestones.back():run.sanity_budget),"TEST reached a path before its training finished");
          const auto paired=fs::exists(fs::path(run.output_directory)/"paired-pooling-card.json");
          test::check(paired?fs::exists(fs::path(run.output_directory)/"comparison-manifest.json"):
              fs::exists(fs::path(run.output_directory)/"selection.json"),
              "fresh TEST data reached provider before durable selection/fixed-budget manifest");
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
        auto prediction=torch::full_like(target,.1*(mode==Mode::mutable_reconstruction?state->progress.completed:budget));
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
struct Fixture {
  ev::PairedPoolingRun run;
  ev::NativeCurveRun candidate_recipe;
  std::shared_ptr<Audits> candidate_audits=std::make_shared<Audits>();
  std::shared_ptr<std::map<uint64_t,ev::CurveTrainer>> reference_trainers=std::make_shared<std::map<uint64_t,ev::CurveTrainer>>();
};
Fixture fixture(const fs::path &root,bool development_only=false) {
  auto prior=configuration(root/"reference");prior.card.tasks={ev::Task::lag_sign};prior.stress_sweep=false;
  prior.development_only=development_only;
  if(development_only){prior.card.test_pairs=0;prior.card.id="native-development-v1";}
  const auto original=factory("reference_dummy",prior,Mode::ordinary,std::make_shared<Audits>());
  Fixture out;auto registered=original;
  registered.factory=[original,registry=out.reference_trainers](const ev::ProviderFitInput &input) {
    auto trainer=original.factory(input);test::check(registry->emplace(input.seed,trainer).second,"reference fixture reinitialized");return trainer;
  };
  ev::run_native_curve(prior,registered);
  out.run.recipe=prior;out.run.recipe.output_directory=(root/"paired").string();out.run.recipe.stress_sweep=true;out.run.recipe.card.comparisons.clear();
  out.run.recipe.development_only=false;out.run.recipe.card.test_pairs=5;
  if(development_only)out.run.training_protocol_namespace="native-development-v1";
  out.run.completed_updates=2;out.candidate_recipe=out.run.recipe;out.candidate_recipe.milestones={0,2};
  for(const auto seed:prior.card.seeds) {
    const auto cohort=fs::path(prior.output_directory)/("seed-"+std::to_string(seed)+"-lag_sign");
    ev::RetainedPoolingCohort input;input.master_seed=seed;input.lineage="generic native-curve dummy archives, explicit TRAIN/VALIDATION roles";
    input.training_observations=(cohort/"controlled-training.pt").string();input.validation_observations=(cohort/"controlled-validation.pt").string();
    input.reference_checkpoint=(cohort/"milestone-2"/"checkpoint.pt").string();input.reference_initial_checkpoint=(cohort/"milestone-0"/"checkpoint.pt").string();
    input.reference_training_features=(cohort/"milestone-2"/"native-training.pt").string();input.reference_validation_features=(cohort/"milestone-2"/"native-validation.pt").string();
    input.reference_initial_training_features=(cohort/"milestone-0"/"native-training.pt").string();input.reference_initial_validation_features=(cohort/"milestone-0"/"native-validation.pt").string();
    input.raw_scaler=(cohort/"controls"/"raw-scaler.pt").string();
    for(const auto &rep:prior.repetitions) {
      input.reference_fits.push_back((cohort/"milestone-2"/rep.id/"fit.pt").string());input.reference_initial_fits.push_back((cohort/"milestone-0"/rep.id/"fit.pt").string());
      input.reference_validation_predictions.push_back((cohort/"milestone-2"/rep.id/"validation-predictions.pt").string());input.reference_initial_validation_predictions.push_back((cohort/"milestone-0"/rep.id/"validation-predictions.pt").string());
      input.raw_fits.push_back((cohort/"controls"/"raw"/rep.id/"fit.pt").string());input.pca_fits.push_back((cohort/"controls"/"pca_only"/rep.id/"fit.pt").string());input.mask_fits.push_back((cohort/"controls"/"mask_metadata"/rep.id/"fit.pt").string());
      input.raw_validation_predictions.push_back((cohort/"controls"/"raw"/rep.id/"validation-predictions.pt").string());input.pca_validation_predictions.push_back((cohort/"controls"/"pca_only"/rep.id/"validation-predictions.pt").string());input.mask_validation_predictions.push_back((cohort/"controls"/"mask_metadata"/rep.id/"validation-predictions.pt").string());
    }
    out.run.cohorts.push_back(std::move(input));
  }
  return out;
}
ev::RetainedCurveSnapshotLoader loader(const Fixture &fixture) {
  return [registry=fixture.reference_trainers,protocol=fixture.run.training_protocol_namespace](const std::string &path,const ev::ProviderFitInput &input) {
    test::check(input.protocol_id==protocol+"/lag_sign","retained snapshot loader received another training namespace");
    auto snapshot=registry->at(input.seed).snapshot(path);
    // A loader that writes through const tensors must not poison the candidate
    // fitting metadata; every callback receives a fresh legal independent clone.
    input.training_observations.data.fill_(555);input.training_observations.feature_mask.fill_(false);
    return snapshot;
  };
}
ev::PoolingInitializationAudit initialization(bool accept=true,const std::string &training_namespace="native-curve-v1") {
  return [accept,training_namespace](const std::string &path,const ev::RetainedPoolingCohort &prior,const ev::ProviderFitInput &input) {
    test::check(input.protocol_id==training_namespace+"/lag_sign" && tensor(path,"completed").item<int64_t>()==0 &&
        tensor(prior.reference_initial_checkpoint,"completed").item<int64_t>()==0,"actual continuous-path initialization was not compared");
    test::check(torch::equal(tensor(path,"train_mean"),tensor(prior.reference_initial_checkpoint,"train_mean")) &&
        torch::equal(tensor(path,"train_scale"),tensor(prior.reference_initial_checkpoint,"train_scale")) &&
        torch::equal(input.training_observations.data,tensor(prior.training_observations,"observed")) &&
        torch::equal(input.training_observations.feature_mask,tensor(prior.training_observations,"feature_mask")),"initialization audit saw altered TRAIN/scaler");
    return std::map<std::string,std::string>{{"common_parameters_exact",accept?"true":"false"},{"scaler_exact","true"},{"training_dataset_exact","true"},{"counter_streams_exact","true"}};
  };
}
void wrappers(const Fixture &input) {
  const auto &cohort=input.run.cohorts.front();const fs::path output(input.run.recipe.output_directory);
  const auto directory=output/("seed-"+std::to_string(cohort.master_seed)+"-lag_sign");
  auto snapshot=input.reference_trainers->at(cohort.master_seed).snapshot(cohort.reference_checkpoint);
  const auto original_threads=at::get_num_threads();torch::set_num_threads(2);torch::manual_seed(917331);
  const auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));const auto rng=generator.get_state().clone();
  const auto split=ev::load_native_development_observations(cohort.validation_observations);
  test::check(torch::equal(split.observed.data,tensor(cohort.validation_observations,"observed")) &&
      torch::equal(split.observed.feature_mask,tensor(cohort.validation_observations,"feature_mask")) &&
      torch::equal(split.clean.data,torch::where(split.observed.feature_mask,split.observed.data,torch::zeros_like(split.observed.data))) &&
      torch::equal(split.clean.feature_mask,split.observed.feature_mask),"development loader changed observations or its legal clean placeholder");
  const auto before_data=split.observed.data.clone(),before_mask=split.observed.feature_mask.clone();
  const auto surface=ev::extract_native_global(snapshot,split.observed,input.run.recipe);
  test::check(torch::equal(surface.values,tensor(cohort.reference_validation_features,"features")) &&
      torch::equal(surface.valid,tensor(cohort.reference_validation_features,"valid")),"native wrapper changed the exact retained export");
  const auto features=directory/"wrapper-validation-features.pt";
  ev::save_native_feature_archive(features.string(),surface,split,input.run.recipe);
  for(const auto &key:{"features","valid","provenance","labels_scoring_only","source_ids_json"})
    test::check(torch::equal(tensor(features,key),tensor(cohort.reference_validation_features,key)),"native wrapper archive changed feature schema or row association");
  const auto archive=directory/"wrapper-validation-reconstruction.pt";
  const auto summary=ev::write_native_patch_reconstruction(archive.string(),split,snapshot,input.run.recipe);
  auto expected=read(directory/"reference-validation-reconstruction.json");
  const std::string previous="reference-validation-reconstruction.pt";
  const auto position=expected.find(previous);test::check(position!=std::string::npos,"existing reconstruction artifact field missing");
  expected.replace(position,previous.size(),archive.filename().string());
  test::check(summary==expected,"public reconstruction wrapper changed fixed-query summary reductions");
  for(const auto &key:{"standardized_prediction","standardized_target","target_mask","requested_observed_target_mask",
      "visible_mask","trial_channel_eligible","channel_target_counts","channel_valid","channel_standardized_mae",
      "channel_standardized_huber","example_valid","example_standardized_mae","example_standardized_huber","source_ids_json"})
    test::check(torch::equal(tensor(archive,key),tensor(directory/"reference-validation-reconstruction.pt",key)),"public reconstruction wrapper changed fixed-query arrays");
  test::check(torch::equal(split.observed.data,before_data) && torch::equal(split.observed.feature_mask,before_mask),"native wrapper callback mutated input observations");
  test::check(torch::equal(generator.get_state(),rng) && at::get_num_threads()==2,"native wrappers changed ambient RNG/thread state");
  const auto feature_bytes=read(features),reconstruction_bytes=read(archive);
  rejects([&]{ev::save_native_feature_archive(features.string(),surface,split,input.run.recipe);},"native feature writer accepted an existing destination");
  rejects([&]{ev::write_native_patch_reconstruction(archive.string(),split,snapshot,input.run.recipe);},"native reconstruction writer accepted an existing destination");
  test::check(read(features)==feature_bytes && read(archive)==reconstruction_bytes,"native wrapper overwrote an existing archive");
  auto wrong_geometry=input.run.recipe;wrong_geometry.patch_length=3;
  rejects([&]{ev::write_native_patch_reconstruction((directory/"invalid-geometry.pt").string(),split,snapshot,wrong_geometry);},"native reconstruction accepted nondivisible patch geometry");
  auto broken_lineage=split;broken_lineage.source_ids.pop_back();
  rejects([&]{ev::save_native_feature_archive((directory/"invalid-lineage.pt").string(),surface,broken_lineage,input.run.recipe);},"native archive accepted missing row lineage");
  auto missing=split;missing.observed.feature_mask=torch::zeros_like(split.observed.feature_mask);
  rejects([&]{ev::save_native_feature_archive((directory/"invalid-support.pt").string(),surface,missing,input.run.recipe);},"native archive accepted allmissing valid features");
  auto wrong_width=surface;wrong_width.values=surface.values.slice(1,0,1);
  rejects([&]{ev::save_native_feature_archive((directory/"invalid-width.pt").string(),wrong_width,split,input.run.recipe);},"native archive accepted an undeclared width");
  test::check(!fs::exists(directory/"invalid-geometry.pt") && !fs::exists(directory/"invalid-lineage.pt") &&
      !fs::exists(directory/"invalid-support.pt") && !fs::exists(directory/"invalid-width.pt"),"invalid native wrapper input produced an archive");
  torch::set_num_threads(original_threads);
}
void custom_protocol(const Fixture &input) {
  ev::PairedPoolingRun defaults;
  test::check(defaults.protocol_id=="paired-pooling-v1" &&
      defaults.fresh_test_namespace=="paired-pooling-v1/fresh-testing" &&
      defaults.fresh_test_stream==0x7070763174657374ULL && defaults.training_protocol_namespace=="native-curve-v1","historical paired protocol defaults changed");
  auto run=input.run;run.recipe.output_directory+="-context";run.recipe.stress_sweep=false;
  run.protocol_id="context-deletion-v1";run.fresh_test_namespace="context-deletion-v1/fresh-testing";
  run.fresh_test_stream=0x6374763174657374ULL;run.candidate_tag="RPB-v6";
  auto recipe=input.candidate_recipe;recipe.output_directory=run.recipe.output_directory;
  const auto audits=std::make_shared<Audits>();
  ev::run_paired_pooling(run,factory("candidate_dummy",recipe,Mode::ordinary,audits),loader(input),initialization());
  const fs::path output(run.recipe.output_directory);
  for(const auto &name:{"paired-pooling-card.json","validation-report.json","comparison-manifest.json","report.json"}) {
    const auto ledger=read(output/name);
    test::check(ledger.find("\"protocol\":\"context-deletion-v1\"")!=std::string::npos &&
        ledger.find("\"protocol\":\"paired-pooling-v1\"")==std::string::npos,"custom protocol ledger retained historical identity");
  }
  test::check(read(output/"paired-pooling-card.json").find("\"fresh_test_namespace\":\"context-deletion-v1/fresh-testing\"")!=std::string::npos,
      "custom fresh-testing namespace was not serialized");
  for(const auto &cohort:run.cohorts) {
    const auto directory=output/("seed-"+std::to_string(cohort.master_seed)+"-lag_sign");
    const auto fresh=ev::make_controlled_test_dataset(ev::Task::lag_sign,run.recipe.card.shape,
        run.recipe.card.test_pairs,ev::stream_seed(cohort.master_seed,run.fresh_test_stream));
    test::close(tensor(directory/"controlled-testing.pt","observed"),fresh.observed.data,"custom protocol used historical TEST stream",0,0);
    test::check(!torch::equal(tensor(directory/"controlled-testing.pt","source_ids_json"),
        tensor(fs::path(input.run.recipe.output_directory)/directory.filename()/"controlled-testing.pt","source_ids_json")),
        "distinct protocol namespace reused historical TEST sources");
  }
  run.recipe.output_directory+="-invalid";recipe.output_directory=run.recipe.output_directory;
  auto invalid_audits=std::make_shared<Audits>();run.fresh_test_namespace="paired-pooling-v1/fresh-testing";
  rejects([&]{ev::run_paired_pooling(run,factory("candidate_dummy",recipe,Mode::ordinary,invalid_audits),loader(input),initialization());},"mismatched protocol namespace accepted");
  test::check(!fs::exists(run.recipe.output_directory) && invalid_audits->empty(),"invalid namespace reached data/provider creation");
  run.fresh_test_namespace="context-deletion-v1/fresh-testing";run.protocol_id="invalid/protocol";
  rejects([&]{ev::run_paired_pooling(run,factory("candidate_dummy",recipe,Mode::ordinary,invalid_audits),loader(input),initialization());},"unsafe protocol identity accepted");
}
void ordinary(const fs::path &temporary) {
  auto input=fixture(temporary/"ordinary");
  std::map<std::string,std::string> originals;
  for(const auto &cohort:input.run.cohorts)for(const auto *paths:{&cohort.reference_fits,&cohort.reference_initial_fits,&cohort.raw_fits,&cohort.pca_fits,&cohort.mask_fits})
    for(const auto &path:*paths)originals.emplace(path,read(path));
  torch::manual_seed(91377);const auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));const auto ambient=generator.get_state().clone();const auto threads=at::get_num_threads();
  ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,Mode::ordinary,input.candidate_audits),loader(input),initialization());
  test::check(torch::equal(generator.get_state(),ambient) && at::get_num_threads()==threads,"paired driver failed to restore ambient RNG/threads");
  const fs::path output(input.run.recipe.output_directory);const auto report=read(output/"report.json"),manifest=read(output/"comparison-manifest.json");
  test::check(manifest.find("\"completed_updates\":2")!=std::string::npos && manifest.find("\"all_testing_after_manifest\":true")!=std::string::npos &&
      !fs::exists(output/"selection.json") && report.find("\"retained_readout_refits\":0")!=std::string::npos &&
      report.find("ridge_candidate_minus_comparator_grouped_interval")!=std::string::npos,"fixed budget/retained fits or paired intervals changed");
  const auto card=read(output/"paired-pooling-card.json"),stress_card=read(output/"stress-card.json");
  test::check(card.find("training_protocol_namespace")==std::string::npos && manifest.find("training_protocol_namespace")==std::string::npos,
      "default paired output acquired a nondefault training namespace field");
  test::check(stress_card.find("fixed-readout-stress-v1")!=std::string::npos,"frozen stress protocol missing");
  for(const auto &pair:{"candidate_vs_reference","candidate_vs_candidate_initial","reference_vs_reference_initial","candidate_vs_pca_only","reference_vs_pca_only"})
    test::check(card.find(pair)!=std::string::npos,"prescoring paired card omits actual declared stress comparisons");
  for(const auto &[path,bytes]:originals)test::check(read(path)==bytes,"paired runner changed immutable reference/control fit archive");
  test::check(read(output/"input-integrity-after.json").find("\"original_inputs_byte_preserved\":true")!=std::string::npos,"input preservation proof missing");
  for(const auto &cohort:input.run.cohorts) {
    const auto seed=cohort.master_seed;const auto directory=output/("seed-"+std::to_string(seed)+"-lag_sign");const auto audit=input.candidate_audits->at(std::to_string(seed)+"/lag_sign");
    test::check(audit->requests==std::vector<int64_t>({0,2}) && audit->factories==1 && audit->saves==2 && audit->snapshots==2,"candidate retrained/reset or changed absolute fixed budget");
    test::check(read(directory/"retained-fit-audit.json").find("\"readout_fitting_constructors\":0")!=std::string::npos,"retained fit restoration constructed a training probe");
    const auto fresh=ev::make_controlled_test_dataset(ev::Task::lag_sign,input.run.recipe.card.shape,input.run.recipe.card.test_pairs,ev::stream_seed(seed,input.run.fresh_test_stream));
    test::close(tensor(directory/"controlled-testing.pt","observed"),fresh.observed.data,"paired TEST draw used wrong namespace/cohort",0,0);
    test::close(tensor(directory/"controlled-training.pt","observed"),tensor(cohort.training_observations,"observed"),"candidate TRAIN observations changed",0,0);
    for(const auto &name:{"candidate","candidate_initial","reference","reference_initial"})
      test::check(torch::equal(tensor(directory/(std::string(name)+"-validation-reconstruction.pt"),"standardized_prediction"),
          tensor(directory/(std::string(name)+"-validation-reconstruction-witness.pt"),"standardized_prediction")),"retained decoder changed after candidate training");
    for(size_t i=0;i<input.run.recipe.repetitions.size();++i) {
      const auto &rep=input.run.recipe.repetitions[i];const auto fit=directory/"candidate-milestone-2"/rep.id/"fit.pt";
      test::check(torch::equal(tensor(fit,"actual_probe_seed_decimal"),tensor(cohort.reference_fits[i],"actual_probe_seed_decimal")),"paired native32 head seeds differ");
      torch::serialize::InputArchive archive;archive.load_from(fit.string(),torch::kCPU);torch::Tensor ignored;
      test::check(!archive.try_read("pca_components",ignored,true),"candidate native export acquired PCA");
      test::close(tensor(fit,"feature_mean"),tensor(directory/"candidate-milestone-2"/"native-training.pt","features").mean(0),"candidate feature scaler used VALIDATION/TEST rows",0,0);
      const auto stress=directory/("testing-"+rep.id)/"stress";
      const auto ordinary_predictions=directory/("testing-"+rep.id)/"candidate-predictions.pt";
      test::check(torch::equal(tensor(stress/"intact-candidate-native-predictions.pt","ridge"),tensor(ordinary_predictions,"ridge")) &&
          torch::equal(tensor(stress/"intact-candidate-native-predictions.pt","tiny"),tensor(ordinary_predictions,"tiny_secondary")),"intact stress differs from frozen TEST predictions");
      for(const auto &name:{"candidate","candidate_initial","reference","reference_initial","raw","pca_only"})
        test::check(!tensor(stress/(std::string("all_absent-")+name+"-native-predictions.pt"),"prediction_valid").any().item<bool>(),"allmissing paired signal invented support");
      test::check(tensor(stress/"all_absent-mask_metadata-native-predictions.pt","prediction_valid").all().item<bool>(),"allmissing metadata control lost support");
      test::check(read(stress/"report.json").find("random_dropout_030")!=std::string::npos,"primary additional30% deletion missing");
    }
  }
  wrappers(input);
  custom_protocol(input);
  rejects([&]{ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,Mode::ordinary,input.candidate_audits),loader(input),initialization());},"existing paired output overwritten");
}
void development_integration(const fs::path &temporary) {
  auto input=fixture(temporary/"development-to-paired",true);
  const auto baseline=fs::path(input.run.cohorts.front().training_observations).parent_path().parent_path();
  const auto complete=read(baseline/"development-complete.json");
  test::check(complete.find("\"protocol\":\"native-development-v1\"")!=std::string::npos &&
      complete.find("\"checked_points\":6")!=std::string::npos && !fs::exists(baseline/"selection.json") &&
      !fs::exists(baseline/"report.json"),"development reference requires a fabricated selection or TEST report");
  for(const auto &entry:fs::recursive_directory_iterator(baseline))
    test::check(entry.path().filename().string().find("testing")==std::string::npos,"development reference contains TEST observations");
  input.run.protocol_id="context-replication-v1";
  input.run.fresh_test_namespace="context-replication-v1/fresh-testing";
  input.run.fresh_test_stream=0x6372763174657374ULL;
  input.run.recipe.stress_sweep=false;
  input.candidate_recipe.stress_sweep=false;
  std::map<std::string,std::string> original_fits;
  for(const auto &cohort:input.run.cohorts)for(const auto *paths:{&cohort.reference_fits,&cohort.reference_initial_fits,&cohort.raw_fits,&cohort.pca_fits,&cohort.mask_fits})
    for(const auto &path:*paths)original_fits.emplace(path,read(path));
  ev::run_paired_pooling(input.run,
      factory("candidate_dummy",input.candidate_recipe,Mode::ordinary,input.candidate_audits,input.run.training_protocol_namespace),
      loader(input),initialization(true,input.run.training_protocol_namespace));
  const fs::path output(input.run.recipe.output_directory);
  for(const auto &name:{"paired-pooling-card.json","comparison-manifest.json"})
    test::check(read(output/name).find("\"training_protocol_namespace\":\"native-development-v1\"")!=std::string::npos,
        "nondefault label-free training namespace not frozen before fresh TEST");
  test::check(read(output/"report.json").find("\"retained_readout_refits\":0")!=std::string::npos &&
      !fs::exists(output/"selection.json"),"development assets were refitted or required a budget search");
  for(const auto &[path,bytes]:original_fits)test::check(read(path)==bytes,"paired integration changed a development TRAIN-fitted readout");
  test::check(read(baseline/"development-complete.json")==complete,"paired integration changed the durable development completion");
  for(const auto &cohort:input.run.cohorts) {
    const auto audit=input.candidate_audits->at(std::to_string(cohort.master_seed)+"/lag_sign");
    test::check(audit->protocol_id=="native-development-v1/lag_sign" && audit->requests==std::vector<int64_t>({0,2}) &&
        audit->testing_calls>0,"paired candidate lost the preparation namespace or did not use one continuous path");
    const auto directory=output/("seed-"+std::to_string(cohort.master_seed)+"-lag_sign");
    test::check(read(directory/"retained-fit-audit.json").find("\"readout_fitting_constructors\":0")!=std::string::npos,
        "development retained fits were reconstructed by fitting");
    const auto fresh=ev::make_controlled_test_dataset(ev::Task::lag_sign,input.run.recipe.card.shape,input.run.recipe.card.test_pairs,
        ev::stream_seed(cohort.master_seed,input.run.fresh_test_stream));
    test::close(tensor(directory/"controlled-testing.pt","observed"),fresh.observed.data,"development integration reused a historical TEST stream",0,0);
  }
  for(const auto &unsafe:{"","../bad","bad/namespace"}) {
    auto bad=input.run;bad.recipe.output_directory+="-invalid-namespace";bad.training_protocol_namespace=unsafe;
    auto recipe=input.candidate_recipe;recipe.output_directory=bad.recipe.output_directory;
    auto audits=std::make_shared<Audits>();
    rejects([&]{ev::run_paired_pooling(bad,factory("candidate_dummy",recipe,Mode::ordinary,audits,bad.training_protocol_namespace),loader(input),initialization());},
        "unsafe or empty training namespace accepted");
    test::check(!fs::exists(bad.recipe.output_directory) && audits->empty(),"invalid training namespace reached output or fitting");
  }
}
void failures(const fs::path &temporary) {
  {
    auto input=fixture(temporary/"skipped-attempt");input.run.recipe.stress_sweep=false;
    rejects([&]{ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,Mode::skipped_attempt,input.candidate_audits),loader(input),initialization());},"unmatched candidate skip trajectory accepted");
    const auto output=fs::path(input.run.recipe.output_directory);
    test::check(!fs::exists(output/"comparison-manifest.json") &&
        !fs::exists(output/"seed-41-lag_sign"/"controlled-testing.pt") &&
        !fs::exists(output/"seed-41-lag_sign"/"candidate-milestone-2"/"checkpoint.pt"),"skipped trajectory admitted the positive checkpoint or fresh TEST");
    for(const auto &[key,audit]:*input.candidate_audits) {
      (void)key;
      test::check(audit->testing_calls==0,"skipped candidate observed fresh TEST before rejection");
    }
  }
  for(const auto &[name,mode]:std::vector<std::pair<std::string,Mode>>{{"live-feature",Mode::mutable_snapshot},{"live-decoder",Mode::mutable_reconstruction}}) {
    auto input=fixture(temporary/name);input.run.recipe.stress_sweep=false;
    rejects([&]{ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,mode,input.candidate_audits),loader(input),initialization());},"live candidate snapshot accepted");
    test::check(!fs::exists(fs::path(input.run.recipe.output_directory)/"comparison-manifest.json") &&
        !fs::exists(fs::path(input.run.recipe.output_directory)/"seed-41-lag_sign"/"controlled-testing.pt"),"invalid snapshot admitted fresh TEST");
  }
  {
    auto input=fixture(temporary/"audit-failure");input.run.recipe.stress_sweep=false;
    rejects([&]{ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,Mode::ordinary,input.candidate_audits),loader(input),initialization(false));},"failed common initialization audit accepted");
    test::check(!fs::exists(fs::path(input.run.recipe.output_directory)/"comparison-manifest.json"),"failed initialization admitted TEST");
  }
  {
    auto input=fixture(temporary/"hash-failure");input.run.cohorts.front().expected_sha256[input.run.cohorts.front().raw_scaler]=std::string(64,'0');
    rejects([&]{ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,Mode::ordinary,input.candidate_audits),loader(input),initialization());},"incorrect declared reference SHA accepted");
    test::check(input.candidate_audits->empty(),"hash verification happened after candidate training");
  }
  {
    auto input=fixture(temporary/"prediction-mismatch");const auto &cohort=input.run.cohorts.front();const auto original=cohort.reference_validation_predictions.front();
    const auto changed=temporary/"changed-validation-predictions.pt";torch::serialize::OutputArchive archive;
    for(const auto &key:{"valid","labels_scoring_only","source_ids_json","tiny_secondary"})archive.write(key,tensor(original,key),true);
    archive.write("ridge",1-tensor(original,"ridge"),true);embedding::archive::save_archive(changed.string(),archive);
    input.run.cohorts.front().reference_validation_predictions.front()=changed.string();
    rejects([&]{ev::run_paired_pooling(input.run,factory("candidate_dummy",input.candidate_recipe,Mode::ordinary,input.candidate_audits),loader(input),initialization());},"loaded fit prediction mismatch was silently refitted");
    test::check(input.candidate_audits->empty(),"original fit parity not verified before candidate training");
  }
}
} // namespace
int main() {
  try {
    const auto directory=fs::temp_directory_path()/("embedding-paired-pooling-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);ordinary(directory);failures(directory);development_integration(directory);
    std::cout << "shared paired pooling tests passed; artifacts=" << directory << '\n';return 0;
  }catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
