// SPDX-License-Identifier: MIT
#include "embedding/shared/learning_curve.h"
#include "shared_test_support.h"
#include <ATen/Context.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
namespace {
template <typename F> void rejects(F &&function, const std::string &message) {
  bool failed = false;
  try { function(); } catch (const std::exception &) { failed = true; }
  test::check(failed, message);
}
std::string read(const fs::path &path) {
  std::ifstream input(path);
  test::check(bool(input), "missing artifact: " + path.string());
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}
std::string object(const std::string &json, const std::string &marker) {
  const auto location = json.find(marker);
  test::check(location != std::string::npos, "report marker missing: " + marker);
  const auto begin = json.rfind('{',location);
  int depth = 0;
  bool quoted = false, escaped = false;
  for (size_t i = begin; i < json.size(); ++i) {
    const char c = json[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') quoted = false;
    } else if (c == '"') quoted = true;
    else if (c == '{') ++depth;
    else if (c == '}' && --depth == 0) return json.substr(begin,i-begin+1);
  }
  throw std::runtime_error("unterminated report object");
}
torch::Tensor tensor(const fs::path &path,const std::string &key) {
  torch::serialize::InputArchive input;
  input.load_from(path.string(),torch::kCPU);
  torch::Tensor out;
  input.read(key,out,true);
  return out;
}
ev::LearningCurveRun configuration(const fs::path &output) {
  ev::LearningCurveRun run;
  run.output_directory=output.string();
  run.source_fingerprint="dummy-curve-source"; run.git_head="dummy-head"; run.git_dirty="true";
  run.shape={2,32,1,torch::kFloat64,torch::kCPU}; run.channel_ids={4,9}; run.feature_units="unitless";
  run.patch_length=8; run.train_pairs=4; run.validation_pairs=6; run.test_pairs=5;
  run.matched_global_width=2; run.matched_channel_width=2;
  run.seeds={41,52}; run.milestones={0,2,4};
  return run;
}
enum class Mode { ordinary, unsupported_last, mutable_snapshot, nonfinite_reconstruction, unsupported_all };
struct Audit {
  int factories{0}, snapshots{0}, saves{0}, asset_saves{0}, test_feature_calls{0};
  std::vector<int64_t> requests;
  std::vector<double> draws;
  torch::Tensor mean,scale,expected_rng;
};
struct State {
  ev::CurveProgress progress;
  std::shared_ptr<Audit> audit;
};
ev::NamedCurveFactory factory(const std::string &name,const ev::LearningCurveRun &run,Mode mode,
                             const std::shared_ptr<std::map<uint64_t,std::shared_ptr<Audit>>> &audits) {
  return {name,"dummy monotonic checkpoint contract",[name,run,mode,audits](const ev::ProviderFitInput &input) {
    test::check(fs::exists(fs::path(run.output_directory)/"learning-curve-card.json") &&
                !fs::exists(fs::path(run.output_directory)/"selection.json"),
                "provider fitted before frozen card or after budget selection");
    const auto expected=ev::make_controlled_development_protocol(ev::Task::lag_sign,run.shape,
        run.train_pairs,run.validation_pairs,input.seed);
    test::check(input.training_source_ids==expected.training.source_ids &&
                input.channel_ids==run.channel_ids && input.training_observations.data.size(0)==8,
                "curve factory received held-out observations or wrong metadata");
    test::check(torch::equal(input.training_observations.data,expected.training.observed.data) &&
                torch::equal(input.training_observations.feature_mask,expected.training.observed.feature_mask),
                "training fixture changed before the label-free factory");
    auto audit=std::make_shared<Audit>(); ++audit->factories;
    audits->emplace(input.seed,audit);
    const ev::ObservationScaler scaler(input.training_observations);
    audit->mean=scaler.mean.clone(); audit->scale=scaler.scale.clone();
    auto state=std::make_shared<State>(); state->audit=audit;
    state->progress.parameter_count=1; state->progress.training_device="cpu";
    state->progress.preprocessing_id="dummy-frozen-scaler-"+std::to_string(input.seed);
    state->progress.training_dataset_id="dummy-training-only-"+std::to_string(input.seed);
    torch::manual_seed(input.seed);
    ev::CurveTrainer trainer;
    trainer.audit_fields={{"seed",std::to_string(input.seed)},{"legal_training_only","true"}};
    trainer.train_to=[state](int64_t budget) {
      auto generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));
      if(state->audit->expected_rng.defined())
        test::check(torch::equal(generator.get_state(),state->audit->expected_rng),
                    "checkpoint measurement reset/consumed the trainer RNG");
      test::check(budget>=state->progress.completed,"absolute budget regressed");
      state->audit->requests.push_back(budget);
      while(state->progress.completed<budget) {
        state->audit->draws.push_back(torch::rand({1},torch::kFloat64).item<double>());
        ++state->progress.completed; ++state->progress.attempted;
        state->progress.sampled_rows+=2;
        state->progress.losses.push_back({state->progress.attempted,state->progress.completed,48,
            1.0/(state->progress.completed+1),.5});
      }
      state->progress.training_seconds=.001*budget;
      state->progress.finite_gradients=budget>0; state->progress.weights_changed=budget>0;
      state->audit->expected_rng=generator.get_state().clone();
      return state->progress;
    };
    trainer.save_checkpoint=[state,scaler,seed=input.seed](const std::string &path) {
      ++state->audit->saves;
      test::check(!fs::exists(path),"checkpoint overwritten");
      torch::serialize::OutputArchive checkpoint;
      checkpoint.write("completed",torch::tensor(state->progress.completed),true);
      checkpoint.write("optimizer_counter",torch::tensor(state->progress.completed),true);
      checkpoint.write("train_mean",scaler.mean,true); checkpoint.write("train_scale",scaler.scale,true);
      checkpoint.write("seed",embedding::archive::text_tensor(std::to_string(seed)),true);
      embedding::archive::save_archive(path,checkpoint);
    };
    trainer.snapshot=[state,run,scaler,name,mode,seed=input.seed](const std::string &path) {
      ++state->audit->snapshots;
      // Snapshot construction intentionally disturbs Torch RNG; the shared
      // measurement stage must restore it before the next training request.
      torch::manual_seed(998877);
      const auto budget=tensor(path,"completed").item<int64_t>();
      auto frozen_scaler=scaler;
      frozen_scaler.mean=tensor(path,"train_mean").clone();
      frozen_scaler.scale=tensor(path,"train_scale").clone();
      frozen_scaler.counts=scaler.counts.clone();
      ev::CurveSnapshot snapshot;
      snapshot.features.provenance="immutable dummy checkpoint; legal observed histories only";
      snapshot.features.audit_fields={{"checkpoint_budget",std::to_string(budget)},{"fit_seed",std::to_string(seed)}};
      snapshot.features.surfaces={
        {"unusual_global",{ev::SurfaceKind::global,"at least one observed channel",{}}},
        {"tokens",{ev::SurfaceKind::channel_concatenation,"all semantic channels observed",run.channel_ids}}};
      snapshot.features.extract=[state,run,mode,budget](const embedding::Batch &batch) {
        const auto current=mode==Mode::mutable_snapshot?state->progress.completed:budget;
        if(batch.data.size(0)==2*run.test_pairs) {
          ++state->audit->test_feature_calls;
          test::check(fs::exists(fs::path(run.output_directory)/"selection.json") &&
                      read(fs::path(run.output_directory)/"selection.json").find("\"selected_budget\":2")!=std::string::npos,
                      "test observations accessed before the common selection was frozen");
        }
        test::check(batch.data.masked_select(batch.feature_mask.logical_not()).eq(0).all().item<bool>(),
                    "feature provider received hidden raw values");
        const auto oracle=ev::raw_oracle(ev::Task::lag_sign,batch);
        const auto sign=oracle.predictions.to(torch::kFloat64)*2-1;
        const auto channel=batch.data.select(1,0), mask=batch.feature_mask.select(1,0);
        const auto nuisance=channel.flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1).to(torch::kFloat64);
        const auto values=torch::stack({sign*double(current),nuisance},1);
        auto channel_valid=batch.feature_mask.flatten(2).any(2);
        auto valid=channel_valid.all(1);
        if(mode==Mode::unsupported_all || (mode==Mode::unsupported_last && current==4))
          valid=torch::zeros_like(valid);
        ev::FeatureMap features{
          {"unusual_global",{values.clone(),valid.clone(),"arbitrary typed global"}},
          {"tokens",{values.clone(),valid.clone(),"arbitrary typed concatenation"}}};
        // A provider cannot corrupt canonical data seen by the other model,
        // reconstruction trials, later checkpoints or selected test controls.
        batch.data.fill_(std::numeric_limits<double>::quiet_NaN());
        batch.feature_mask.fill_(false);
        torch::manual_seed(554433);
        return features;
      };
      snapshot.features.save_assets=[state,frozen_scaler,budget](const std::string &directory) {
        ++state->audit->asset_saves;
        torch::serialize::OutputArchive assets;
        assets.write("snapshot_budget",torch::tensor(budget),true);
        assets.write("train_mean",frozen_scaler.mean,true); assets.write("train_scale",frozen_scaler.scale,true);
        embedding::archive::save_archive((fs::path(directory)/"dummy-assets.pt").string(),assets);
      };
      snapshot.reconstruct=[state,frozen_scaler,run,budget,mode](const embedding::Batch &batch,const torch::Tensor &hidden) {
        test::check(torch::equal(frozen_scaler.mean,state->audit->mean) &&
                    torch::equal(frozen_scaler.scale,state->audit->scale),"checkpoint/scoring refitted the training scaler");
        const auto visible=batch.feature_mask.logical_and(hidden.logical_not());
        const auto B=batch.data.size(0),C=run.shape.channel_count,K=run.shape.history_length/run.patch_length;
        const auto query=batch.feature_mask.logical_and(hidden);
        const auto eligible=visible.reshape({B,C,K,-1}).any(-1).sum(-1).ge(2).logical_and(query.flatten(2).any(2));
        const auto target=frozen_scaler.transform(batch).data;
        auto prediction=torch::full_like(target,.1*budget);
        if(mode==Mode::nonfinite_reconstruction)
          prediction.fill_(std::numeric_limits<double>::quiet_NaN());
        batch.data.fill_(123456); batch.feature_mask.fill_(false); hidden.fill_(true);
        torch::rand({1});
        return ev::CurveReconstruction{prediction,target,eligible};
      };
      return snapshot;
    };
    return trainer;
  }};
}
void ordinary(const fs::path &temporary) {
  const auto output=temporary/"ordinary";
  const auto run=configuration(output);
  auto independent=std::make_shared<std::map<uint64_t,std::shared_ptr<Audit>>>();
  auto mixer=std::make_shared<std::map<uint64_t,std::shared_ptr<Audit>>>();
  ev::run_learning_curve(run,{factory("independent",run,Mode::ordinary,independent),factory("mixer",run,Mode::ordinary,mixer)});
  const auto selection=read(output/"selection.json"),report=read(output/"report.json"),validation=read(output/"validation-report.json");
  test::check(selection.find("\"selected_budget\":2")!=std::string::npos &&
              object(selection,"\"budget\":2").find("\"utility\":1")!=std::string::npos &&
              object(selection,"\"budget\":4").find("\"utility\":1")!=std::string::npos &&
              selection.find("\"budget\":0")==std::string::npos,
              "common equal-weight curve selection ignored exact ties or selected initialization");
  test::check(validation.find("\"milestone\":0")!=std::string::npos &&
              validation.find("\"training_reconstruction\"")!=std::string::npos &&
              validation.find("\"validation_reconstruction\"")!=std::string::npos &&
              validation.find("\"testing\"")==std::string::npos &&
              report.find("\"ridge_candidate_minus_comparator_grouped_interval\"")!=std::string::npos &&
              report.find("\"fitted_readouts_reused\":true")!=std::string::npos &&
              report.find("\"source_fingerprint\":\"dummy-curve-source\"")!=std::string::npos,
              "curve/test report lost split separation, frozen readout/uncertainty/provenance");
  const auto card=read(output/"learning-curve-card.json");
  test::check(card.find("\"ridge_penalty\":1")!=std::string::npos &&
              card.find("\"learning_rate\":0.01")!=std::string::npos &&
              card.find("\"initialization_is_diagnostic_only\":true")!=std::string::npos,
              "pre-data card omitted numerical readout/selection recipe");
  for(const auto seed:run.seeds) {
    const auto a=independent->at(seed),b=mixer->at(seed);
    test::check(a->requests==std::vector<int64_t>({0,2,4}) && b->requests==a->requests &&
                a->factories==1 && a->snapshots==3 && a->saves==3 && a->asset_saves==3 &&
                a->test_feature_calls==1 && b->test_feature_calls==1 && a->draws==b->draws && a->draws.size()==4,
                "optimizer/RNG/factory/checkpoint counters restarted or selected test was repeated");
    const auto seed_dir=output/("seed-"+std::to_string(seed)),point=seed_dir/"independent"/"milestone-2";
    test::check(tensor(point/"checkpoint.pt","completed").item<int64_t>()==2 &&
                tensor(point/"checkpoint.pt","optimizer_counter").item<int64_t>()==2 &&
                tensor(seed_dir/"independent"/"milestone-4"/"checkpoint.pt","completed").item<int64_t>()==4,
                "selected checkpoint was replaced by the final checkpoint");
    const auto selected_features=tensor(point/"tokens-selected-testing.pt","features");
    test::check(selected_features.select(1,0).abs().eq(2).all().item<bool>(),
                "selected test used the later mutable snapshot");
    const auto training=tensor(point/"tokens-training.pt","features");
    test::close(tensor(point/"tokens-matched_channels-fit.pt","feature_mean"),training.mean(0),
                "selected feature normalizer fitted validation/test rows",0,0);
    test::check(tensor(point/"tokens-matched_channels-fit.pt","fitted_rows").item<int64_t>()==8,
                "selected readout fit absorbed held-out rows");
    test::close(tensor(point/"provider-assets"/"dummy-assets.pt","train_mean"),a->mean,"frozen training scaler changed",0,0);
    const auto query=tensor(point/"validation-reconstruction.pt","target_mask");
    const auto observed=tensor(seed_dir/"controlled-validation.pt","feature_mask");
    test::check(torch::equal(query.sum(0),observed.to(torch::kInt64)) &&
                torch::equal(tensor(point/"validation-reconstruction.pt","requested_observed_target_mask").sum(0),
                             observed.to(torch::kInt64)) &&
                torch::isfinite(tensor(point/"validation-reconstruction.pt","example_standardized_mae")).all().item<bool>(),
                "fixed complete-patch trials changed natural masks/query denominators");
    const auto expected=ev::make_controlled_development_protocol(ev::Task::lag_sign,run.shape,4,6,seed);
    test::check(torch::equal(tensor(seed_dir/"controlled-training.pt","observed"),expected.training.observed.data) &&
                torch::equal(tensor(seed_dir/"controlled-training.pt","feature_mask"),expected.training.observed.feature_mask),
                "mutating callbacks corrupted fixed training observations");
    const auto fresh=ev::make_controlled_test_dataset(ev::Task::lag_sign,run.shape,5,
        ev::stream_seed(seed,0x6375727665746573ULL));
    test::check(torch::equal(tensor(seed_dir/"controlled-selected-testing.pt","observed"),fresh.observed.data),
                "selected test did not use the declared fresh source namespace");
    test::check(report.find("\"mask_metadata_ridge\":{\"total\":10,\"valid\":10,\"correct\":5,\"coverage\":1,\"accuracy\":0.5}")!=std::string::npos,
                "mask-only train-fit control fabricated signal performance");
  }
  rejects([&]{ev::run_learning_curve(run,{factory("independent",run,Mode::ordinary,independent),
                                        factory("mixer",run,Mode::ordinary,mixer)});},
          "existing artifacts were overwritten");
}
void failures(const fs::path &temporary) {
  for(const auto mode:{Mode::unsupported_last,Mode::mutable_snapshot,Mode::nonfinite_reconstruction,Mode::unsupported_all}) {
    const auto output=temporary/("failure-"+std::to_string(static_cast<int>(mode)));
    auto run=configuration(output); run.seeds={41};
    auto a=std::make_shared<std::map<uint64_t,std::shared_ptr<Audit>>>(),b=std::make_shared<std::map<uint64_t,std::shared_ptr<Audit>>>();
    const auto providers=std::vector<ev::NamedCurveFactory>{factory("independent",run,Mode::ordinary,a),factory("mixer",run,mode,b)};
    if(mode==Mode::unsupported_last) {
      ev::run_learning_curve(run,providers);
      const auto candidate=object(read(output/"selection.json"),"\"budget\":4");
      test::check(candidate.find("\"status\":\"unsupported_common_budget\"")!=std::string::npos &&
                  candidate.find("\"utility\":null")!=std::string::npos,
                  "unsupported architecture was silently removed from common utility");
    } else {
      rejects([&]{ev::run_learning_curve(run,providers);},"invalid snapshot/score/whole-budget failure was retained as measured");
      for(const auto &entry:fs::recursive_directory_iterator(output))
        test::check(entry.path().filename()!="controlled-selected-testing.pt",
                    "testing sources were generated before snapshot/selection validation passed");
    }
  }
  auto run=configuration(temporary/"invalid-card");
  auto audits=std::make_shared<std::map<uint64_t,std::shared_ptr<Audit>>>();
  const auto providers=std::vector<ev::NamedCurveFactory>{factory("independent",run,Mode::ordinary,audits),factory("mixer",run,Mode::ordinary,audits)};
  run.milestones={2,4};
  rejects([&]{ev::run_learning_curve(run,providers);},"curve accepted missing initialization");
  run.milestones={0,2,2};
  rejects([&]{ev::run_learning_curve(run,providers);},"curve accepted non-monotonic budget");
  run.milestones={0,2};run.patch_length=16;
  rejects([&]{ev::run_learning_curve(run,providers);},"curve accepted fewer than3patch groups");
  test::check(!fs::exists(run.output_directory),"invalid recipe generated data/output");
}
} // namespace
int main() {
  try {
    const auto path=fs::temp_directory_path()/("embedding-learning-curve-test-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(path);
    ordinary(path);
    failures(path);
    std::cout << "shared learning curve tests passed; artifacts=" << path << '\n';
    return 0;
  } catch(const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
