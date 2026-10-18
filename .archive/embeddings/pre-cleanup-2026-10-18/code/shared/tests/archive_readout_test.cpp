// SPDX-License-Identifier: MIT
#include "embedding/shared/archive_readout.h"
#include "shared_test_support.h"
#include <ATen/Context.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace ev=embedding::evaluation;
namespace fs=std::filesystem;
namespace {
template<class F>void rejects(F &&function,const std::string &message) {
  bool failed=false;try{function();}catch(const std::exception&){failed=true;}test::check(failed,message);
}
std::string read(const fs::path &path) {
  std::ifstream input(path,std::ios::binary);test::check(bool(input),"missing file: "+path.string());
  std::ostringstream out;out << input.rdbuf();return out.str();
}
torch::Tensor tensor(const fs::path &path,const std::string &key) {
  torch::serialize::InputArchive input;input.load_from(path.string(),torch::kCPU);
  torch::Tensor out;input.read(key,out,true);return out;
}
bool has_key(const fs::path &path,const std::string &key) {
  torch::serialize::InputArchive input;input.load_from(path.string(),torch::kCPU);
  torch::Tensor out;return input.try_read(key,out,true);
}
std::string sources(const std::string &prefix,int64_t pairs) {
  std::ostringstream out;out << '[';
  for(int64_t group=0;group<pairs;++group)for(int64_t variant=0;variant<2;++variant) {
    if(group||variant) {out << ',';}
    out << '"' << prefix << "-source-" << group << '"';
  }
  return out.str()+']';
}
struct RuntimeState {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  int threads{at::get_num_threads()};
  RuntimeState() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i)generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &generator:generators)states.push_back(generator.get_state().clone());
  }
  void unchanged() const {
    test::check(at::get_num_threads()==threads,"archive readout changed ambient thread count");
    for(size_t i=0;i<generators.size();++i)test::check(torch::equal(generators[i].get_state(),states[i]),"archive readout changed ambient RNG");
  }
};
struct Fixture {
  fs::path directory;
  embedding::Batch training,validation;
  ev::FeatureSurface native_training,native_validation;
  torch::Tensor labels,validation_labels;
  std::string validation_sources{"validation"};
  Fixture(const fs::path &root):directory(root) {
    fs::create_directories(directory);torch::manual_seed(714);
    auto values=torch::randn({16,2,4,1},torch::kFloat64);
    auto mask=torch::ones_like(values,torch::kBool);
    labels=torch::tensor({0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1},torch::kInt64);
    validation_labels=labels.clone();
    for(int64_t group=0;group<8;++group) {
      mask.narrow(0,2*group,2).select(2,group%4).fill_(false);
      values[2*group][0][0][0]=-2.0;values[2*group+1][0][0][0]=2.0;
    }
    mask.narrow(0,14,2).fill_(false);
    training={torch::where(mask,values,torch::full_like(values,std::numeric_limits<double>::quiet_NaN())),mask};
    validation={torch::where(mask,values*0.7+0.3,torch::full_like(values,std::numeric_limits<double>::quiet_NaN())),mask.clone()};
    // Exact PCA-only outputs serve as a generic frozen native fixture. This
    // exercises paired equal-width head initialization without an encoder.
    ev::ObservationScaler scaler(training);const auto train=scaler.transform(training),val=scaler.transform(validation);
    auto raw=[&](const embedding::Batch &batch) {
      return ev::FeatureSurface{torch::cat({batch.data.flatten(1),batch.feature_mask.to(torch::kFloat64).flatten(1)},1),
          batch.feature_mask.flatten(1).any(1),"dummy raw"};
    };
    const auto raw_training=raw(train),raw_validation=raw(val);ev::FeatureNormalizer normalizer(raw_training);
    ev::TrainPca pca(normalizer.transform(raw_training),2);
    native_training=pca.transform(normalizer.transform(raw_training));native_validation=pca.transform(normalizer.transform(raw_validation));
    native_training.provenance=native_validation.provenance="dummy exact frozen native export; declared producer row order";
    native_training.values.narrow(0,14,2).fill_(std::numeric_limits<double>::quiet_NaN());
    native_validation.values.narrow(0,14,2).fill_(std::numeric_limits<double>::quiet_NaN());
  }
  void save_observations(const fs::path &path,const embedding::Batch &batch,const std::string &prefix,const torch::Tensor &targets) const {
    torch::serialize::OutputArchive out;out.write("observed",batch.data,true);out.write("feature_mask",batch.feature_mask,true);
    out.write("labels_scoring_only",targets,true);out.write("source_ids_json",embedding::archive::text_tensor(sources(prefix,8)),true);
    embedding::archive::save_archive(path.string(),out);
  }
  void save_native(const fs::path &path,const ev::FeatureSurface &surface) const {
    torch::serialize::OutputArchive out;out.write("features",surface.values,true);out.write("valid",surface.valid,true);
    out.write("provenance",embedding::archive::text_tensor(surface.provenance),true);embedding::archive::save_archive(path.string(),out);
  }
  void save() const {
    save_observations(directory/"controlled-training.pt",training,"train",labels);
    save_observations(directory/"controlled-validation.pt",validation,validation_sources,validation_labels);
    save_native(directory/"native-training.pt",native_training);save_native(directory/"native-validation.pt",native_validation);
    std::ofstream sentinel(directory/"controlled-testing.pt");sentinel << "INVALID TORCH ARCHIVE; NEVER DISCOVER OR OPEN";
  }
  ev::ArchiveReadoutInput input(const std::string &id="dummy") const {
    ev::ArchiveReadoutInput out;out.id=id;out.tag="dummy-v1";out.task="lag_sign";out.master_seed=901;out.checkpoint_steps=512;
    out.producer_source_fingerprint="dummy-producer";out.cohort_provenance="dummy audited TRAIN/VALIDATION association";
    out.training_observations=(directory/"controlled-training.pt").string();out.validation_observations=(directory/"controlled-validation.pt").string();
    out.training_features=(directory/"native-training.pt").string();out.validation_features=(directory/"native-validation.pt").string();
    out.expected_feature_provenance=native_training.provenance;return out;
  }
};
ev::ArchiveReadoutRun configuration(const fs::path &output,const Fixture &fixture) {
  ev::ArchiveReadoutRun run;run.output_directory=output.string();run.shape={2,4,1,torch::kFloat64,torch::kCPU};run.compact_width=2;
  run.source_fingerprint="dummy-shared-producer";run.inputs={fixture.input()};run.repetitions={{"rep-1",2701}};run.bootstrap_replicates=100;
  // Exercise explicit recipe fields rather than merely their production defaults.
  run.ridge_penalty=0.7;run.tiny_hidden=5;run.tiny_steps=7;run.tiny_learning_rate=0.02;return run;
}
void compare_fit(const fs::path &left,const fs::path &right) {
  for(const auto &key:{"feature_mean","feature_scale","fitted_rows","ridge_mean","ridge_scale","ridge_weights","ridge_intercept",
      "tiny_mean","tiny_scale","tiny_w1","tiny_b1","tiny_w2","tiny_b2","actual_probe_seed_decimal"})
    test::check(torch::equal(tensor(left,key),tensor(right,key)),"heldout/hidden payload changed TRAIN-fitted asset: "+std::string(key));
}
void normal_and_isolation(const fs::path &root) {
  Fixture fixture(root/"original-inputs");fixture.save();
  const auto original_observations=read(fixture.directory/"controlled-training.pt");
  const auto original_features=read(fixture.directory/"native-training.pt");
  auto first=configuration(root/"first-results",fixture);RuntimeState runtime;ev::run_archive_readout(first);runtime.unchanged();
  test::check(read(fixture.directory/"controlled-training.pt")==original_observations && read(fixture.directory/"native-training.pt")==original_features,"input archive modified");
  const auto card=read(root/"first-results/archive-readout-card.json"),report=read(root/"first-results/report.json");
  test::check(card.find("\"post_encoder_pca\":false")!=std::string::npos && card.find("\"ridge\":{\"penalty\":")!=std::string::npos &&
      card.find("\"hidden\":5")!=std::string::npos && card.find("\"updates\":7")!=std::string::npos,"frozen recipe fields missing");
  test::check(report.find("\"valid_rows\":14")!=std::string::npos && report.find("\"total_rows\":16")!=std::string::npos &&
      report.find("\"coverage\":0.875")!=std::string::npos && report.find("\"source_groups\":7")!=std::string::npos,"paired validity denominators changed");
  test::check(read(root/"first-results/input-manifest.json").find("sha256-file-bytes")!=std::string::npos &&
      read(root/"first-results/output-manifest.json").find("raw-scaler.pt")!=std::string::npos,"immutable input/output hash manifests missing");
  test::check(torch::equal(tensor(root/"first-results/dummy/native-training.pt","features").narrow(0,0,14),fixture.native_training.values.narrow(0,0,14)),"native archive values projected or changed");
  test::check(!has_key(root/"first-results/dummy/rep-1/native-fit.pt","pca_components"),"native fit contains post-encoder PCA");
  const auto raw=tensor(root/"first-results/dummy/raw-training.pt","features");
  test::check(raw.size(1)==16 && raw.narrow(1,0,8).masked_select(fixture.training.feature_mask.flatten(1).logical_not()).eq(0).all().item<bool>(),"hidden raw values reached raw features");
  test::check(torch::equal(raw.narrow(1,8,8),fixture.training.feature_mask.to(torch::kFloat64).flatten(1)),"raw visibility flags changed");
  const auto native_seed=embedding::archive::tensor_text(tensor(root/"first-results/dummy/rep-1/native-fit.pt","actual_probe_seed_decimal"));
  const auto pca_seed=embedding::archive::tensor_text(tensor(root/"first-results/dummy/rep-1/pca_only-fit.pt","actual_probe_seed_decimal"));
  test::check(native_seed==pca_seed && native_seed==std::to_string(ev::stream_seed(2701,2)),"equal-width probe seeds not paired");
  // Compare against the existing primitives with nondefault recipe values:
  // this checks driver wiring without depending on JSON floating formatting.
  RuntimeState reference_state;
  ev::FeatureNormalizer expected_outer(fixture.native_training);
  const auto expected_features=expected_outer.transform(fixture.native_training);
  ev::RidgeProbe expected_ridge(expected_features,fixture.labels,0.7);
  ev::TinyProbe expected_tiny(expected_features,fixture.labels,ev::stream_seed(2701,2),7,5,0.02);
  test::close(tensor(root/"first-results/dummy/rep-1/native-fit.pt","ridge_weights"),expected_ridge.weights,"custom ridge penalty ignored",1e-12,1e-12);
  for(const auto &[key,expected]:std::vector<std::pair<std::string,torch::Tensor>>{
      {"tiny_w1",expected_tiny.w1},{"tiny_b1",expected_tiny.b1},{"tiny_w2",expected_tiny.w2},{"tiny_b2",expected_tiny.b2}})
    test::close(tensor(root/"first-results/dummy/rep-1/native-fit.pt",key),expected,"custom neural recipe ignored",1e-12,1e-12);
  for(size_t i=0;i<reference_state.generators.size();++i)reference_state.generators[i].set_state(reference_state.states[i]);
  reference_state.unchanged();
  for(const auto &key:{"ridge_weights","tiny_w1","tiny_b1","tiny_w2","tiny_b2"})
    test::close(tensor(root/"first-results/dummy/rep-1/native-fit.pt",key),tensor(root/"first-results/dummy/rep-1/pca_only-fit.pt",key),"identical two-dimensional inputs received different head fits",1e-10,1e-10);
  // Separate archive paths, same TRAIN data except masked storage. All heldout
  // numeric values change. Both mutations must leave every fitted asset exact.
  Fixture changed(root/"changed-inputs");changed.training=fixture.training;changed.native_training=fixture.native_training;
  changed.training.data=changed.training.data.clone().masked_fill(changed.training.feature_mask.logical_not(),1e150);
  changed.validation.data=changed.validation.data.clone()+71.0;
  changed.validation_labels=1-changed.validation_labels;
  changed.native_validation.values=changed.native_validation.values.clone()*11.0+37.0;changed.save();
  auto second=configuration(root/"second-results",changed);RuntimeState before_second;ev::run_archive_readout(second);before_second.unchanged();
  for(const auto &method:{"raw","pca_only","native"})compare_fit(root/(std::string("first-results/dummy/rep-1/")+method+"-fit.pt"),root/(std::string("second-results/dummy/rep-1/")+method+"-fit.pt"));
  for(const auto &key:{"mean","scale","counts"})test::check(torch::equal(tensor(root/"first-results/dummy/raw-scaler.pt",key),tensor(root/"second-results/dummy/raw-scaler.pt",key)),"raw scaler fit saw heldout/hidden values");
  for(const auto &key:{"pca_mean","pca_components","pca_singular_values","pca_numerical_rank","fitted_rows"})
    test::check(torch::equal(tensor(root/"first-results/dummy/pca-only.pt",key),tensor(root/"second-results/dummy/pca-only.pt",key)),"PCA fit saw heldout/hidden values");
  test::check(!torch::equal(tensor(root/"first-results/dummy/rep-1/native-validation-predictions.pt","probe_features"),tensor(root/"second-results/dummy/rep-1/native-validation-predictions.pt","probe_features")),"heldout perturbation fixture ineffective");
  rejects([&]{ev::run_archive_readout(first);},"existing output accepted");
}
void unsupported_and_abstention(const fs::path &root) {
  Fixture rank(root/"rank-inputs");rank.training.feature_mask.fill_(true);
  rank.training.data=rank.labels.to(torch::kFloat64).mul(2).sub(1).reshape({16,1,1,1}).expand({16,2,4,1}).clone();
  rank.native_training.valid.fill_(true);rank.native_training.values.narrow(0,14,2).fill_(0);rank.save();
  auto rank_run=configuration(root/"rank-results",rank);ev::run_archive_readout(rank_run);
  const auto rank_report=read(root/"rank-results/report.json");
  test::check(rank_report.find("PCA dimensions exceed numerical training rank")!=std::string::npos &&
      rank_report.find("\"raw_pca_numerical_rank\":null")!=std::string::npos && fs::exists(root/"rank-results/dummy/rep-1/native-fit.pt") &&
      fs::exists(root/"rank-results/dummy/rep-1/raw-fit.pt") && !fs::exists(root/"rank-results/dummy/pca-only.pt"),"unsupported PCA dropped raw/native or fabricated fit");
  Fixture absent(root/"absent-inputs");absent.validation.feature_mask.fill_(false);absent.validation.data.fill_(std::numeric_limits<double>::quiet_NaN());
  absent.native_validation.valid.fill_(false);absent.native_validation.values.fill_(std::numeric_limits<double>::quiet_NaN());absent.save();
  auto absent_run=configuration(root/"absent-results",absent);ev::run_archive_readout(absent_run);
  const auto report=read(root/"absent-results/report.json");
  test::check(report.find("\"total\":16,\"valid\":0,\"correct\":0,\"abstained\":16")!=std::string::npos &&
      report.find("\"coverage\":0,\"accuracy\":null")!=std::string::npos && report.find("\"source_groups\":0")!=std::string::npos,"all-missing validation laundered support");
  test::check(tensor(root/"absent-results/dummy/rep-1/raw-validation-predictions.pt","valid").eq(false).all().item<bool>(),"all-missing rows became valid");
  Fixture unequal(root/"unequal-inputs");
  unequal.native_training.valid.narrow(0,0,2).fill_(false);
  unequal.native_validation.valid.narrow(0,0,4).fill_(false);
  unequal.native_training.values.narrow(0,0,2).fill_(std::numeric_limits<double>::quiet_NaN());
  unequal.native_validation.values.narrow(0,0,4).fill_(std::numeric_limits<double>::quiet_NaN());unequal.save();
  auto unequal_run=configuration(root/"unequal-results",unequal);ev::run_archive_readout(unequal_run);
  test::check(tensor(root/"unequal-results/dummy/rep-1/raw-fit.pt","fitted_rows").item<int64_t>()==14 &&
      tensor(root/"unequal-results/dummy/rep-1/native-fit.pt","fitted_rows").item<int64_t>()==12,"native coverage changed raw TRAIN fit population");
  test::check(tensor(root/"unequal-results/dummy/rep-1/raw-validation-predictions.pt","valid").sum().item<int64_t>()==14 &&
      tensor(root/"unequal-results/dummy/rep-1/native-validation-predictions.pt","valid").sum().item<int64_t>()==10,"method-specific coverage was laundered");
  const auto unequal_report=read(root/"unequal-results/report.json");
  const auto start=unequal_report.find("\"id\":\"native_minus_pca_only\"");
  test::check(start!=std::string::npos && unequal_report.substr(start,600).find("\"valid_rows\":10")!=std::string::npos &&
      unequal_report.substr(start,600).find("\"valid_source_groups\":5")!=std::string::npos,"paired common population ignored unequal native support");
  Fixture invalid_native(root/"invalid-native-inputs");invalid_native.native_training.valid.fill_(false);invalid_native.native_validation.valid.fill_(false);
  invalid_native.native_training.values.fill_(std::numeric_limits<double>::quiet_NaN());invalid_native.native_validation.values.fill_(std::numeric_limits<double>::quiet_NaN());invalid_native.save();
  auto invalid_run=configuration(root/"invalid-native-results",invalid_native);ev::run_archive_readout(invalid_run);
  test::check(fs::exists(root/"invalid-native-results/dummy/rep-1/raw-fit.pt") && fs::exists(root/"invalid-native-results/dummy/rep-1/pca_only-fit.pt") &&
      !fs::exists(root/"invalid-native-results/dummy/rep-1/native-fit.pt"),"unsupported native fit disabled independent raw/PCA methods");
}
void rejections(const fs::path &root) {
  Fixture fixture(root/"rejection-inputs");fixture.save();auto run=configuration(root/"rejected",fixture);
  auto wrong=run;wrong.inputs[0].validation_observations=(fixture.directory/"controlled-testing.pt").string();
  rejects([&]{ev::run_archive_readout(wrong);},"testing-marked input accepted");
  test::check(!fs::exists(root/"rejected"),"testing guard deserialized before rejection");
  wrong=run;wrong.inputs[0].training_features_sha256=std::string(64,'0');rejects([&]{ev::run_archive_readout(wrong);},"mismatched archive hash accepted");
  test::check(!fs::exists(root/"rejected"),"hash guard ran after fitting");
  fixture.validation_sources="train";fixture.save();RuntimeState state;
  rejects([&]{ev::run_archive_readout(run);},"overlapping TRAIN/VALIDATION source accepted");state.unchanged();
  test::check(fs::exists(root/"rejected/archive-readout-card.json") && !fs::exists(root/"rejected/dummy"),"card was not frozen before role validation");
  fixture.validation_sources="validation";fixture.save();run.output_directory=(root/"bad-support").string();
  auto mask=fixture.native_validation.valid.clone();fixture.native_validation.valid=torch::ones_like(mask);fixture.save();
  rejects([&]{ev::run_archive_readout(run);},"native support exceeding observations accepted");
  fixture.native_validation.valid=mask;fixture.native_validation.values=fixture.native_validation.values.narrow(0,0,14);fixture.save();
  run.output_directory=(root/"bad-row-count").string();rejects([&]{ev::run_archive_readout(run);},"native row count mismatch accepted");
  Fixture nonfinite(root/"nonfinite-inputs");nonfinite.training.data[0][0][1][0]=std::numeric_limits<double>::infinity();nonfinite.save();
  auto finite_run=configuration(root/"nonfinite-results",nonfinite);rejects([&]{ev::run_archive_readout(finite_run);},"nonfinite observed value accepted");
  Fixture overflow(root/"overflow-inputs");overflow.native_validation.values.narrow(0,0,14).fill_(std::numeric_limits<double>::max());overflow.save();
  auto overflow_run=configuration(root/"overflow-results",overflow);RuntimeState overflow_state;
  rejects([&]{ev::run_archive_readout(overflow_run);},"finite validation overflow concealed by argmax");overflow_state.unchanged();
}
void known_sha256(const fs::path &root) {
  Fixture fixture(root/"sha-inputs");fixture.save();
  // The SHA guard accepts the standard 'abc' vector; deserialization then fails.
  // Card existence distinguishes hash success from rejecting a malformed file.
  const auto path=fixture.directory/"abc-training.pt";{std::ofstream out(path,std::ios::binary);out << "abc";}
  auto run=configuration(root/"abc-results",fixture);run.inputs[0].training_observations=path.string();
  run.inputs[0].training_observations_sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  rejects([&]{ev::run_archive_readout(run);},"invalid abc archive accepted");
  test::check(fs::exists(root/"abc-results/archive-readout-card.json"),"SHA-256 standard vector rejected before archive decoding");
}
ev::ArchiveReadoutValidationView save_view(const Fixture &fixture,const std::string &id,
    const embedding::Batch &observed,const ev::FeatureSurface &features,
    const torch::Tensor &labels,const std::string &source_prefix="validation") {
  ev::ArchiveReadoutValidationView view;
  view.id=id;view.observations=(fixture.directory/(id+"-validation.pt")).string();
  view.features=(fixture.directory/(id+"-features.pt")).string();
  view.expected_feature_provenance=features.provenance;view.corruption_provenance="declared dummy deletion-only VALIDATION view";
  fixture.save_observations(view.observations,observed,source_prefix,labels);fixture.save_native(view.features,features);
  return view;
}
void validation_views_and_fit_isolation(const fs::path &root) {
  Fixture fixture(root/"view-inputs");fixture.save();
  auto ordinary=configuration(root/"view-ordinary",fixture);ev::run_archive_readout(ordinary);
  test::check(read(root/"view-ordinary/archive-readout-card.json").find("validation_views")==std::string::npos &&
      read(root/"view-ordinary/report.json").find("validation_views")==std::string::npos,"empty view option changed historical schema");
  const embedding::Batch intact{torch::where(fixture.validation.feature_mask,fixture.validation.data,
      torch::zeros_like(fixture.validation.data)),fixture.validation.feature_mask.clone()};
  const auto identity=save_view(fixture,"identity",intact,fixture.native_validation,fixture.validation_labels);
  const embedding::Batch absent{torch::zeros_like(intact.data),torch::zeros_like(intact.feature_mask)};
  const ev::FeatureSurface absent_features{torch::full_like(fixture.native_validation.values,std::numeric_limits<double>::quiet_NaN()),
      torch::zeros_like(fixture.native_validation.valid),fixture.native_validation.provenance};
  const auto empty=save_view(fixture,"absent",absent,absent_features,fixture.validation_labels);
  auto reduced=fixture.native_validation;reduced.valid=reduced.valid.clone();reduced.values=reduced.values.clone();
  reduced.valid.narrow(0,0,4).fill_(false);reduced.values.narrow(0,0,4).fill_(std::numeric_limits<double>::quiet_NaN());
  const auto unequal=save_view(fixture,"unequal",intact,reduced,fixture.validation_labels);
  auto run=configuration(root/"view-results",fixture);run.inputs[0].validation_views={identity,empty,unequal};
  const auto input_before=read(identity.observations),features_before=read(identity.features);RuntimeState runtime;
  ev::run_archive_readout(run);runtime.unchanged();
  test::check(read(identity.observations)==input_before && read(identity.features)==features_before,"view input archives changed");
  const auto result=root/"view-results/dummy",base=root/"view-ordinary/dummy";
  for(const auto &method:{"raw","pca_only","native"}) {
    compare_fit(base/"rep-1"/(std::string(method)+"-fit.pt"),result/"rep-1"/(std::string(method)+"-fit.pt"));
    const auto normal=result/"rep-1"/(std::string(method)+"-validation-predictions.pt");
    const auto view=result/"rep-1/validation-views/identity"/(std::string(method)+"-predictions.pt");
    for(const auto &key:{"ridge","tiny_secondary","valid","probe_features","labels_scoring_only","source_ids_json"})
      test::check(torch::equal(tensor(normal,key),tensor(view,key)),"identity view changed exact frozen predictions/row lineage");
    test::check(!fs::exists(result/"rep-1/validation-views/identity"/(std::string(method)+"-fit.pt")),"validation view fabricated a refit artifact");
    test::check(tensor(result/"rep-1/validation-views/absent"/(std::string(method)+"-predictions.pt"),"valid").sum().item<int64_t>()==0,
        "all-absent validation view acquired support");
  }
  for(const auto &key:{"mean","scale","counts"})test::check(torch::equal(tensor(base/"raw-scaler.pt",key),tensor(result/"raw-scaler.pt",key)),"view changed TRAIN observation scaler");
  for(const auto &key:{"pca_mean","pca_components","pca_singular_values","pca_numerical_rank","fitted_rows"})
    test::check(torch::equal(tensor(base/"pca-only.pt",key),tensor(result/"pca-only.pt",key)),"view changed TRAIN PCA tensors");
  const auto report=read(root/"view-results/report.json"),card=read(root/"view-results/archive-readout-card.json");
  test::check(card.find("validation_view_policy")!=std::string::npos &&
      card.find("same tensors; no view fit")!=std::string::npos && report.find("view_minus_ordinary_full_population_interval")!=std::string::npos,
      "view recipe/effects were not frozen and reported");
  const auto start=report.find("\"view\":{\"id\":\"absent\"");test::check(start!=std::string::npos,"absent view was silently omitted");
  const auto effect=report.find("\"view_minus_ordinary_full_population_interval\":",start);
  const auto estimate=report.find("\"estimate\":",effect);
  const auto base_predictions=tensor(base/"rep-1/raw-validation-predictions.pt","ridge");
  const auto base_valid=tensor(base/"rep-1/raw-validation-predictions.pt","valid");
  const auto expected=-double(ev::score(base_predictions,fixture.validation_labels,base_valid).correct)/16;
  test::check(std::abs(std::stod(report.substr(estimate+11))-expected)<1e-12,
      "full-population view effect dropped abstentions or changed denominator");
  test::check(tensor(result/"rep-1/validation-views/unequal/native-predictions.pt","valid").sum().item<int64_t>()==10 &&
      tensor(result/"rep-1/validation-views/unequal/raw-predictions.pt","valid").sum().item<int64_t>()==14,
      "unequal native view support changed raw coverage");
  // A view's legal frozen native output can change substantially without changing
  // any ordinary TRAIN fit or base prediction. Its observation lineage stays fixed.
  auto changed=fixture.native_validation;changed.values=changed.values.clone()*17.0+37.0;
  const auto perturb=save_view(fixture,"perturbed",intact,changed,fixture.validation_labels);
  auto changed_run=configuration(root/"view-changed-results",fixture);changed_run.inputs[0].validation_views={perturb};ev::run_archive_readout(changed_run);
  for(const auto &method:{"raw","pca_only","native"})
    compare_fit(base/"rep-1"/(std::string(method)+"-fit.pt"),root/"view-changed-results/dummy/rep-1"/(std::string(method)+"-fit.pt"));
  test::check(!torch::equal(tensor(result/"rep-1/validation-views/identity/native-predictions.pt","probe_features"),
      tensor(root/"view-changed-results/dummy/rep-1/validation-views/perturbed/native-predictions.pt","probe_features")),"view perturbation fixture ineffective");
}
void validation_view_rejections_and_unsupported(const fs::path &root) {
  Fixture fixture(root/"bad-view-inputs");fixture.save();
  const embedding::Batch intact{torch::where(fixture.validation.feature_mask,fixture.validation.data,torch::zeros_like(fixture.validation.data)),
      fixture.validation.feature_mask.clone()};
  const auto attempt=[&](const std::string &id,const embedding::Batch &batch,const torch::Tensor &labels,const std::string &prefix="validation") {
    auto run=configuration(root/("bad-view-"+id),fixture);
    run.inputs[0].validation_views={save_view(fixture,id,batch,fixture.native_validation,labels,prefix)};
    RuntimeState runtime;rejects([&]{ev::run_archive_readout(run);},"illegal validation view accepted: "+id);runtime.unchanged();
    test::check(!fs::exists(fs::path(run.output_directory)/"dummy"),"view validation happened after ordinary fits");
  };
  attempt("labels",intact,1-fixture.validation_labels);
  attempt("sources",intact,fixture.validation_labels,"other-validation");
  auto introduced=intact;introduced.data=intact.data.clone();introduced.feature_mask=intact.feature_mask.clone();
  introduced.feature_mask.narrow(0,0,2).select(1,0).select(1,0).fill_(true);introduced.data[0][0][0][0]=1;
  attempt("support",introduced,fixture.validation_labels);
  auto changed=intact;changed.data=intact.data.clone();changed.data[0][0][1][0]+=1;
  attempt("retained-value",changed,fixture.validation_labels);
  changed=intact;changed.data=intact.data.clone();changed.data[0][0][0][0]=1;
  attempt("hidden-value",changed,fixture.validation_labels);
  auto wrong=configuration(root/"bad-view-hash",fixture);
  wrong.inputs[0].validation_views={save_view(fixture,"hash",intact,fixture.native_validation,fixture.validation_labels)};
  wrong.inputs[0].validation_views[0].observations_sha256=std::string(64,'0');rejects([&]{ev::run_archive_readout(wrong);},"view hash mismatch accepted");
  test::check(!fs::exists(wrong.output_directory),"view hash guard ran after deserialization");
  wrong.output_directory=(root/"bad-view-testing").string();wrong.inputs[0].validation_views[0].observations_sha256.clear();
  wrong.inputs[0].validation_views[0].observations=(fixture.directory/"controlled-testing.pt").string();rejects([&]{ev::run_archive_readout(wrong);},"testing-marked view accepted");
  test::check(!fs::exists(wrong.output_directory),"view TEST guard ran after decoding");
  Fixture rank(root/"view-rank-inputs");rank.training.feature_mask.fill_(true);
  rank.training.data=rank.labels.to(torch::kFloat64).mul(2).sub(1).reshape({16,1,1,1}).expand({16,2,4,1}).clone();
  rank.native_training.valid.fill_(true);rank.native_training.values.narrow(0,14,2).fill_(0);rank.save();
  const embedding::Batch rank_view{torch::where(rank.validation.feature_mask,rank.validation.data,torch::zeros_like(rank.validation.data)),rank.validation.feature_mask.clone()};
  auto rank_run=configuration(root/"view-rank-results",rank);
  rank_run.inputs[0].validation_views={save_view(rank,"rank-view",rank_view,rank.native_validation,rank.validation_labels)};ev::run_archive_readout(rank_run);
  test::check(fs::exists(root/"view-rank-results/dummy/rep-1/validation-views/rank-view/native-predictions.pt") &&
      fs::exists(root/"view-rank-results/dummy/rep-1/validation-views/rank-view/raw-predictions.pt") &&
      !fs::exists(root/"view-rank-results/dummy/rep-1/validation-views/rank-view/pca_only-predictions.pt") &&
      read(root/"view-rank-results/report.json").find("PCA dimensions exceed numerical training rank")!=std::string::npos,
      "unsupported PCA view disabled legal methods or fabricated predictions");
}
} // namespace

int main() {
  try {
    const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root=fs::temp_directory_path()/("archive-readout-test-"+std::to_string(nonce));fs::create_directories(root);
    normal_and_isolation(root);unsupported_and_abstention(root);rejections(root);known_sha256(root);
    validation_views_and_fit_isolation(root);validation_view_rejections_and_unsupported(root);
    std::cout << "archive readout tests passed: " << root.string() << '\n';return 0;
  } catch(const std::exception &error) {std::cerr << error.what() << '\n';return 1;}
}
