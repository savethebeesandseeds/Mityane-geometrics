// SPDX-License-Identifier: MIT
#include "embedding/shared/projection_diagnostic.h"
#include "shared_test_support.h"
#include <ATen/Context.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
  std::ifstream input(path, std::ios::binary);
  test::check(bool(input), "missing artifact: " + path.string());
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}
torch::Tensor tensor(const fs::path &path, const std::string &key) {
  torch::serialize::InputArchive input;
  input.load_from(path.string(), torch::kCPU);
  torch::Tensor out;
  input.read(key, out, true);
  return out;
}
size_t occurrences(const std::string &text, const std::string &marker) {
  size_t count = 0, position = 0;
  while ((position = text.find(marker, position)) != std::string::npos) { ++count; position += marker.size(); }
  return count;
}
struct RngState {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RngState() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    for (size_t i = 0; i < generators.size(); ++i)
      test::check(torch::equal(generators[i].get_state(), states[i]), label + " changed ambient generator state");
  }
};
std::string sources(const std::string &prefix, int64_t pairs) {
  std::ostringstream out;
  out << '[';
  for (int64_t group = 0; group < pairs; ++group) {
    for (int64_t variant = 0; variant < 2; ++variant) {
      if (group || variant) out << ',';
      // Exercise JSON escapes and UTF-8 conversion without changing grouping.
      out << '"' << prefix << "\\u03b1-\\\"source-" << group << '"';
    }
  }
  return out.str() + ']';
}
struct Fixture {
  fs::path directory;
  torch::Tensor training, validation, labels, valid;
  explicit Fixture(const fs::path &path) : directory(path) {
    fs::create_directories(directory);
    // Eight valid training rows: three independent dimensions plus a constant
    // column. Last opposite-label pair is invalid and has NaN storage.
    training = torch::tensor({
        {-1.0,-1.0,-1.0,7.0},{1.0,-1.0,-1.0,7.0},
        {-1.0,1.0,-1.0,7.0},{1.0,1.0,-1.0,7.0},
        {-1.0,-1.0,1.0,7.0},{1.0,-1.0,1.0,7.0},
        {-1.0,1.0,1.0,7.0},{1.0,1.0,1.0,7.0},
        {0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0}}, torch::kFloat64);
    training.narrow(0,8,2).fill_(std::numeric_limits<double>::quiet_NaN());
    validation = training.clone();
    validation.narrow(0,0,8).narrow(1,0,3).mul_(0.7);
    labels = torch::tensor({0,1,0,1,0,1,0,1,0,1},torch::kInt64);
    valid = torch::tensor({true,true,true,true,true,true,true,true,false,false},torch::kBool);
  }
  void save_features(const fs::path &path, const torch::Tensor &values, const torch::Tensor &mask) const {
    torch::serialize::OutputArchive output;
    output.write("features",values,true); output.write("valid",mask,true);
    output.write("provenance",embedding::archive::text_tensor("dummy frozen checkpoint; train-fitted preparation"),true);
    embedding::archive::save_archive(path.string(),output);
  }
  void save_labels(const fs::path &path, const std::string &prefix) const {
    torch::serialize::OutputArchive output;
    output.write("labels_scoring_only",labels,true);
    output.write("source_ids_json",embedding::archive::text_tensor(sources(prefix,5)),true);
    embedding::archive::save_archive(path.string(),output);
  }
  void save() const {
    save_features(directory/"features-training.pt",training,valid);
    save_features(directory/"features-validation.pt",validation,valid);
    save_labels(directory/"controlled-training.pt","train");
    save_labels(directory/"controlled-validation.pt","validation");
    // Invalid sentinel must never be discovered/opened by the diagnostic.
    std::ofstream sentinel(directory/"controlled-testing.pt",std::ios::binary);
    sentinel << "NOT A TORCH ARCHIVE; TEST ACCESS IS FORBIDDEN";
  }
  ev::ProjectionArchiveInput input(const std::string &id = "dummy") const {
    ev::ProjectionArchiveInput out;
    out.id=id; out.architecture="dummy_encoder"; out.master_seed=901; out.checkpoint_steps=128;
    out.training_features=(directory/"features-training.pt").string();
    out.validation_features=(directory/"features-validation.pt").string();
    out.training_labels=(directory/"controlled-training.pt").string();
    out.validation_labels=(directory/"controlled-validation.pt").string();
    return out;
  }
};
ev::ProjectionDiagnosticRun configuration(const fs::path &output, const Fixture &fixture) {
  ev::ProjectionDiagnosticRun run;
  run.output_directory=output.string(); run.inputs={fixture.input()};
  run.source_fingerprint="dummy-source"; run.git_head="dummy-head"; run.git_dirty="true";
  run.widths={2,3,4}; run.repetitions={{"rep-1",1701,2701},{"rep-2",1802,2802}};
  run.bootstrap_replicates=100;
  return run;
}
void compare_fit(const fs::path &left, const fs::path &right) {
  for (const std::string key : {"feature_mean","feature_scale","fitted_rows","ridge_mean","ridge_scale","ridge_weights",
        "ridge_intercept","tiny_mean","tiny_scale","tiny_w1","tiny_b1","tiny_w2","tiny_b2","actual_probe_seed_decimal"})
    test::check(torch::equal(tensor(left,key),tensor(right,key)), "heldout values changed fitted tensor: " + key);
}
void primitive() {
  torch::manual_seed(937);
  RngState initial;
  ev::FrozenOrthonormalProjection first(7,3,1701), again(7,3,1701), other(7,3,1802);
  initial.unchanged("projection primitive");
  test::check(torch::equal(first.components,again.components),"same projection seed is not reproducible");
  test::check(!torch::equal(first.components,other.components),"different projection seeds did not change map");
  test::close(first.components.transpose(0,1).matmul(first.components),torch::eye(3,torch::kFloat64),
              "orthonormal projection",1e-12,1e-12);
  auto values=torch::arange(21,torch::kFloat64).reshape({3,7});
  values[2].fill_(std::numeric_limits<double>::quiet_NaN());
  ev::FeatureSurface surface{values,torch::tensor({true,true,false},torch::kBool),"frozen"};
  const auto original=surface.values.narrow(0,0,2).clone();
  const auto transformed=first.transform(surface);
  test::close(transformed.values.narrow(0,0,2),original.matmul(first.components),"random transform");
  test::check(transformed.values[2].eq(0).all().item<bool>() && torch::equal(transformed.valid,surface.valid),
              "invalid random rows were laundered or nonzero");
  test::check(torch::equal(surface.values.narrow(0,0,2),original) && torch::isnan(surface.values[2]).all().item<bool>(),
              "random transform mutated source features");
  rejects([] { ev::FrozenOrthonormalProjection projection(4,0,1); },"zero width accepted");
  rejects([] { ev::FrozenOrthonormalProjection projection(4,5,1); },"oversized width accepted");
  rejects([&] { first.transform({torch::ones({3,6},torch::kFloat64),surface.valid,"bad"}); },"native width mismatch accepted");
  auto invalid=surface;
  invalid.valid=torch::ones({3},torch::kBool);
  rejects([&] { first.transform(invalid); },"nonfinite valid projection input accepted");
}
void engine(const fs::path &root) {
  Fixture fixture(root/"frozen-inputs"); fixture.save();
  const auto training_bytes=read(fixture.directory/"features-training.pt");
  const auto validation_bytes=read(fixture.directory/"features-validation.pt");
  const auto training_label_bytes=read(fixture.directory/"controlled-training.pt");
  const auto validation_label_bytes=read(fixture.directory/"controlled-validation.pt");
  const auto test_sentinel=read(fixture.directory/"controlled-testing.pt");
  auto run=configuration(root/"ordinary",fixture);
  // A second checkpoint declaration uses identical raw data: fixed maps and
  // paired probe initialization must not depend on checkpoint/architecture ID.
  auto alternative=fixture.input("alternative");
  alternative.architecture="other_encoder"; alternative.checkpoint_steps=512;
  run.inputs.push_back(alternative);
  torch::manual_seed(844);
  RngState before;
  ev::run_projection_diagnostic(run);
  before.unchanged("projection diagnostic");
  const auto report=read(root/"ordinary"/"report.json");
  const auto card=read(root/"ordinary"/"projection-card.json");
  test::check(card.find("\"test_access\":false")!=std::string::npos && card.find("\"validation_selection\":\"none;")!=std::string::npos,
              "card omits validation-only/no-selection contract");
  test::check(card.find("\"updates\":100")!=std::string::npos && card.find("\"hidden\":16")!=std::string::npos,
              "fixed secondary recipe missing from pre-score card");
  test::check(occurrences(report,"\"numerical_training_rank\":3")==2,"valid-only centered rank is wrong");
  test::check(occurrences(report,"\"status\":\"unsupported_compression\"")==8,"rank-unsupported widths did not retain both methods/all repetitions");
  test::check(occurrences(report,"\"status\":\"measured\"")==20,"native/map/repetition entries were dropped");
  test::check(report.find("\"class_valid_rows\":[4,4]")!=std::string::npos &&
              report.find("\"complete_source_pairs\":4")!=std::string::npos &&
              report.find("\"source_groups\":4")!=std::string::npos, "group/class support did not exclude invalid pair");
  test::check(report.find("\"ridge_parameters\":10")!=std::string::npos && report.find("\"tiny_parameters\":114")!=std::string::npos,
              "native parameter count incorrect");
  const auto outer=root/"ordinary"/"dummy"/"outer-normalizer.pt";
  test::close(tensor(outer,"feature_mean"),torch::tensor({0.0,0.0,0.0,7.0},torch::kFloat64),"valid-only outer mean");
  test::check(tensor(outer,"fitted_rows").item<int64_t>()==8,"invalid rows entered normalizer fit");
  for (const std::string rep : {"rep-1","rep-2"}) {
    const auto left=root/"ordinary"/"dummy"/rep/"width-2",right=root/"ordinary"/"alternative"/rep/"width-2";
    test::check(torch::equal(tensor(left/"random-map.pt","random_components"),tensor(right/"random-map.pt","random_components")),
                "architecture/checkpoint changed fixed random map");
    compare_fit(left/"random-fit.pt",right/"random-fit.pt");
    test::check(torch::equal(tensor(left/"pca-fit.pt","actual_probe_seed_decimal"),tensor(left/"random-fit.pt","actual_probe_seed_decimal")),
                "same-width methods use different readout seeds");
    const auto prediction=tensor(left/"random-validation-predictions.pt","valid");
    test::check(torch::equal(prediction,fixture.valid),"projection prediction validity changed");
    test::check(tensor(left/"random-validation-predictions.pt","probe_features").narrow(0,8,2).eq(0).all().item<bool>(),
                "invalid feature rows passed into probe");
    const auto comparison=read(left/"comparison.json");
    test::check(comparison.find("\"candidate\":\"random\",\"comparator\":\"pca\"")!=std::string::npos &&
                comparison.find("\"valid_rows\":8")!=std::string::npos, "same-width paired comparison population missing");
  }
  test::check(!torch::equal(tensor(root/"ordinary"/"dummy"/"rep-1"/"width-2"/"random-map.pt","random_components"),
                          tensor(root/"ordinary"/"dummy"/"rep-2"/"width-2"/"random-map.pt","random_components")),
              "all projection repetitions reused one map");
  test::check(!fs::exists(root/"ordinary"/"dummy"/"rep-1"/"width-4"/"random-map.pt"),
              "rank-unsupported random method was fitted");
  test::check(read(fixture.directory/"features-training.pt")==training_bytes &&
              read(fixture.directory/"features-validation.pt")==validation_bytes &&
              read(fixture.directory/"controlled-training.pt")==training_label_bytes &&
              read(fixture.directory/"controlled-validation.pt")==validation_label_bytes &&
              read(fixture.directory/"controlled-testing.pt")==test_sentinel, "frozen input archive bytes changed");
  rejects([&] { ev::run_projection_diagnostic(run); },"existing output was overwritten");

  // Validation statistics and labels never influence fitting. Change heldout
  // values/labels/source IDs and verify every train-fit parameter/map exactly.
  Fixture shifted(root/"shifted-inputs");
  shifted.validation.narrow(0,0,8).narrow(1,0,3).mul_(70).add_(19);
  shifted.labels=1-shifted.labels;
  shifted.save();
  // Preserve training labels: only the validation labels are reversed.
  torch::serialize::OutputArchive labels;
  labels.write("labels_scoring_only",fixture.labels,true);
  labels.write("source_ids_json",embedding::archive::text_tensor(sources("train",5)),true);
  embedding::archive::save_archive((shifted.directory/"controlled-training.pt").string(),labels);
  auto shifted_run=configuration(root/"shifted",shifted);
  ev::run_projection_diagnostic(shifted_run);
  for (const std::string rep : {"rep-1","rep-2"}) {
    compare_fit(root/"ordinary"/"dummy"/rep/"native-fit.pt",root/"shifted"/"dummy"/rep/"native-fit.pt");
    for (const int width : {2,3}) {
      const auto a=root/"ordinary"/"dummy"/rep/("width-"+std::to_string(width));
      const auto b=root/"shifted"/"dummy"/rep/("width-"+std::to_string(width));
      compare_fit(a/"pca-fit.pt",b/"pca-fit.pt"); compare_fit(a/"random-fit.pt",b/"random-fit.pt");
      test::check(torch::equal(tensor(a/"pca-map.pt","pca_components"),tensor(b/"pca-map.pt","pca_components")) &&
                  torch::equal(tensor(a/"random-map.pt","random_components"),tensor(b/"random-map.pt","random_components")),
                  "validation distribution/labels changed fitted map");
    }
  }
}
void unsupported_and_validation(const fs::path &root) {
  Fixture empty(root/"empty-inputs"); empty.valid.zero_(); empty.save();
  auto run=configuration(root/"empty",empty);
  ev::run_projection_diagnostic(run);
  const auto report=read(root/"empty"/"report.json");
  test::check(occurrences(report,"\"status\":\"unsupported_fit\"")==14 && report.find("\"valid_rows\":0")!=std::string::npos,
              "all-invalid fit did not preserve complete declared results/coverage");
  test::check(!fs::exists(root/"empty"/"dummy"/"outer-normalizer.pt"),"all-invalid training created fitted statistics");

  Fixture onlyzero(root/"single-class-inputs");
  onlyzero.valid=onlyzero.labels.eq(0).logical_and(onlyzero.valid); onlyzero.save();
  run=configuration(root/"single-class",onlyzero);
  ev::run_projection_diagnostic(run);
  test::check(read(root/"single-class"/"report.json").find("\"status\":\"unsupported_fit\"")!=std::string::npos,
              "missing train class fitted a probe");

  Fixture small(root/"row-bound-inputs");
  small.valid.zero_(); small.valid.narrow(0,0,2).fill_(true); small.save();
  run=configuration(root/"row-bound",small);
  ev::run_projection_diagnostic(run);
  const auto small_report=read(root/"row-bound"/"report.json");
  test::check(small_report.find("\"valid_centered_training_row_bound\":1")!=std::string::npos &&
              occurrences(small_report,"\"status\":\"unsupported_compression\"")==12 &&
              occurrences(small_report,"\"status\":\"measured\"")==2,
              "valid centered row bound did not exclude compressed fits while retaining native");

  Fixture overlapping(root/"overlap-inputs"); overlapping.save();
  overlapping.save_labels(overlapping.directory/"controlled-validation.pt","train");
  run=configuration(root/"overlap",overlapping);
  RngState before;
  rejects([&] { ev::run_projection_diagnostic(run); },"overlapping train/validation source groups accepted");
  before.unchanged("failed projection diagnostic");
  test::check(fs::exists(root/"overlap"/"projection-card.json") && !fs::exists(root/"overlap"/"report.json"),
              "failed run did not freeze recipe before scoring");

  Fixture nonfinite(root/"nonfinite-inputs"); nonfinite.save();
  nonfinite.valid.fill_(true);
  nonfinite.save_features(nonfinite.directory/"features-training.pt",nonfinite.training,nonfinite.valid);
  run=configuration(root/"nonfinite",nonfinite);
  rejects([&] { ev::run_projection_diagnostic(run); },"nonfinite valid training feature accepted");

  run=configuration(root/"forbidden",overlapping);
  run.inputs[0].validation_labels=(overlapping.directory/"controlled-testing.pt").string();
  rejects([&] { ev::run_projection_diagnostic(run); },"testing-marked archive accepted");
  test::check(!fs::exists(root/"forbidden"),"forbidden test archive reached output/scoring");
  run=configuration(root/"same-archive",overlapping);
  run.inputs[0].validation_features=run.inputs[0].training_features;
  rejects([&] { ev::run_projection_diagnostic(run); },"one feature archive used for both training and validation");
  test::check(!fs::exists(root/"same-archive"),"conflicting archive roles reached fitting");
  run=configuration(root/"unsafe",overlapping); run.inputs[0].id="../escape";
  rejects([&] { ev::run_projection_diagnostic(run); },"unsafe input ID accepted");
  run=configuration(root/"duplicate-width",overlapping); run.widths={2,2};
  rejects([&] { ev::run_projection_diagnostic(run); },"duplicate widths accepted");
  run=configuration(root/"duplicate-rep",overlapping); run.repetitions.push_back(run.repetitions[0]);
  rejects([&] { ev::run_projection_diagnostic(run); },"duplicate repetition accepted");
}
} // namespace
int main() {
  try {
    torch::set_num_threads(1);
    const auto root=fs::temp_directory_path()/("projection-diagnostic-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    primitive(); engine(root); unsupported_and_validation(root);
    std::cout << "projection_diagnostic_test passed; artifacts " << root.string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "projection_diagnostic_test failed: " << error.what() << '\n';
    return 1;
  }
}
