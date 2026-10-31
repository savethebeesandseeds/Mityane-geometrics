#include "state_fusion/model.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double kClose = 1e-4;
constexpr double kGap = 1e-3;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

state_fusion::SetBatch clone_batch(const state_fusion::SetBatch& batch) {
  state_fusion::SetBatch out;
  out.scalars = batch.scalars.clone();
  out.scalar_observed = batch.scalar_observed.clone();
  out.categoricals = batch.categoricals.clone();
  out.categorical_observed = batch.categorical_observed.clone();
  out.semantic = batch.semantic.clone();
  out.semantic_observed = batch.semantic_observed.clone();
  out.record_observed = batch.record_observed.clone();
  return out;
}

state_fusion::SetBatch permute_records(const state_fusion::SetBatch& batch, const torch::Tensor& perm) {
  state_fusion::SetBatch out;
  out.scalars = batch.scalars.index_select(1, perm);
  out.scalar_observed = batch.scalar_observed.index_select(1, perm);
  out.categoricals = batch.categoricals.index_select(1, perm);
  out.categorical_observed = batch.categorical_observed.index_select(1, perm);
  out.semantic = batch.semantic.index_select(1, perm);
  out.semantic_observed = batch.semantic_observed.index_select(1, perm);
  out.record_observed = batch.record_observed.index_select(1, perm);
  return out;
}

double max_abs(const torch::Tensor& left, const torch::Tensor& right) {
  require(torch::isfinite(left).all().item<bool>() && torch::isfinite(right).all().item<bool>(),
          "comparison received a non-finite answer");
  return (left - right).abs().max().item<double>();
}

bool rejects(const std::function<void()>& call, const std::string& fragment) {
  try {
    call();
  } catch (const std::invalid_argument& error) {
    return std::string(error.what()).find(fragment) != std::string::npos;
  }
  return false;
}

state_fusion::SetBatch observed_set(const state_fusion::Config& config, int64_t rows, int64_t records,
                                    torch::Device device) {
  const auto floats = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto flags = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto ids = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());
  state_fusion::SetBatch batch;
  batch.scalars = torch::randn({rows, records, config.scalar_features}, floats);
  batch.scalar_observed = torch::ones({rows, records, config.scalar_features}, flags);
  batch.semantic = torch::randn({rows, records, config.semantic_width}, floats);
  batch.semantic_observed = torch::ones({rows, records}, flags);
  batch.record_observed = torch::ones({rows, records}, flags);
  batch.categoricals = torch::zeros({rows, records, fields}, ids);
  batch.categorical_observed = torch::ones({rows, records, fields}, flags);
  for (int64_t field = 0; field < fields; ++field) {
    const auto card = config.categorical_cardinalities[static_cast<size_t>(field)];
    batch.categoricals.select(2, field).copy_(torch::randint(0, card, {rows, records}, ids));
  }
  return batch;
}

// A query row with every field unobserved. Stored numbers are garbage so a
// hole that leaks into the token changes the answer.
state_fusion::SetBatch empty_query(const state_fusion::Config& config, int64_t rows, int64_t queries,
                                   torch::Device device) {
  const auto floats = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto flags = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto ids = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());
  state_fusion::SetBatch batch;
  batch.scalars = torch::full({rows, queries, config.scalar_features}, 1000.f, floats);
  batch.scalar_observed = torch::zeros({rows, queries, config.scalar_features}, flags);
  batch.semantic = torch::full({rows, queries, config.semantic_width}, 1000.f, floats);
  batch.semantic_observed = torch::zeros({rows, queries}, flags);
  batch.record_observed = torch::ones({rows, queries}, flags);
  batch.categoricals = torch::full({rows, queries, fields}, static_cast<int64_t>(888888), ids);
  batch.categorical_observed = torch::zeros({rows, queries, fields}, flags);
  return batch;
}

// Place and kind are observed. Every other field stays a hole.
state_fusion::SetBatch place_kind_query(const state_fusion::Config& config, int64_t rows,
                                        torch::Device device) {
  auto batch = empty_query(config, rows, 1, device);
  batch.categoricals.select(2, 0).fill_(1);
  batch.categorical_observed.select(2, 0).fill_(true);
  batch.categoricals.select(2, 1).fill_(2);
  batch.categorical_observed.select(2, 1).fill_(true);
  return batch;
}

}  // namespace

int main() {
  try {
    require(torch::cuda::is_available(), "CUDA is required for the state fusion query test");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    torch::manual_seed(2101);
    const auto device = torch::Device(torch::kCUDA, 0);
    const auto index = torch::TensorOptions().dtype(torch::kLong).device(device);

    state_fusion::Config config;
    state_fusion::StateFusionModel model(config);
    model->to(device);
    model->eval();

    constexpr int64_t kRows = 4;
    constexpr int64_t kRecords = 5;
    const auto records = observed_set(config, kRows, kRecords, device);
    const auto question = place_kind_query(config, kRows, device);
    double permutation_gap = 0;
    double hole_gap = 0;
    double zero_gap = 0;
    double filter_gap = 0;
    double cached_gap = 0;
    double two_gap = 0;
    {
      torch::NoGradGuard guard;
      const auto state = model->encode(records);
      const auto answer = model->query(state, question);
      require(answer.is_cuda(), "query left CUDA");
      require(answer.size(0) == kRows && answer.size(1) == 1 && answer.size(2) == config.d_model,
              "query shape is not [batch, queries, d_model]");
      require(torch::isfinite(answer).all().item<bool>(), "query answer is not finite");

      const auto perm = torch::tensor({2, 0, 4, 1, 3}, index);
      const auto permuted = model->encode(permute_records(records, perm));
      permutation_gap = max_abs(answer, model->query(permuted, question));
      require(permutation_gap <= kClose,
              "permuting records changed the query answer by " + std::to_string(permutation_gap));

      auto holes = clone_batch(question);
      holes.scalars.masked_fill_(question.scalar_observed.logical_not(),
                                 std::numeric_limits<float>::quiet_NaN());
      holes.semantic.masked_fill_(question.semantic_observed.logical_not().unsqueeze(-1),
                                  std::numeric_limits<float>::quiet_NaN());
      holes.categoricals.masked_fill_(question.categorical_observed.logical_not(), static_cast<int64_t>(1));
      hole_gap = max_abs(answer, model->query(state, holes));
      require(hole_gap <= kClose, "editing a query hole changed the answer by " + std::to_string(hole_gap));

      auto observed_zero = clone_batch(question);
      observed_zero.scalars.select(2, 0).fill_(0);
      observed_zero.scalar_observed.select(2, 0).fill_(true);
      auto zero_hole = clone_batch(observed_zero);
      zero_hole.scalar_observed.select(2, 0).fill_(false);
      zero_gap = max_abs(model->query(state, observed_zero), model->query(state, zero_hole));
      require(zero_gap > kGap, "an observed zero matched a query hole; gap " + std::to_string(zero_gap));

      const auto blank = empty_query(config, kRows, 1, device);
      filter_gap = max_abs(answer, model->query(state, blank));
      require(filter_gap > kGap,
              "a place-and-kind query matched an empty query; gap " + std::to_string(filter_gap));

      auto both = empty_query(config, kRows, 2, device);
      both.categoricals.select(1, 0).select(1, 0).fill_(1);
      both.categorical_observed.select(1, 0).select(1, 0).fill_(true);
      both.categoricals.select(1, 0).select(1, 1).fill_(2);
      both.categorical_observed.select(1, 0).select(1, 1).fill_(true);
      const auto paired = model->query(state, both);
      require(paired.size(1) == 2, "two queries did not return two answers");
      const double paired_gap = max_abs(paired.select(1, 0), paired.select(1, 1));
      require(paired_gap > kGap, "paired filter and empty queries matched; gap " + std::to_string(paired_gap));

      auto unobserved = clone_batch(blank);
      unobserved.record_observed.fill_(false);
      const double record_gap = max_abs(model->query(state, blank), model->query(state, unobserved));
      require(record_gap <= kClose,
              "an unobserved query row left the absent-field token; gap " + std::to_string(record_gap));

      const auto cached = state.detach().clone();
      cached_gap = max_abs(answer, model->query(cached, question));
      require(cached_gap <= kClose, "a cloned state changed the answer by " + std::to_string(cached_gap));

      state_fusion::Config two_seeds = config;
      two_seeds.pma_seeds = 2;
      state_fusion::StateFusionModel second(two_seeds);
      second->to(device);
      second->eval();
      const auto two_state = second->encode(records);
      const auto two_answer = second->query(two_state, question);
      require(two_answer.size(1) == 1 && two_state.size(1) == 2, "two-seed query shape is wrong");
      two_gap = max_abs(two_answer, second->query(second->encode(permute_records(records, perm)), question));
      require(two_gap <= kClose, "two-seed permutation changed the query answer by " + std::to_string(two_gap));
    }

    auto on_cpu = clone_batch(question);
    on_cpu.scalars = on_cpu.scalars.to(torch::kCPU);
    on_cpu.scalar_observed = on_cpu.scalar_observed.to(torch::kCPU);
    on_cpu.categoricals = on_cpu.categoricals.to(torch::kCPU);
    on_cpu.categorical_observed = on_cpu.categorical_observed.to(torch::kCPU);
    on_cpu.semantic = on_cpu.semantic.to(torch::kCPU);
    on_cpu.semantic_observed = on_cpu.semantic_observed.to(torch::kCPU);
    on_cpu.record_observed = on_cpu.record_observed.to(torch::kCPU);
    torch::Tensor state;
    {
      torch::NoGradGuard guard;
      state = model->encode(records);
    }
    require(rejects([&] { model->query(state, on_cpu); }, "CUDA"), "a CPU query batch was accepted");
    require(rejects([&] { model->query(state.to(torch::kCPU), question); }, "CUDA"),
            "a CPU state was accepted");

    state_fusion::StateFusionModel cpu_model(config);
    require(rejects([&] { cpu_model->query(state, question); }, "CUDA"), "CPU parameters were accepted");

    auto no_queries = empty_query(config, kRows, 1, device);
    no_queries.scalars = no_queries.scalars.narrow(1, 0, 0);
    no_queries.scalar_observed = no_queries.scalar_observed.narrow(1, 0, 0);
    no_queries.categoricals = no_queries.categoricals.narrow(1, 0, 0);
    no_queries.categorical_observed = no_queries.categorical_observed.narrow(1, 0, 0);
    no_queries.semantic = no_queries.semantic.narrow(1, 0, 0);
    no_queries.semantic_observed = no_queries.semantic_observed.narrow(1, 0, 0);
    no_queries.record_observed = no_queries.record_observed.narrow(1, 0, 0);
    require(rejects([&] { model->query(state, no_queries); }, "query"), "a batch with no queries was accepted");

    std::cout << "PASS state_fusion_query device=cuda:0"
              << " permutation_max_abs=" << permutation_gap
              << " hole_max_abs=" << hole_gap
              << " observed_zero_gap=" << zero_gap
              << " filter_gap=" << filter_gap
              << " cached_max_abs=" << cached_gap
              << " two_seed_permutation_max_abs=" << two_gap << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << "\n";
    return 1;
  }
}
