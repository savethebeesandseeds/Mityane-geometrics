// Extracted from cuwacunu_torch; see code/data/README.md for provenance.
/* memory_mapped_dataloader.h */
#pragma once

#include <optional>
#include <mutex>
#include <torch/torch.h>
#include <utility>
#include <vector>

#include "cuwacunu/data/record_types.h"
#include "cuwacunu/data/interval.h"
#include "cuwacunu/data/support.h"

#include "cuwacunu/data/source_config.h"
#include "cuwacunu/data/sample.h"
#include "cuwacunu/data/memory_mapped_dataset.h"

namespace cuwacunu::data {

using ::cuwacunu::data::edge_sample_t;

/**
 * @brief A class that creates and manages a DataLoader for memory-mapped
 * datasets.
 *
 * @tparam Dataset_t The dataset type. This dataset should be compatible with
 * torch::data::Dataset and return samples of type Sample_t from its
 * `get(...)` method.
 * @tparam Sample_t The sample type returned by the dataset (e.g., a struct
 * containing features and mask tensors).
 * @tparam Datatype_t The underlying data type used by the dataset, if required.
 * @tparam Sampler_t The type of Sampler used
 *
 * This class:
 *  - Constructs the dataset Dataset_t internally (must adjust the constructor
 * call).
 *  - Accepts a custom collate function as a constructor argument.
 *  - Allows specifying batch size, number of workers, and optionally, a
 * different sampler type.
 *  - Provides begin() and end() to iterate over batches and a reset() method to
 * re-start iteration.
 */
template <typename Dataset_t, typename Sample_t, typename Datatype_t,
          typename Sampler_t = torch::data::samplers::SequentialSampler>
class MemoryMappedDataLoader {
private:
  /* Define the DataLoader type with the chosen Sampler and sample type
   * Sample_t */
  using DataLoaderType = torch::data::StatelessDataLoader<Dataset_t, Sampler_t>;
  DataLoaderType data_loader_;

  /* Static assertions to validate template parameters */
  static_assert(
      std::is_base_of<torch::data::datasets::Dataset<Dataset_t, Sample_t>,
                      Dataset_t>::value,
      "(memory_mapped_dataloader)[] Dataset_t must derive from "
      "torch::data::Dataset");
  static_assert(
      std::is_base_of<torch::data::samplers::Sampler<>, Sampler_t>::value,
      "(memory_mapped_dataloader)[] Sampler_t must derive from "
      "torch::data::samplers::Sampler");

public:
  // int64_t B; /* batch size */ // Sampler does not guarantee a constant batch
  // size.
  int64_t C_;  /* channel count */
  int64_t Hx_; /* input horizon */
  int64_t D_;  /* feature width */

  /**
   * @brief Construct a new MemoryMappedDataLoader object.
   *
   * @param memory_mapped_dataset Dataset.
   * @param sampler The sampler to be used.
   * @param options DataLoaderOptions to configure the DataLoader.
   *
   * Note:
   * - Ensure that Dataset_t has a constructor that matches the arguments
   * provided here.
   * - The dataset Dataset_t should implement size() to return the number of
   * samples.
   */
  MemoryMappedDataLoader(Dataset_t &memory_mapped_dataset,
                         const Sampler_t &sampler,
                         const torch::data::DataLoaderOptions &options)
      : data_loader_(memory_mapped_dataset, sampler, options) {
    const auto dataset_size = memory_mapped_dataset.size();
    if (!dataset_size.has_value() || dataset_size.value() == 0) {
      C_ = 0;
      Hx_ = 0;
      D_ = 0;
      log_warn(
          "[MemoryMappedDataLoader] Dataset is empty; shape probe skipped.\n");
      return;
    }
    Sample_t probe_sample(memory_mapped_dataset.get(0));
    C_ = probe_sample.features.size(0);
    Hx_ = probe_sample.features.size(1);
    D_ = probe_sample.features.size(2);
    maybe_emit_policy_warnings_(C_);

    probe_sample.reset();
  }

  auto begin() { return data_loader_.begin(); }
  auto end() { return data_loader_.end(); }
  void reset() { data_loader_.reset(); }

  DataLoaderType &inner() { return data_loader_; }
  const DataLoaderType &inner() const { return data_loader_; }

private:
  static void maybe_emit_policy_warnings_(int64_t channels) {
    static std::once_flag multi_channel_warning;
    static std::once_flag high_channel_warning;

    if (channels > 1) {
      std::call_once(multi_channel_warning, [] {
#if (CUWACUNU_EDGE_DATASET_GRID_STEP_POLICY == CUWACUNU_EDGE_DATASET_GRID_STEP_POLICY_MIN)
        constexpr const char *grid = "finest";
#else
        constexpr const char *grid = "coarsest";
#endif
        log_warn("[MemoryMappedDataLoader] The global anchor grid uses the %s source cadence; "
                 "native channel timestamps are retained and encoder alignment must be checked separately.\n", grid);
      });
    }

    if (channels > 16) {
      std::call_once(high_channel_warning, [channels] { log_warn(
          "[MemoryMappedDataLoader] High channel count detected (%lld); "
          "multi-channel loading is supported, but validate that your channel "
          "protocol policy matches experiment goals.\n",
          static_cast<long long>(channels)); });
    }
  }
};

/**
 * @brief Method to create a dataloader from an explicit source materialization
 * request.
 *
 * @tparam Dataset_t The dataset type. This dataset should be compatible with
 * torch::data::Dataset and return samples of type Sample_t from its
 * `get(...)` method.
 * @tparam Sample_t The sample type returned by the dataset (e.g., a struct
 * containing features and mask tensors).
 * @tparam Datatype_t The underlying data type used by the dataset, if required.
 * @tparam Sampler_t The type of Sampler used.
 */
template <typename Dataset_t, typename Sample_t, typename Datatype_t,
          typename Sampler_t>
typename std::enable_if<
    std::is_same<Sampler_t, torch::data::samplers::SequentialSampler>::value,
    MemoryMappedDataLoader<Dataset_t, Sample_t, Datatype_t, Sampler_t>>::type
create_memory_mapped_dataloader(
    const cuwacunu::data::instrument_signature_t
        &edge_signature,
    const source_materialization_request_t &materialization_request,
    bool force_rebuild_cache = false, std::size_t batch_size = 64,
    std::size_t workers = 0,
    torch::ScalarType feature_dtype = torch::kFloat32) {
  /* variables */
  auto dataset = create_memory_mapped_edge_dataset<Datatype_t>(
      edge_signature, materialization_request, force_rebuild_cache, feature_dtype);
  auto sampler = dataset.SequentialSampler();
  auto sampler_options = dataset.SequentialSampler_options(batch_size, workers);
  /* dataloader */
  return cuwacunu::data::
      MemoryMappedDataLoader<Dataset_t, Sample_t, Datatype_t, Sampler_t>(
          /* dataset      */ dataset,
          /* sampler      */ sampler,
          /* options      */ sampler_options);
}

template <typename Dataset_t, typename Sample_t, typename Datatype_t,
          typename Sampler_t>
typename std::enable_if<
    std::is_same<Sampler_t, torch::data::samplers::RandomSampler>::value,
    MemoryMappedDataLoader<Dataset_t, Sample_t, Datatype_t, Sampler_t>>::type
create_memory_mapped_dataloader(
    const cuwacunu::data::instrument_signature_t
        &edge_signature,
    const source_materialization_request_t &materialization_request,
    bool force_rebuild_cache = false, std::size_t batch_size = 64,
    std::size_t workers = 0,
    torch::ScalarType feature_dtype = torch::kFloat32) {
  /* variables */
  auto dataset = create_memory_mapped_edge_dataset<Datatype_t>(
      edge_signature, materialization_request, force_rebuild_cache, feature_dtype);
  auto sampler = dataset.RandomSampler();
  auto sampler_options = dataset.RandomSampler_options(batch_size, workers);
  /* dataloader */
  return cuwacunu::data::
      MemoryMappedDataLoader<Dataset_t, Sample_t, Datatype_t, Sampler_t>(
          /* dataset      */ dataset,
          /* sampler      */ sampler,
          /* options      */ sampler_options);
}

// ----------------------------------------------------------------------------
// Compatibility source-spec overloads.




// ----------------------------------------------------------------------------
// Explicit source materialization DataLoaders (sequential / random)
//
//   auto sequential = make_sequential_dataloader<Datatype_t>(edge_signature,
//                                                            spec);
//   auto random = make_random_dataloader<Datatype_t>(edge_signature, spec);
// ----------------------------------------------------------------------------
template <typename Datatype_t, typename Sampler>
inline auto make_dataloader_with_sampler(
    const cuwacunu::data::instrument_signature_t
        &edge_signature,
    const source_materialization_request_t &materialization_request,
    bool force_rebuild_cache = false, std::size_t batch_size = 64,
    std::size_t workers = 0,
    torch::ScalarType feature_dtype = torch::kFloat32) {
  using Dataset = MemoryMappedEdgeDataset<Datatype_t>;

  return create_memory_mapped_dataloader<Dataset, edge_sample_t, Datatype_t,
                                         Sampler>(
      edge_signature, materialization_request, force_rebuild_cache, batch_size,
      workers, feature_dtype);
}



/**
 * ---------------------------------- sequential ----------------------------
 * @tparam Datatype_t The underlying data type used by the dataset.
 */
template <typename Datatype_t>
inline auto make_sequential_dataloader(
    const cuwacunu::data::instrument_signature_t
        &edge_signature,
    const source_materialization_request_t &materialization_request,
    bool force_rebuild_cache = false, std::size_t batch_size = 64,
    std::size_t workers = 0,
    torch::ScalarType feature_dtype = torch::kFloat32) {
  using Sampler = torch::data::samplers::SequentialSampler;
  return make_dataloader_with_sampler<Datatype_t, Sampler>(
      edge_signature, materialization_request, force_rebuild_cache, batch_size,
      workers, feature_dtype);
}



/**
 * ---------------------------------- random --------------------------------
 * @tparam Datatype_t The underlying data type used by the dataset.
 */
template <typename Datatype_t>
inline auto make_random_dataloader(
    const cuwacunu::data::instrument_signature_t
        &edge_signature,
    const source_materialization_request_t &materialization_request,
    bool force_rebuild_cache = false, std::size_t batch_size = 64,
    std::size_t workers = 0,
    torch::ScalarType feature_dtype = torch::kFloat32) {
  using Sampler = torch::data::samplers::RandomSampler;
  return make_dataloader_with_sampler<Datatype_t, Sampler>(
      edge_signature, materialization_request, force_rebuild_cache, batch_size,
      workers, feature_dtype);
}



} // namespace cuwacunu::data
