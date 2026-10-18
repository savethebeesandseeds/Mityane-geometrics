// SPDX-License-Identifier: MIT
#pragma once
// Closed legal CPU observations only; labels are used by the fixed heads.
#include "embedding/shared/paired_pooling.h"
#include "frozen_role_guard.h"
#include <random>
#include <regex>
#include <sys/stat.h>

namespace multiband_screen {
namespace fs = std::filesystem;
namespace ev = embedding::evaluation;
using namespace ev::frozen_inputs;
enum class DatasetTask { slow_lag_sign, component_balance };
inline constexpr const char *data_recipe = "two-component-v1";
inline constexpr uint64_t quality_masters[] = {920903, 920904};
inline std::string task_name(DatasetTask task) {
  require(task == DatasetTask::slow_lag_sign || task == DatasetTask::component_balance, "closed task");
  return task == DatasetTask::slow_lag_sign ? "slow_lag_sign" : "component_balance";
}
inline std::string dataset_id(DatasetTask task) {
  return task_name(task) == "slow_lag_sign" ? "TEMPO-4" : "AMP-2";
}
inline int64_t task_bit(DatasetTask task) { return task_name(task) == "slow_lag_sign" ? 0 : 1; }
inline std::string helper_source(int64_t id, DatasetTask task) {
  return std::string(data_recipe) + "/" + task_name(task) + "/source-" + std::to_string(id);
}
inline std::string role_prefix(uint64_t master, DatasetTask task) {
  return "results/seed-" + std::to_string(master) + "-" + task_name(task) + "/";
}
inline torch::Tensor read(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor t; a.read(key, t, true); return t;
}
inline void direct_directory(const fs::path &path) {
  require(path.is_absolute() && fs::is_directory(path) && fs::canonical(path) == path,
          "canonical directory");
  for (auto p = path; !p.empty() && p != p.root_path(); p = p.parent_path())
    require(!fs::is_symlink(p), "direct directory ancestors");
}
inline void admit_files(const std::vector<fs::path> &paths) {
  std::set<fs::path> names; std::set<std::pair<dev_t, ino_t>> nodes;
  for (const auto &p : paths) {
    require(p.is_absolute() && fs::is_regular_file(p) && !fs::is_symlink(p) && fs::canonical(p) == p,
            "canonical regular input");
    direct_directory(p.parent_path()); struct stat s{};
    require(::stat(p.c_str(), &s) == 0 && s.st_nlink == 1 && names.insert(p).second &&
            nodes.emplace(s.st_dev, s.st_ino).second, "whole distinct single-link role matrix");
  }
}
inline void new_leaf(const fs::path &path) {
  require(path.is_absolute() && path.lexically_normal() == path && !path.filename().empty(),
          "absolute normalized output leaf");
  direct_directory(path.parent_path()); const fs::path bound("/embedding/output/runs/rpb-multiband-screen");
  const auto relative = path.lexically_relative(bound);
  require(!relative.empty() && !relative.is_absolute() && *relative.begin() != ".." && path != bound,
          "output in multiband protocol root");
  require(!fs::exists(path) && !fs::is_symlink(path) && fs::create_directory(path), "exclusive output leaf");
}
inline ev::ControlledDataset load_controlled(const fs::path &path, int64_t rows, uint64_t master,
                                            DatasetTask task, bool engineering = false) {
  require(rows == 256 || rows == 128, "fixed controlled population");
  require(engineering ? master == 910901 : master == 920903 || master == 920904, "declared master");
  const auto bit = task_bit(task); torch::serialize::InputArchive a; a.load_from(path.string(), torch::kCPU);
  const auto keys = a.keys();
  require(std::set<std::string>(keys.begin(), keys.end()) ==
          std::set<std::string>{"observations", "feature_mask", "labels_scoring_only", "source_ids"},
          "exact four-key legal archive");
  ev::ControlledDataset out; out.observed = {read(a, "observations"), read(a, "feature_mask")};
  out.labels = read(a, "labels_scoring_only"); const auto ids = read(a, "source_ids");
  const std::vector<int64_t> shape{rows, 3, 32, 3};
  require(out.observed.data.device().is_cpu() && out.observed.data.scalar_type() == torch::kFloat64 &&
          out.observed.data.sizes() == torch::IntArrayRef(shape) && out.observed.data.is_contiguous(), "CPU F64 observations");
  require(out.observed.feature_mask.device().is_cpu() && out.observed.feature_mask.scalar_type() == torch::kBool &&
          out.observed.feature_mask.sizes() == torch::IntArrayRef(shape) && out.observed.feature_mask.is_contiguous(), "CPU Bool mask");
  require(out.labels.device().is_cpu() && out.labels.scalar_type() == torch::kInt64 &&
          out.labels.sizes() == torch::IntArrayRef({rows}) && out.labels.is_contiguous(), "CPU Long scoring labels");
  require(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
          ids.sizes() == torch::IntArrayRef({rows}) && ids.is_contiguous(), "CPU Long packed source IDs");
  require(torch::isfinite(out.observed.data).all().item<bool>() &&
          out.observed.data.masked_select(~out.observed.feature_mask).eq(0).all().item<bool>(), "finite legal values and hidden zeros");
  const auto source = ids.accessor<int64_t, 1>(); const auto labels = out.labels.accessor<int64_t, 1>();
  int64_t previous = -1;
  for (int64_t row = 0; row < rows; row += 2) {
    const auto id = source[row]; const auto index = (uint64_t(id) & 0x7fffffffULL) >> 1;
    require(id >= 0 && source[row + 1] == id && id > previous && (uint64_t(id) >> 31) == master &&
            (id & 1) == bit && index < 192, "ascending packed master/task/source pairs"); previous = id;
    require((labels[row] == 0 && labels[row + 1] == 1) ||
            (labels[row] == 1 && labels[row + 1] == 0), "opposite randomized labels");
    require(torch::equal(out.observed.feature_mask[row], out.observed.feature_mask[row + 1]), "pair-shared masks");
    require(torch::equal(out.observed.data[row][2], out.observed.data[row + 1][2]), "label-independent channel2");
    if (bit == 0) require(torch::equal(out.observed.data[row][0], out.observed.data[row + 1][0]), "timing channel0 pair equality");
    out.source_ids.push_back(helper_source(id, task)); out.source_ids.push_back(helper_source(id, task));
  }
  // This compatibility field contains only a legal observation copy, never clean truth.
  out.clean = {out.observed.data.clone(), out.observed.feature_mask.clone()}; return out;
}
inline void check_splits(const ev::ControlledDataset &t, const ev::ControlledDataset &v,
                         const ev::ControlledDataset &d, uint64_t master, DatasetTask task) {
  const std::set<std::string> ti(t.source_ids.begin(), t.source_ids.end()), vi(v.source_ids.begin(), v.source_ids.end());
  auto all = ti; all.insert(vi.begin(), vi.end());
  require(ti.size() == 128 && vi.size() == 64 && all.size() == 192 && d.source_ids == v.source_ids &&
          torch::equal(d.labels, v.labels), "whole pair-disjoint split and unchanged deleted rows");
  for (uint64_t source = 0; source < 192; ++source)
    require(all.count(helper_source(int64_t((master << 31) | (source << 1) | uint64_t(task_bit(task))), task)) == 1,
            "complete original source universe");
  require(!d.observed.feature_mask.logical_and(~v.observed.feature_mask).any().item<bool>() &&
          torch::equal(d.observed.data.masked_select(d.observed.feature_mask),
                       v.observed.data.masked_select(d.observed.feature_mask)), "deleted legal values are original subset");
  for (int64_t row = 0; row < 128; row += 2) {
    const auto id_text = v.source_ids[row]; const auto id = std::stoull(id_text.substr(id_text.rfind('-') + 1));
    std::mt19937_64 rng(ev::stream_seed(ev::stream_seed(master, 0x74632d64656c7631ULL), id));
    const auto original = v.observed.feature_mask.accessor<bool, 4>();
    const auto deleted = d.observed.feature_mask.accessor<bool, 4>();
    for (int c = 0; c < 3; ++c) for (int time = 0; time < 32; ++time) for (int f = 0; f < 3; ++f) {
      const bool erase = double(rng() >> 11) / 9007199254740992.0 < .30;
      for (const auto paired : {row, row + 1})
        require(deleted[paired][c][time][f] == (original[paired][c][time][f] && !erase), "exact source-keyed .30 deletion");
    }
  }
}
struct Cohort { ev::ControlledDataset training, validation, deleted; };
inline std::map<std::pair<uint64_t, DatasetTask>, Cohort> load_all(const Guard &inputs) {
  std::map<std::pair<uint64_t, DatasetTask>, Cohort> out; std::set<std::string> all;
  for (const auto master : quality_masters) for (const auto task : {DatasetTask::slow_lag_sign, DatasetTask::component_balance}) {
    const auto prefix = role_prefix(master, task);
    Cohort c{load_controlled(inputs.bind(prefix + "controlled-training.pt"), 256, master, task),
             load_controlled(inputs.bind(prefix + "controlled-validation.pt"), 128, master, task),
             load_controlled(inputs.bind(prefix + "controlled-validation-deleted.pt"), 128, master, task)};
    check_splits(c.training, c.validation, c.deleted, master, task);
    for (const auto *split : {&c.training, &c.validation}) {
      const std::set<std::string> unique(split->source_ids.begin(), split->source_ids.end());
      for (const auto &id : unique) require(all.insert(id).second, "all tasks/masters/splits independent");
    }
    out.emplace(std::make_pair(master, task), std::move(c));
  }
  require(all.size() == 768 && out.size() == 4, "all twelve legal archives admitted before models"); return out;
}

inline embedding::Batch loader_engineering(const fs::path &root) {
  using Fields = std::map<std::string, torch::Tensor>; constexpr uint64_t master = 910901;
  auto fields = [](int64_t rows, DatasetTask task) {
    Fields f; auto values = torch::sin(torch::arange((rows / 2) * 288, torch::kFloat64).reshape({rows / 2, 3, 32, 3}) * .11);
    f["observations"] = values.repeat_interleave(2, 0).contiguous();
    f["feature_mask"] = torch::ones({rows, 3, 32, 3}, torch::kBool);
    std::vector<int64_t> labels, ids;
    for (int64_t row = 0; row < rows; ++row) {
      labels.push_back((row / 2) % 2 ? 1 - row % 2 : row % 2);
      const auto source = rows == 128 ? 3 * (row / 2) + 2 : row / 2 + (row / 2) / 2;
      ids.push_back(int64_t((master << 31) | (uint64_t(source) << 1) | uint64_t(task_bit(task))));
    }
    f["labels_scoring_only"] = torch::tensor(labels, torch::kInt64); f["source_ids"] = torch::tensor(ids, torch::kInt64); return f;
  };
  auto save = [&](const std::string &name, const Fields &f) {
    const auto path = root / (name + ".serialized-archive"); require(!fs::exists(path), "new artificial archive");
    torch::serialize::OutputArchive a; for (const auto &[key, value] : f) a.write(key, value, true); a.save_to(path.string()); return path;
  };
  int positives = 0, negatives = 0; ev::ControlledDataset training;
  for (const auto task : {DatasetTask::slow_lag_sign, DatasetTask::component_balance}) {
    auto tf = fields(256, task), vf = fields(128, task), df = vf;
    df["feature_mask"] = vf["feature_mask"].clone(); df["observations"] = vf["observations"].clone();
    auto mask = df["feature_mask"].accessor<bool, 4>();
    for (int64_t row = 0; row < 128; row += 2) {
      const auto id = vf["source_ids"][row].item<int64_t>();
      std::mt19937_64 rng(ev::stream_seed(ev::stream_seed(master, 0x74632d64656c7631ULL), uint64_t(id)));
      for (int c = 0; c < 3; ++c) for (int t = 0; t < 32; ++t) for (int f = 0; f < 3; ++f) {
        const bool keep = double(rng() >> 11) / 9007199254740992.0 >= .30;
        mask[row][c][t][f] = keep; mask[row + 1][c][t][f] = keep;
      }
    }
    df["observations"] = torch::where(df["feature_mask"], df["observations"], torch::zeros_like(df["observations"]));
    const auto tr = load_controlled(save(task_name(task) + "-training", tf), 256, master, task, true);
    const auto va = load_controlled(save(task_name(task) + "-validation", vf), 128, master, task, true);
    const auto de = load_controlled(save(task_name(task) + "-deleted", df), 128, master, task, true);
    check_splits(tr, va, de, master, task); positives += 3; if (task == DatasetTask::slow_lag_sign) training = tr;
  }
  for (const auto defect : {0, 1, 2, 3, 4, 5, 6}) {
    auto f = fields(128, DatasetTask::slow_lag_sign);
    if (defect == 0) f["feature_mask"] = f["feature_mask"].to(torch::kInt64);
    if (defect == 1) f["observed"] = f["observations"];
    if (defect == 2) f["labels_scoring_only"][1] = 0;
    if (defect == 3) f["source_ids"][1] = f["source_ids"][2];
    if (defect == 4) f["observations"][0][0][0][0] = std::numeric_limits<double>::quiet_NaN();
    const auto path = save("malformed-" + std::to_string(defect), f); bool rejected = false;
    try { (void)load_controlled(path, 128, defect == 5 ? 910902 : master,
            defect == 6 ? DatasetTask::component_balance : DatasetTask::slow_lag_sign, true); }
    catch (const std::exception &) { rejected = true; }
    require(rejected, "serialized malformed loader must reject"); ++negatives;
  }
  require(positives == 6 && negatives == 7, "all artificial two-task loader cases");
  return {training.observed.data.narrow(0, 0, 16).clone(), training.observed.feature_mask.narrow(0, 0, 16).clone()};
}
} // namespace multiband_screen
