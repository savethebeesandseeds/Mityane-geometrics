// One LEGO box is one situation. One loose part in one color is one record.
// The covered pile is left out of the state. The query names its color, part
// category, and part number, and the probe answers with a brick count.
// Spares are excluded. Minifigs are not opened. The year is unused.
#include "state_fusion/model.h"

#include <torch/cuda.h>
#include <torch/optim/adam.h>
#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDataDir = "/mityane/.temp/fusion/state-fusion-transformer/lego/rebrickable";
constexpr const char* kCheckpoint = "/mityane/.temp/fusion/state-fusion-transformer/lego-box.pt";
constexpr int kMinPiles = 8;
constexpr int kMaxPiles = 24;
constexpr int kMinThemeSets = 40;
constexpr int kThemePercent = 15;
constexpr int kBatch = 32;
constexpr int kSteps = 4000;
constexpr int kLogEvery = 800;
constexpr int64_t kSeed = 2101;
constexpr double kReloadTolerance = 1e-4;

struct RawLine {
  int part = 0;
  int color = 0;
  int quantity = 0;
};

struct Box {
  std::string set_num;
  int root = 0;
  std::vector<RawLine> lines;
};

struct Theme {
  std::string name;
  int parent = -1;
};

struct Vocab {
  std::unordered_map<int, int64_t> color;
  std::unordered_map<int, int64_t> category;
  std::unordered_map<int, int64_t> part;
  int64_t color_card = 1;
  int64_t category_card = 1;
  int64_t part_card = 1;
};

struct Scaler {
  double mean = 0;
  double std = 1;
};

struct Example {
  size_t box = 0;
  int hidden = 0;
};

struct Score {
  double model = 0;
  double zero = 0;
  double box_mean = 0;
  double catalog = 0;
  int64_t lines = 0;
  int64_t known_pairs = 0;
};

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::string current;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          current.push_back('"');
          ++i;
        } else {
          quoted = false;
        }
      } else {
        current.push_back(c);
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.push_back(std::move(current));
      current.clear();
    } else if (c != '\r') {
      current.push_back(c);
    }
  }
  fields.push_back(std::move(current));
  return fields;
}

int parse_int(const std::string& text, const char* what) {
  try {
    size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size()) {
      throw std::runtime_error("trailing");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("bad integer for ") + what + ": " + text);
  }
}

bool spare_flag(std::string text) {
  for (char& c : text) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  if (text == "1" || text == "t" || text == "true") {
    return true;
  }
  if (text == "0" || text == "f" || text == "false" || text.empty()) {
    return false;
  }
  throw std::runtime_error("bad spare flag: " + text);
}

std::vector<std::vector<std::string>> read_table(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open " + path.string());
  }
  std::vector<std::vector<std::string>> rows;
  std::string line;
  bool header = true;
  while (std::getline(input, line)) {
    if (header) {
      header = false;
      continue;
    }
    if (line.empty() || line == "\r") {
      continue;
    }
    rows.push_back(split_csv(line));
  }
  return rows;
}

int root_theme(int theme, const std::unordered_map<int, Theme>& themes) {
  int hops = 0;
  while (hops < 16) {
    const auto found = themes.find(theme);
    if (found == themes.end() || found->second.parent < 0) {
      return theme;
    }
    theme = found->second.parent;
    ++hops;
  }
  return theme;
}

std::string theme_name(int theme, const std::unordered_map<int, Theme>& themes) {
  const auto found = themes.find(theme);
  if (found == themes.end() || found->second.name.empty()) {
    return "theme-" + std::to_string(theme);
  }
  return found->second.name;
}

uint32_t fnv1a(const std::string& text) {
  uint32_t hash = 2166136261u;
  for (unsigned char c : text) {
    hash ^= c;
    hash *= 16777619u;
  }
  return hash;
}

uint64_t pair_key(int part, int color) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(part)) << 32) |
         static_cast<uint32_t>(color);
}

double log_quantity(int quantity) { return std::log1p(static_cast<double>(quantity)); }

double brick_count(double scaled, const Scaler& scaler) {
  const double raw = std::expm1(scaled * scaler.std + scaler.mean);
  return raw > 0.0 ? raw : 0.0;
}

int64_t lookup_id(const std::unordered_map<int, int64_t>& vocab, int raw) {
  const auto found = vocab.find(raw);
  return found == vocab.end() ? 0 : found->second;
}

void add_id(std::unordered_map<int, int64_t>& vocab, int64_t& card, int raw) {
  if (vocab.find(raw) == vocab.end()) {
    vocab.emplace(raw, card);
    ++card;
  }
}

std::vector<Box> load_boxes(const std::unordered_map<int, Theme>& themes) {
  const auto set_rows = read_table(std::filesystem::path(kDataDir) / "sets.csv");
  std::unordered_map<std::string, int> set_theme;
  set_theme.reserve(set_rows.size());
  for (const auto& row : set_rows) {
    if (row.size() < 4) {
      throw std::runtime_error("sets.csv has a short row");
    }
    set_theme.emplace(row[0], parse_int(row[3], "theme_id"));
  }

  const auto inventory_rows = read_table(std::filesystem::path(kDataDir) / "inventories.csv");
  struct Choice {
    int version = -1;
    int id = -1;
  };
  std::unordered_map<std::string, Choice> choice;
  for (const auto& row : inventory_rows) {
    if (row.size() < 3) {
      throw std::runtime_error("inventories.csv has a short row");
    }
    const int id = parse_int(row[0], "inventory id");
    const int version = parse_int(row[1], "inventory version");
    const auto& set_num = row[2];
    if (set_theme.find(set_num) == set_theme.end()) {
      continue;
    }
    auto& slot = choice[set_num];
    if (version > slot.version || (version == slot.version && id > slot.id)) {
      slot = Choice{version, id};
    }
  }
  std::unordered_map<int, std::string> inventory_set;
  inventory_set.reserve(choice.size());
  for (const auto& item : choice) {
    inventory_set.emplace(item.second.id, item.first);
  }

  const auto part_rows = read_table(std::filesystem::path(kDataDir) / "parts.csv");
  std::unordered_map<std::string, int> part_index;
  part_index.reserve(part_rows.size());
  std::vector<int> part_category;
  for (const auto& row : part_rows) {
    if (row.size() < 3 || row[0].empty()) {
      continue;
    }
    const int index = static_cast<int>(part_category.size());
    part_index.emplace(row[0], index);
    part_category.push_back(parse_int(row[2], "part_cat_id"));
  }

  std::unordered_map<int, Box> boxes;
  {
    std::ifstream input(std::filesystem::path(kDataDir) / "inventory_parts.csv");
    if (!input) {
      throw std::runtime_error("cannot open inventory_parts.csv");
    }
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
      if (line.empty() || line == "\r") {
        continue;
      }
      const auto row = split_csv(line);
      if (row.size() < 5) {
        throw std::runtime_error("inventory_parts.csv has a short row");
      }
      if (spare_flag(row[4])) {
        continue;
      }
      const int inventory = parse_int(row[0], "inventory_id");
      const auto set = inventory_set.find(inventory);
      if (set == inventory_set.end()) {
        continue;
      }
      const auto part = part_index.find(row[1]);
      if (part == part_index.end()) {
        continue;
      }
      const int quantity = parse_int(row[3], "quantity");
      if (quantity <= 0) {
        continue;
      }
      auto& box = boxes[inventory];
      if (box.set_num.empty()) {
        box.set_num = set->second;
        const auto theme = set_theme.find(box.set_num);
        box.root = root_theme(theme == set_theme.end() ? -1 : theme->second, themes);
      }
      box.lines.push_back(RawLine{part->second, parse_int(row[2], "color_id"), quantity});
    }
  }

  std::vector<Box> kept;
  kept.reserve(boxes.size());
  for (auto& item : boxes) {
    auto& lines = item.second.lines;
    std::sort(lines.begin(), lines.end(), [](const RawLine& left, const RawLine& right) {
      if (left.part != right.part) {
        return left.part < right.part;
      }
      return left.color < right.color;
    });
    std::vector<RawLine> merged;
    for (const auto& line : lines) {
      if (!merged.empty() && merged.back().part == line.part && merged.back().color == line.color) {
        merged.back().quantity += line.quantity;
      } else {
        merged.push_back(line);
      }
    }
    if (static_cast<int>(merged.size()) < kMinPiles || static_cast<int>(merged.size()) > kMaxPiles) {
      continue;
    }
    item.second.lines = std::move(merged);
    kept.push_back(std::move(item.second));
  }
  std::sort(kept.begin(), kept.end(),
            [](const Box& left, const Box& right) { return left.set_num < right.set_num; });
  return kept;
}

std::unordered_map<int, Theme> load_themes() {
  std::unordered_map<int, Theme> themes;
  for (const auto& row : read_table(std::filesystem::path(kDataDir) / "themes.csv")) {
    if (row.size() < 2) {
      throw std::runtime_error("themes.csv has a short row");
    }
    Theme theme;
    theme.name = row[1];
    if (row.size() >= 3 && !row[2].empty()) {
      theme.parent = parse_int(row[2], "parent_id");
    }
    themes.emplace(parse_int(row[0], "theme id"), std::move(theme));
  }
  return themes;
}

int choose_theme(const std::vector<Box>& boxes, const std::unordered_map<int, Theme>& themes) {
  std::unordered_map<int, int> counts;
  for (const auto& box : boxes) {
    ++counts[box.root];
  }
  struct Candidate {
    int count = 0;
    std::string name;
    int root = 0;
  };
  std::vector<Candidate> eligible;
  const int total = static_cast<int>(boxes.size());
  for (const auto& item : counts) {
    if (item.second < kMinThemeSets) {
      continue;
    }
    eligible.push_back(Candidate{item.second, theme_name(item.first, themes), item.first});
  }
  std::sort(eligible.begin(), eligible.end(), [](const Candidate& left, const Candidate& right) {
    if (left.count != right.count) {
      return left.count > right.count;
    }
    return left.name < right.name;
  });
  std::cout << "qualifying_boxes=" << total << " themes_at_least_" << kMinThemeSets << "=" << eligible.size()
            << "\n";
  for (const auto& item : eligible) {
    std::cout << "theme " << item.name << " sets=" << item.count << "\n";
  }
  std::cout << std::flush;
  Candidate best;
  bool found = false;
  for (const auto& item : eligible) {
    if (item.count * 100 > total * kThemePercent) {
      continue;
    }
    if (!found || item.count > best.count || (item.count == best.count && item.name < best.name)) {
      best = item;
      found = true;
    }
  }
  if (!found && !eligible.empty()) {
    best = *std::min_element(eligible.begin(), eligible.end(),
                             [](const Candidate& left, const Candidate& right) {
                               if (left.count != right.count) {
                                 return left.count < right.count;
                               }
                               return left.name < right.name;
                             });
    found = true;
  }
  if (!found) {
    throw std::runtime_error("no theme has enough qualifying boxes to hold out");
  }
  std::cout << "holdout_theme=" << best.name << " holdout_sets=" << best.count << "\n" << std::flush;
  return best.root;
}

struct Split {
  std::vector<Box> train;
  std::vector<Box> seen;
  std::vector<Box> theme;
  int holdout_root = 0;
};

Split split_boxes(std::vector<Box> boxes, int holdout_root) {
  Split split;
  split.holdout_root = holdout_root;
  for (auto& box : boxes) {
    if (box.root == holdout_root) {
      split.theme.push_back(std::move(box));
    } else if (fnv1a(box.set_num) % 5 == 0) {
      split.seen.push_back(std::move(box));
    } else {
      split.train.push_back(std::move(box));
    }
  }
  return split;
}

Vocab fit_vocab(const std::vector<Box>& train, const std::vector<int>& part_category) {
  Vocab vocab;
  for (const auto& box : train) {
    for (const auto& line : box.lines) {
      add_id(vocab.color, vocab.color_card, line.color);
      add_id(vocab.category, vocab.category_card, part_category[static_cast<size_t>(line.part)]);
      add_id(vocab.part, vocab.part_card, line.part);
    }
  }
  return vocab;
}

Scaler fit_scaler(const std::vector<Box>& train) {
  double sum = 0;
  double square = 0;
  int64_t count = 0;
  for (const auto& box : train) {
    for (const auto& line : box.lines) {
      const double value = log_quantity(line.quantity);
      sum += value;
      square += value * value;
      ++count;
    }
  }
  if (count < 2) {
    throw std::runtime_error("the training boxes do not have enough piles to scale");
  }
  Scaler scaler;
  scaler.mean = sum / static_cast<double>(count);
  const double variance = square / static_cast<double>(count) - scaler.mean * scaler.mean;
  scaler.std = std::sqrt(std::max(variance, 1e-12));
  return scaler;
}

struct Catalog {
  std::unordered_map<uint64_t, std::pair<double, int64_t>> pairs;
  std::unordered_map<int, std::pair<double, int64_t>> parts;
  double global = 0;
};

void add_mean(std::pair<double, int64_t>& slot, int quantity) {
  slot.first += quantity;
  ++slot.second;
}

Catalog fit_catalog(const std::vector<Box>& train) {
  Catalog catalog;
  double sum = 0;
  int64_t count = 0;
  for (const auto& box : train) {
    for (const auto& line : box.lines) {
      add_mean(catalog.pairs[pair_key(line.part, line.color)], line.quantity);
      add_mean(catalog.parts[line.part], line.quantity);
      sum += line.quantity;
      ++count;
    }
  }
  catalog.global = count == 0 ? 0 : sum / static_cast<double>(count);
  return catalog;
}

double catalog_guess(const Catalog& catalog, const RawLine& line) {
  const auto pair = catalog.pairs.find(pair_key(line.part, line.color));
  if (pair != catalog.pairs.end() && pair->second.second > 0) {
    return pair->second.first / static_cast<double>(pair->second.second);
  }
  const auto part = catalog.parts.find(line.part);
  if (part != catalog.parts.end() && part->second.second > 0) {
    return part->second.first / static_cast<double>(part->second.second);
  }
  return catalog.global;
}

bool pair_known(const Catalog& catalog, const RawLine& line) {
  const auto pair = catalog.pairs.find(pair_key(line.part, line.color));
  return pair != catalog.pairs.end() && pair->second.second > 0;
}

state_fusion::SetBatch make_batch(const std::vector<Box>& boxes, const std::vector<Example>& examples,
                                  const Vocab& vocab, const std::vector<int>& part_category,
                                  const Scaler& scaler, bool query, torch::Device device) {
  const int64_t rows = static_cast<int64_t>(examples.size());
  int64_t width = query ? 1 : 1;
  if (!query) {
    for (const auto& example : examples) {
      width = std::max(width, static_cast<int64_t>(boxes[example.box].lines.size()) - 1);
    }
  }
  const int64_t fields = 5;
  const int64_t semantic_width = 16;
  std::vector<float> scalars(static_cast<size_t>(rows * width), 0.f);
  std::vector<uint8_t> scalar_observed(static_cast<size_t>(rows * width), 0);
  std::vector<int64_t> categoricals(static_cast<size_t>(rows * width * fields), 0);
  std::vector<uint8_t> categorical_observed(static_cast<size_t>(rows * width * fields), 0);
  std::vector<float> semantic(static_cast<size_t>(rows * width * semantic_width), 0.f);
  std::vector<uint8_t> semantic_observed(static_cast<size_t>(rows * width), 0);
  std::vector<uint8_t> record_observed(static_cast<size_t>(rows * width), 0);

  for (int64_t row = 0; row < rows; ++row) {
    const auto& example = examples[static_cast<size_t>(row)];
    const auto& box = boxes[example.box];
    if (query) {
      const auto& line = box.lines[static_cast<size_t>(example.hidden)];
      const size_t base = static_cast<size_t>(row) * static_cast<size_t>(fields);
      categoricals[base] = lookup_id(vocab.color, line.color);
      categoricals[base + 1] = lookup_id(vocab.category, part_category[static_cast<size_t>(line.part)]);
      categoricals[base + 2] = lookup_id(vocab.part, line.part);
      categorical_observed[base] = 1;
      categorical_observed[base + 1] = 1;
      categorical_observed[base + 2] = 1;
      record_observed[static_cast<size_t>(row)] = 1;
      continue;
    }
    int64_t slot = 0;
    for (int index = 0; index < static_cast<int>(box.lines.size()); ++index) {
      if (index == example.hidden) {
        continue;
      }
      const auto& line = box.lines[static_cast<size_t>(index)];
      const size_t at = static_cast<size_t>(row * width + slot);
      scalars[at] = static_cast<float>((log_quantity(line.quantity) - scaler.mean) / scaler.std);
      scalar_observed[at] = 1;
      const size_t category_at = at * static_cast<size_t>(fields);
      categoricals[category_at] = lookup_id(vocab.color, line.color);
      categoricals[category_at + 1] = lookup_id(vocab.category, part_category[static_cast<size_t>(line.part)]);
      categoricals[category_at + 2] = lookup_id(vocab.part, line.part);
      categorical_observed[category_at] = 1;
      categorical_observed[category_at + 1] = 1;
      categorical_observed[category_at + 2] = 1;
      record_observed[at] = 1;
      ++slot;
    }
  }

  auto upload_float = [&](std::vector<float>& values, std::initializer_list<int64_t> shape) {
    return torch::from_blob(values.data(), shape, torch::kFloat).clone().to(device);
  };
  auto upload_bool = [&](std::vector<uint8_t>& values, std::initializer_list<int64_t> shape) {
    return torch::from_blob(values.data(), shape, torch::kUInt8).clone().to(torch::kBool).to(device);
  };
  auto upload_long = [&](std::vector<int64_t>& values, std::initializer_list<int64_t> shape) {
    return torch::from_blob(values.data(), shape, torch::kLong).clone().to(device);
  };

  state_fusion::SetBatch batch;
  batch.scalars = upload_float(scalars, {rows, width, 1});
  batch.scalar_observed = upload_bool(scalar_observed, {rows, width, 1});
  batch.categoricals = upload_long(categoricals, {rows, width, fields});
  batch.categorical_observed = upload_bool(categorical_observed, {rows, width, fields});
  batch.semantic = upload_float(semantic, {rows, width, semantic_width});
  batch.semantic_observed = upload_bool(semantic_observed, {rows, width});
  batch.record_observed = upload_bool(record_observed, {rows, width});
  batch.scalars.set_requires_grad(false);
  return batch;
}

torch::Tensor example_targets(const std::vector<Box>& boxes, const std::vector<Example>& examples,
                              const Scaler& scaler, torch::Device device) {
  std::vector<float> values(examples.size(), 0.f);
  for (size_t i = 0; i < examples.size(); ++i) {
    const auto& line = boxes[examples[i].box].lines[static_cast<size_t>(examples[i].hidden)];
    values[i] = static_cast<float>((log_quantity(line.quantity) - scaler.mean) / scaler.std);
  }
  return torch::from_blob(values.data(), {static_cast<int64_t>(values.size()), 1, 1}, torch::kFloat)
      .clone()
      .to(device);
}

Score evaluate(state_fusion::StateFusionModel& model, const std::vector<Box>& boxes, const Vocab& vocab,
               const std::vector<int>& part_category, const Scaler& scaler, const Catalog& catalog,
               torch::Device device) {
  torch::NoGradGuard guard;
  model->eval();
  Score score;
  std::vector<Example> batch;
  auto flush = [&]() {
    if (batch.empty()) {
      return;
    }
    const auto visible = make_batch(boxes, batch, vocab, part_category, scaler, false, device);
    const auto queries = make_batch(boxes, batch, vocab, part_category, scaler, true, device);
    const auto predicted = model->query_scalars(model->encode(visible), queries).detach().cpu();
    for (size_t i = 0; i < batch.size(); ++i) {
      const auto& box = boxes[batch[i].box];
      const auto& hidden = box.lines[static_cast<size_t>(batch[i].hidden)];
      const double guess = brick_count(predicted[static_cast<int64_t>(i)][0][0].item<double>(), scaler);
      double others = 0;
      int other_count = 0;
      for (int line = 0; line < static_cast<int>(box.lines.size()); ++line) {
        if (line == batch[i].hidden) {
          continue;
        }
        others += box.lines[static_cast<size_t>(line)].quantity;
        ++other_count;
      }
      const double box_mean = others / static_cast<double>(other_count);
      const double actual = hidden.quantity;
      score.model += std::abs(guess - actual);
      score.zero += std::abs(actual);
      score.box_mean += std::abs(box_mean - actual);
      score.catalog += std::abs(catalog_guess(catalog, hidden) - actual);
      ++score.lines;
      if (pair_known(catalog, hidden)) {
        ++score.known_pairs;
      }
    }
    batch.clear();
  };
  for (size_t box = 0; box < boxes.size(); ++box) {
    for (int line = 0; line < static_cast<int>(boxes[box].lines.size()); ++line) {
      batch.push_back(Example{box, line});
      if (batch.size() == 64) {
        flush();
      }
    }
  }
  flush();
  if (score.lines == 0) {
    throw std::runtime_error("an evaluation split has no piles");
  }
  score.model /= static_cast<double>(score.lines);
  score.zero /= static_cast<double>(score.lines);
  score.box_mean /= static_cast<double>(score.lines);
  score.catalog /= static_cast<double>(score.lines);
  return score;
}

void print_score(const std::string& label, const Score& score) {
  std::cout << label << " lines=" << score.lines << " known_part_color=" << score.known_pairs
            << " model_mae=" << score.model << " zero_mae=" << score.zero
            << " box_mean_mae=" << score.box_mean << " catalog_mae=" << score.catalog << "\n"
            << std::flush;
}

std::vector<int> load_part_categories_of_parts() {
  std::vector<int> categories;
  for (const auto& row : read_table(std::filesystem::path(kDataDir) / "parts.csv")) {
    if (row.size() < 3 || row[0].empty()) {
      continue;
    }
    categories.push_back(parse_int(row[2], "part_cat_id"));
  }
  return categories;
}

}  // namespace

int main() {
  try {
    if (!torch::cuda::is_available()) {
      std::cerr << "CUDA is required for the LEGO box study.\n";
      return 1;
    }
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);

    const auto themes = load_themes();
    const auto part_category = load_part_categories_of_parts();
    auto boxes = load_boxes(themes);
    const int holdout_root = choose_theme(boxes, themes);
    const auto split = split_boxes(std::move(boxes), holdout_root);
    const auto vocab = fit_vocab(split.train, part_category);
    const auto scaler = fit_scaler(split.train);
    const auto catalog = fit_catalog(split.train);
    std::cout << "train_boxes=" << split.train.size() << " seen_theme_boxes=" << split.seen.size()
              << " left_out_theme_boxes=" << split.theme.size() << " colors=" << vocab.color_card
              << " categories=" << vocab.category_card << " parts=" << vocab.part_card
              << " log_quantity_mean=" << scaler.mean << " log_quantity_std=" << scaler.std << "\n";
    std::cout << "note=lego_box_count_not_a_forecast\n" << std::flush;

    torch::manual_seed(static_cast<uint64_t>(kSeed));
    torch::cuda::manual_seed_all(kSeed);
    const auto device = torch::Device(torch::kCUDA, 0);
    state_fusion::Config config;
    config.scalar_features = 1;
    config.semantic_width = 16;
    config.categorical_cardinalities = {vocab.color_card, vocab.category_card, vocab.part_card, 1, 1};
    state_fusion::StateFusionModel model(config);
    model->to(device);

    const Score untrained_theme =
        evaluate(model, split.theme, vocab, part_category, scaler, catalog, device);
    const Score untrained_seen = evaluate(model, split.seen, vocab, part_category, scaler, catalog, device);
    print_score("untrained_left_out_theme", untrained_theme);
    print_score("untrained_seen_theme", untrained_seen);

    std::mt19937_64 rng(static_cast<uint64_t>(kSeed));
    std::uniform_int_distribution<size_t> box_pick(0, split.train.size() - 1);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    for (int step = 1; step <= kSteps; ++step) {
      std::vector<Example> batch;
      batch.reserve(kBatch);
      for (int i = 0; i < kBatch; ++i) {
        const size_t box = box_pick(rng);
        std::uniform_int_distribution<int> line_pick(0, static_cast<int>(split.train[box].lines.size()) - 1);
        batch.push_back(Example{box, line_pick(rng)});
      }
      const auto visible = make_batch(split.train, batch, vocab, part_category, scaler, false, device);
      const auto queries = make_batch(split.train, batch, vocab, part_category, scaler, true, device);
      const auto targets = example_targets(split.train, batch, scaler, device);
      optimizer.zero_grad();
      const auto state = model->encode(visible);
      const auto predicted = model->query_scalars(state, queries);
      const auto loss = (predicted - targets).square().mean();
      loss.backward();
      if (step == 1) {
        double query_grad = 0;
        double head_grad = 0;
        for (const auto& item : model->named_parameters()) {
          if (!item.value().grad().defined()) {
            continue;
          }
          const double amount = item.value().grad().abs().sum().item<double>();
          if (item.key().rfind("query.", 0) == 0) {
            query_grad += amount;
          }
          if (item.key().rfind("query_scalar_head", 0) == 0) {
            head_grad += amount;
          }
        }
        std::cout << "query_block_grad=" << query_grad << " query_scalar_head_grad=" << head_grad << "\n"
                  << std::flush;
        if (!(query_grad > 0.0) || !(head_grad > 0.0)) {
          throw std::runtime_error("the query block received no gradient");
        }
      }
      optimizer.step();
      if (!torch::isfinite(loss).item<bool>()) {
        std::cerr << "LEGO loss is not finite at step " << step << "\n";
        return 1;
      }
      if (step % kLogEvery == 0) {
        print_score("step=" + std::to_string(step) + " left_out_theme",
                    evaluate(model, split.theme, vocab, part_category, scaler, catalog, device));
        model->train();
      }
    }

    const Score theme_score = evaluate(model, split.theme, vocab, part_category, scaler, catalog, device);
    const Score seen_score = evaluate(model, split.seen, vocab, part_category, scaler, catalog, device);
    print_score("final_left_out_theme", theme_score);
    print_score("final_seen_theme", seen_score);

    const std::filesystem::path checkpoint(kCheckpoint);
    std::filesystem::create_directories(checkpoint.parent_path());
    state_fusion::save_checkpoint(model, checkpoint.string());
    auto restored = state_fusion::load_checkpoint(checkpoint.string(), device);
    restored->eval();
    model->eval();
    double reload_gap = 0;
    {
      torch::NoGradGuard guard;
      std::vector<Example> sample;
      for (size_t box = 0; box < split.theme.size() && sample.size() < 32; ++box) {
        sample.push_back(Example{box, 0});
      }
      const auto visible = make_batch(split.theme, sample, vocab, part_category, scaler, false, device);
      const auto queries = make_batch(split.theme, sample, vocab, part_category, scaler, true, device);
      const auto original = model->query_scalars(model->encode(visible), queries);
      const auto loaded = restored->query_scalars(restored->encode(visible), queries);
      reload_gap = (original - loaded).abs().max().item<double>();
    }
    std::cout << "checkpoint=" << checkpoint.string() << " checkpoint_query_scalars_max_abs=" << reload_gap
              << "\n" << std::flush;

    const bool beats_zero = theme_score.model < theme_score.zero;
    const bool beats_box = theme_score.model < theme_score.box_mean;
    const bool beats_catalog = theme_score.model < theme_score.catalog;
    const bool reload_ok = std::isfinite(reload_gap) && reload_gap <= kReloadTolerance;
    if (!beats_zero || !beats_box || !beats_catalog || !reload_ok) {
      std::cerr << "lego gates failed zero=" << beats_zero << " box_mean=" << beats_box
                << " catalog=" << beats_catalog << " reload=" << reload_ok << "\n";
      return 1;
    }
    std::cout << "state_fusion_lego: pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
