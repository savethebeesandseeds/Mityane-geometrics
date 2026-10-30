// Read a model-agnostic saved-vector file and rank it with exact search.
// The file format does not name an encoder. Constructors come from geometry.h.

#include "mityane/operations/retrieval/geometry.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>

namespace retrieval = mityane::operations::retrieval::v1;

namespace {

void fail(const std::string& message) {
  std::cerr << "saved-vector inspect: " << message << "\n";
  std::exit(1);
}

std::string reason_name(retrieval::Reason reason) {
  switch (reason) {
    case retrieval::Reason::none: return "none";
    case retrieval::Reason::invalid_request: return "invalid_request";
    case retrieval::Reason::duplicate_source_id: return "duplicate_source_id";
    case retrieval::Reason::incompatible_space: return "incompatible_space";
    case retrieval::Reason::nonfinite_value: return "nonfinite_value";
    case retrieval::Reason::undefined_geometry: return "undefined_geometry";
    case retrieval::Reason::dimension_mismatch: return "dimension_mismatch";
    case retrieval::Reason::collinear_inputs: return "collinear_inputs";
    case retrieval::Reason::fit: return "fit";
    case retrieval::Reason::directness: return "directness";
    case retrieval::Reason::outside_cone: return "outside_cone";
  }
  return "unknown";
}

std::string constructor_outcome_name(retrieval::ConstructorOutcome outcome) {
  switch (outcome) {
    case retrieval::ConstructorOutcome::constructed: return "constructed";
    case retrieval::ConstructorOutcome::abstained: return "abstained";
    case retrieval::ConstructorOutcome::failed: return "failed";
  }
  return "unknown";
}

std::string skip_name(retrieval::SkipReason reason) {
  switch (reason) {
    case retrieval::SkipReason::self_match: return "self_match";
    case retrieval::SkipReason::excluded_source: return "excluded_source";
    case retrieval::SkipReason::undefined_geometry: return "undefined_geometry";
    case retrieval::SkipReason::outside_cone: return "outside_cone";
  }
  return "unknown";
}

std::string outcome_name(retrieval::Outcome outcome) {
  switch (outcome) {
    case retrieval::Outcome::ranked: return "ranked";
    case retrieval::Outcome::empty: return "empty";
    case retrieval::Outcome::abstained: return "abstained";
    case retrieval::Outcome::failed: return "failed";
  }
  return "unknown";
}

struct SavedFile {
  retrieval::SpaceBinding binding;
  std::vector<retrieval::VectorRecord> records;
  std::map<std::string, std::size_t> index;
};

SavedFile load_saved(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("cannot open " + path);
  std::string magic;
  std::getline(input, magic);
  if (magic != "mityane-saved-vectors-v1") fail(path + " has the wrong header");
  SavedFile saved;
  std::string key;
  std::size_t count = 0;
  bool saw_count = false;
  while (input >> key) {
    if (key == "vectors") break;
    if (key == "space_id") input >> saved.binding.space_id;
    else if (key == "snapshot_id") input >> saved.binding.snapshot_id;
    else if (key == "metric") {
      std::string value;
      input >> value;
      if (value == "cosine") saved.binding.metric = retrieval::Metric::cosine;
      else if (value == "l2") saved.binding.metric = retrieval::Metric::l2;
      else fail("unknown metric");
    } else if (key == "normalization") {
      std::string value;
      input >> value;
      if (value == "none") saved.binding.normalization = retrieval::Normalization::none;
      else if (value == "l2") saved.binding.normalization = retrieval::Normalization::l2;
      else fail("unknown normalization");
    } else if (key == "dimensions") input >> saved.binding.dimensions;
    else if (key == "count") {
      input >> count;
      saw_count = true;
    } else fail("unknown header field " + key);
  }
  if (!saw_count || saved.binding.dimensions == 0) fail(path + " is missing dimensions or count");
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream row(line);
    retrieval::VectorRecord record;
    if (!(row >> record.source_id)) fail("missing source id");
    record.values.resize(saved.binding.dimensions);
    for (double& value : record.values) {
      std::string token;
      if (!(row >> token)) fail("short vector " + record.source_id);
      char* end = nullptr;
      value = std::strtod(token.c_str(), &end);
      if (end == token.c_str() || *end != '\0') fail("bad value in " + record.source_id);
    }
    std::string extra;
    if (row >> extra) fail("long vector " + record.source_id);
    if (saved.index.count(record.source_id) != 0) fail("duplicate " + record.source_id);
    saved.index.emplace(record.source_id, saved.records.size());
    saved.records.push_back(std::move(record));
  }
  if (saved.records.size() != count) fail(path + " count does not match the rows");
  return saved;
}

const retrieval::VectorRecord& require(const SavedFile& saved, const std::string& source_id) {
  const auto found = saved.index.find(source_id);
  if (found == saved.index.end()) fail("missing point " + source_id);
  return saved.records[found->second];
}

void print_search(const retrieval::SearchResult& result, const std::vector<std::string>& labels) {
  std::cout << "outcome " << outcome_name(result.outcome) << " reason " << reason_name(result.reason)
            << " eligible " << result.eligible_count << "\n";
  for (const retrieval::Hit& hit : result.hits) {
    std::cout << "hit " << hit.rank << " " << hit.source_id << " distance " << hit.distance
              << " similarity " << hit.similarity << "\n";
  }
  std::map<std::string, std::size_t> skips;
  for (const retrieval::Skipped& skip : result.skipped) skips[skip_name(skip.reason)] += 1;
  std::cout << "skips";
  for (const char* name : {"self_match", "excluded_source", "undefined_geometry", "outside_cone"}) {
    std::cout << " " << name << " " << skips[name];
  }
  std::cout << "\n";
  for (const std::string& label : labels) {
    const auto found = std::find_if(result.hits.begin(), result.hits.end(),
                                     [&](const retrieval::Hit& hit) { return hit.source_id == label; });
    if (found == result.hits.end()) std::cout << "label " << label << " absent\n";
    else std::cout << "label " << label << " rank " << found->rank << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) fail("usage: saved_vector_inspect corpus.vec points.vec queries.txt");
  const SavedFile corpus_file = load_saved(argv[1]);
  const SavedFile points = load_saved(argv[2]);
  if (!(corpus_file.binding == points.binding)) fail("corpus and points use different spaces");
  retrieval::Corpus corpus;
  corpus.binding = corpus_file.binding;
  corpus.records = corpus_file.records;
  std::ifstream queries(argv[3]);
  if (!queries) fail("cannot open queries");
  std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
  std::cout.precision(8);
  std::string line;
  while (std::getline(queries, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::string command;
    row >> command;
    std::vector<std::string> ids;
    std::vector<std::string> labels;
    std::size_t k = 0;
    bool labels_now = false;
    bool theta_next = false;
    bool have_theta = false;
    double theta = 0;
    std::string theta_token;
    std::string token;
    while (row >> token) {
      if (theta_next) {
        theta_token = token;
        if (token == "0") theta = 0;
        else if (token == "pi") theta = std::numbers::pi_v<double>;
        else if (token == "pi/2") theta = std::numbers::pi_v<double> / 2;
        else if (token == "pi/3") theta = std::numbers::pi_v<double> / 3;
        else if (token == "pi/6") theta = std::numbers::pi_v<double> / 6;
        else fail("unknown aperture " + token);
        have_theta = true;
        theta_next = false;
        continue;
      }
      if (token == "theta") {
        theta_next = true;
        continue;
      }
      if (token == "label") {
        labels_now = true;
        continue;
      }
      if (!labels_now && k == 0 && token.find_first_not_of("0123456789") == std::string::npos && !ids.empty()) {
        k = static_cast<std::size_t>(std::stoull(token));
        continue;
      }
      if (labels_now) labels.push_back(token);
      else ids.push_back(token);
    }
    if (k == 0) fail("query is missing k: " + line);
    std::cout << "query " << command;
    for (const std::string& id : ids) std::cout << " " << id;
    if (have_theta) std::cout << " theta " << theta_token;
    std::cout << "\n";
    retrieval::Query probe;
    probe.binding = corpus.binding;
    probe.query_id = command;
    if (command == "point") {
      if (ids.size() != 1) fail("point needs one id");
      probe.values = require(points, ids[0]).values;
      probe.query_id = ids[0];
    } else if (command == "difference") {
      if (ids.size() != 2) fail("difference needs two ids");
      const retrieval::ConstructorResult built =
          retrieval::difference(points.binding, require(points, ids[0]), require(points, ids[1]));
      if (built.outcome != retrieval::ConstructorOutcome::constructed) {
        std::cout << "constructor " << constructor_outcome_name(built.outcome) << " "
                  << reason_name(built.reason) << "\n";
        continue;
      }
      probe.values = built.values;
    } else if (command == "shared" || command == "cone") {
      if (ids.size() < 2 || ids.size() > 8) fail(command + " needs two to eight ids");
      std::vector<retrieval::VectorRecord> group;
      for (const std::string& id : ids) group.push_back(require(points, id));
      const retrieval::ConstructorResult built = retrieval::shared_direction(points.binding, group);
      if (built.outcome != retrieval::ConstructorOutcome::constructed) {
        std::cout << "constructor " << constructor_outcome_name(built.outcome) << " "
                  << reason_name(built.reason) << "\n";
        continue;
      }
      if (command == "shared") {
        probe.values = built.values;
      } else {
        if (!have_theta) fail("cone needs theta");
        const retrieval::ConeResult cone = retrieval::make_cone(points.binding, built.values, theta);
        if (cone.outcome != retrieval::ConstructorOutcome::constructed) {
          std::cout << "constructor " << constructor_outcome_name(cone.outcome) << " "
                    << reason_name(cone.reason) << "\n";
          continue;
        }
        const retrieval::SearchResult found = retrieval::search_cone(corpus, cone.cone, k, {});
        const retrieval::ConeReport diagnostic = retrieval::cone_diagnostics(corpus, cone.cone, k, labels);
        std::cout << "cone-report outcome " << outcome_name(diagnostic.outcome) << " reason "
                  << reason_name(diagnostic.reason) << " inside " << diagnostic.inside_count << " purity ";
        if (diagnostic.purity.has_value()) std::cout << diagnostic.purity.value();
        else std::cout << "unset";
        std::cout << "\n";
        print_search(found, labels);
        continue;
      }
    } else {
      fail("unknown query " + command);
    }
    print_search(retrieval::search(corpus, probe, k, {}), labels);
  }
  return 0;
}
