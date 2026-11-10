// Print K geometric retrievals of a query. Not a registered tool.

#include "mityane/operations/retrieval/combination.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace retrieval = mityane::operations::retrieval::v1;

namespace {

const char* outcome_name(retrieval::Outcome outcome) {
  switch (outcome) {
    case retrieval::Outcome::ranked: return "ranked";
    case retrieval::Outcome::empty: return "empty";
    case retrieval::Outcome::abstained: return "abstained";
    case retrieval::Outcome::failed: return "failed";
  }
  return "unknown";
}

const char* reason_name(retrieval::Reason reason) {
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

std::vector<std::pair<std::string, std::string>> load_surfaces(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path);
  std::vector<std::pair<std::string, std::string>> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos) rows.push_back({line, ""});
    else rows.push_back({line.substr(0, tab), line.substr(tab + 1)});
  }
  return rows;
}

void load_f32(const std::string& path, std::vector<retrieval::VectorRecord>& records) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path);
  std::vector<float> row(768);
  for (retrieval::VectorRecord& record : records) {
    input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    if (!input) throw std::runtime_error("short float file " + path);
    record.values.assign(row.begin(), row.end());
  }
}

std::string gloss_of(const std::map<std::string, std::string>& gloss, const std::string& id) {
  const auto found = gloss.find(id);
  return found == gloss.end() ? id : found->second;
}

std::string member_of(const std::string& gloss) {
  const std::size_t colon = gloss.find(": ");
  return colon == std::string::npos ? gloss : gloss.substr(0, colon);
}

std::string definition_of(const std::string& gloss) {
  const std::size_t colon = gloss.find(": ");
  return colon == std::string::npos ? "" : gloss.substr(colon + 2);
}

void print_hits(const std::map<std::string, std::string>& gloss,
                const std::vector<retrieval::CombinationHit>& hits) {
  for (const retrieval::CombinationHit& hit : hits) {
    const std::string surface = gloss_of(gloss, hit.source_id);
    std::cout << "  " << hit.weight << "  " << member_of(surface) << "\n";
    const std::string definition = definition_of(surface);
    if (!definition.empty()) std::cout << "         " << definition << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: report_query corpus.f32 surfaces.tsv queries.f32 queries.tsv [tau]\n";
    return 2;
  }
  std::cout << std::setprecision(6);
  const auto surfaces = load_surfaces(argv[2]);
  const auto queries = load_surfaces(argv[4]);
  retrieval::Corpus corpus;
  corpus.binding.space_id = "local-text-768-cosine-l2-sentence-similarity";
  corpus.binding.snapshot_id = "or01-ret-abs-0.2.0-local-text";
  corpus.binding.metric = retrieval::Metric::cosine;
  corpus.binding.normalization = retrieval::Normalization::l2;
  corpus.binding.dimensions = 768;
  corpus.records.resize(surfaces.size());
  std::map<std::string, std::string> gloss;
  for (std::size_t index = 0; index < surfaces.size(); ++index) {
    corpus.records[index].source_id = surfaces[index].first;
    gloss[surfaces[index].first] = surfaces[index].second;
  }
  load_f32(argv[1], corpus.records);

  std::vector<retrieval::VectorRecord> query_records(queries.size());
  for (std::size_t index = 0; index < queries.size(); ++index) {
    query_records[index].source_id = queries[index].first;
  }
  load_f32(argv[3], query_records);

  retrieval::CombinationParameters parameters;
  parameters.minimum_weight = argc == 6 ? std::stod(argv[5]) : 0.05;
  parameters.fit_floor = 0.2;
  parameters.directness_ceiling = 3;
  parameters.early_stop_cosine = 1;
  const std::size_t reconstruction_count = 2;

  for (std::size_t index = 0; index < query_records.size(); ++index) {
    const retrieval::VectorRecord& query = query_records[index];
    std::cout << "query: " << queries[index].second << "\n";
    const retrieval::ReconstructionSet pack =
        retrieval::reconstruct_k(corpus, query.values, reconstruction_count, parameters);
    for (std::size_t reading = 0; reading < pack.reconstructions.size(); ++reading) {
      const retrieval::GeometricReconstruction& reconstruction = pack.reconstructions[reading];
      const std::string start = gloss_of(gloss, reconstruction.start_source_id);
      std::cout << "\nGeometric retrieval " << (reading + 1) << " of " << pack.reconstructions.size()
                << "\n";
      std::cout << "  anchor: " << member_of(start);
      if (reading == 0) std::cout << "  (closest)";
      else if (reading == 1) std::cout << "  (second closest)";
      std::cout << "  cosine " << reconstruction.start_similarity << "\n";
      const std::string definition = definition_of(start);
      if (!definition.empty()) std::cout << "  " << definition << "\n";
      std::cout << "  leftover " << reconstruction.leftover_ratio << "  depth " << reconstruction.depth
                << "  " << outcome_name(reconstruction.combination.outcome) << " "
                << reason_name(reconstruction.combination.reason) << "\n";
      print_hits(gloss, reconstruction.combination.hits);
    }
    std::cout << "\n";
  }
  return 0;
}
