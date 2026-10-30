// SPDX-License-Identifier: MIT
// Private netstring bridge for OR01-RET. Reads saved vectors and calls exact search.
// It does not parse JSON, call a provider, or judge relevance.
#include "mityane/operations/retrieval/exact_search.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace retrieval = mityane::operations::retrieval::v1;
namespace {
constexpr std::size_t maximum_request = 64U * 1024U * 1024U;
constexpr std::size_t maximum_token = 65536U;
constexpr std::int64_t safe_integer = 9007199254740991LL;

[[noreturn]] void malformed(const char* detail) { throw std::runtime_error(detail); }

bool valid_utf8(std::string_view value) {
  for (std::size_t index = 0; index < value.size();) {
    const auto lead = static_cast<unsigned char>(value[index++]);
    if (lead < 0x80) { if (lead == 0) return false; continue; }
    std::uint32_t code = 0, minimum = 0;
    std::size_t remaining = 0;
    if (lead >= 0xc2 && lead <= 0xdf) { code = lead & 0x1fU; minimum = 0x80; remaining = 1; }
    else if (lead >= 0xe0 && lead <= 0xef) { code = lead & 0x0fU; minimum = 0x800; remaining = 2; }
    else if (lead >= 0xf0 && lead <= 0xf4) { code = lead & 0x07U; minimum = 0x10000; remaining = 3; }
    else return false;
    if (remaining > value.size() - index) return false;
    while (remaining-- != 0) {
      const auto next = static_cast<unsigned char>(value[index++]);
      if ((next & 0xc0U) != 0x80U) return false;
      code = (code << 6U) | (next & 0x3fU);
    }
    if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
  }
  return true;
}

class Reader {
 public:
  explicit Reader(std::string bytes) : bytes_(std::move(bytes)) {}
  std::string_view token() {
    const auto colon = bytes_.find(':', position_);
    if (colon == std::string::npos || colon == position_ || colon - position_ > 6)
      malformed("Invalid private netstring length");
    const std::string_view digits(bytes_.data() + position_, colon - position_);
    if (digits.size() > 1 && digits.front() == '0') malformed("Noncanonical netstring length");
    std::size_t count = 0;
    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || count > maximum_token)
      malformed("Invalid or oversized private token");
    const auto start = colon + 1;
    if (count >= bytes_.size() - start || bytes_[start + count] != ',') malformed("Truncated private token");
    position_ = start + count + 1;
    return std::string_view(bytes_.data() + start, count);
  }
  std::string text() {
    const auto value = token();
    if (!valid_utf8(value)) malformed("Private string must be valid UTF-8 without NUL");
    return std::string(value);
  }
  std::string hash() {
    const auto value = text();
    if (value.size() != 64) malformed("Private sha256 must be 64 hex characters");
    for (const char code : value)
      if (!((code >= '0' && code <= '9') || (code >= 'a' && code <= 'f')))
        malformed("Private sha256 must be lowercase hex");
    return value;
  }
  std::int64_t integer() {
    const auto value = token();
    std::int64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        result < -safe_integer || result > safe_integer) malformed("Invalid safe integer");
    return result;
  }
  std::size_t size(std::size_t maximum) {
    const auto value = integer();
    if (value < 0 || static_cast<std::uint64_t>(value) > maximum) malformed("Invalid size value");
    return static_cast<std::size_t>(value);
  }
  double number() {
    const auto value = token();
    double result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result,
                                        std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !std::isfinite(result))
      malformed("Invalid finite binary64 number");
    return result;
  }
  bool boolean() {
    const auto value = token();
    if (value == "true") return true;
    if (value == "false") return false;
    malformed("Invalid boolean");
  }
  void null_token() { if (token() != "null") malformed("Expected null relevance rule"); }
  template <typename Function> auto optional(Function function) -> std::optional<decltype(function())> {
    const auto present = token();
    if (present == "0") return std::nullopt;
    if (present != "1") malformed("Invalid optional marker");
    return function();
  }
  void finish() const { if (position_ != bytes_.size()) malformed("Trailing private request bytes"); }

 private:
  std::string bytes_;
  std::size_t position_ = 0;
};

struct Artifact { std::string id, sha256; };
struct VectorPoint {
  std::string source_id;
  Artifact parent;
  std::string representation_kind;
  std::vector<double> values;
  std::optional<std::int64_t> available_at;
  std::optional<std::int64_t> window_start;
  std::optional<std::int64_t> window_end;
};
struct Request {
  std::string operation_run_id, claim_level, profile, contract_id, contract_revision, rer_id, rer_revision;
  bool live_call = false;
  std::string code_sha256, configuration_sha256, temporal_mode, non_temporal_justification;
  std::optional<std::int64_t> as_of;
  std::string space_id, snapshot_id, provider_id, metric, normalization, task_role, representation_kind;
  std::string fitted_transform_id;
  std::optional<std::int64_t> fitted_available_at;
  std::size_t dimensions = 0;
  VectorPoint query;
  std::string query_id;
  std::vector<VectorPoint> corpus;
  std::size_t k = 0;
  bool exclude_self = true;
  std::vector<std::string> excluded_source_ids;
  std::string overlap_policy;
};

Artifact artifact(Reader& reader) { return {reader.text(), reader.hash()}; }
std::vector<double> values(Reader& reader, std::size_t maximum) {
  const auto count = reader.size(maximum);
  std::vector<double> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) result.push_back(reader.number());
  return result;
}
VectorPoint point(Reader& reader, bool with_query_id, std::string& query_id) {
  if (with_query_id) query_id = reader.text();
  VectorPoint value;
  value.source_id = reader.text();
  value.parent = artifact(reader);
  value.representation_kind = reader.text();
  value.values = values(reader, 8192);
  value.available_at = reader.optional([&] { return reader.integer(); });
  value.window_start = reader.optional([&] { return reader.integer(); });
  value.window_end = reader.optional([&] { return reader.integer(); });
  return value;
}

Request request(Reader& reader) {
  if (reader.text() != "mityane-retrieval-request-v1") malformed("Unexpected private request version");
  Request value;
  reader.text();  // request_id
  value.operation_run_id = reader.text();
  value.contract_id = reader.text();
  value.contract_revision = reader.text();
  value.rer_id = reader.text();
  value.rer_revision = reader.text();
  value.claim_level = reader.text();
  value.profile = reader.text();
  value.live_call = reader.boolean();
  reader.text();  // purpose
  value.code_sha256 = reader.hash();
  value.configuration_sha256 = reader.hash();
  reader.text();  // evidence_destination
  value.temporal_mode = reader.text();
  value.as_of = reader.optional([&] { return reader.integer(); });
  value.non_temporal_justification = reader.text();
  const Artifact space = artifact(reader);
  artifact(reader);  // provider artifact
  artifact(reader);  // revision artifact
  value.fitted_transform_id = artifact(reader).id;
  value.space_id = reader.text();
  value.snapshot_id = reader.text();
  value.provider_id = reader.text();
  reader.text();  // revision_id
  value.dimensions = reader.size(8192);
  value.normalization = reader.text();
  value.metric = reader.text();
  value.task_role = reader.text();
  value.representation_kind = reader.text();
  value.fitted_available_at = reader.optional([&] { return reader.integer(); });
  if (space.id.empty() || value.space_id.empty()) malformed("Missing space identity");
  value.query = point(reader, true, value.query_id);
  const auto count = reader.size(4096);
  value.corpus.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::string ignored;
    value.corpus.push_back(point(reader, false, ignored));
  }
  value.k = reader.size(4096);
  value.exclude_self = reader.boolean();
  const auto excluded = reader.size(4096);
  value.excluded_source_ids.reserve(excluded);
  for (std::size_t index = 0; index < excluded; ++index) value.excluded_source_ids.push_back(reader.text());
  value.overlap_policy = reader.text();
  reader.null_token();
  reader.finish();
  return value;
}

struct Diagnostic { std::string code, record_id, message; };
struct HitRow {
  std::string source_id, parent_id, representation_kind;
  std::size_t rank = 0;
  double distance = 0, similarity = 0;
};
struct SkipRow { std::string source_id, parent_id, reason; };
struct Outcome {
  bool admitted = false;
  std::vector<Diagnostic> diagnostics;
  std::string outcome = "rejected";
  std::vector<HitRow> hits;
  std::vector<SkipRow> skipped;
  std::size_t eligible_count = 0;
  std::size_t requested = 1, succeeded = 0, abstained = 0, failed = 0, unresolved = 0;
};

bool blank(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") == std::string::npos;
}
bool window_ok(const VectorPoint& point) {
  if (point.window_start && point.window_end) return *point.window_start <= *point.window_end;
  return !point.window_start && !point.window_end;
}
bool overlaps(const VectorPoint& query, const VectorPoint& member) {
  if (!query.window_start || !member.window_start) return false;
  return *query.window_start <= *member.window_end && *member.window_start <= *query.window_end;
}
void refuse(Outcome& outcome, const std::string& code, const std::string& id, const std::string& message) {
  outcome.admitted = false;
  outcome.outcome = "rejected";
  outcome.diagnostics.push_back({code, id, message});
}

Outcome evaluate(const Request& input) {
  Outcome outcome;
  const auto reject = [&](const std::string& code, const std::string& id, const std::string& message) {
    refuse(outcome, code, id, message);
    return outcome;
  };
  if (input.claim_level == "task_association")
    return reject("RELEVANCE_UNBOUND", input.operation_run_id,
                  "Task association needs a bound relevance rule. This tool records exploratory ranking only.");
  if (input.claim_level != "exploratory_description")
    return reject("UNSUPPORTED_CLAIM", input.operation_run_id,
                  "OR01-RET 0.1.0 engineering calls accept exploratory_description only.");
  if (input.live_call)
    return reject("LIVE_CALL_REFUSED", input.profile,
                  "A live provider call stays with Hx18. This tool ranks saved vectors.");
  if (input.contract_id != "OR01-RET" || input.contract_revision != "0.1.0" ||
      input.rer_id != "OR01-RET-RER" || input.rer_revision != "0.1.0")
    return reject("INVALID_RECORD", input.contract_id, "The request must name OR01-RET 0.1.0 and OR01-RET-RER 0.1.0.");
  if (blank(input.operation_run_id) || blank(input.query_id) || blank(input.space_id) ||
      blank(input.snapshot_id) || blank(input.task_role) || blank(input.fitted_transform_id))
    return reject("INVALID_RECORD", input.operation_run_id, "Run, query, space, snapshot, task role, and fitted identity are required.");
  if ((input.profile == "text-gemini-embedding-2") != (input.provider_id == "google-gemini-embedding-2"))
    return reject("INCOMPATIBLE_ROLE_OR_SPACE", input.profile,
                  "The text profile uses provider google-gemini-embedding-2. The local profile uses a different provider id.");
  if (input.metric != "cosine" && input.metric != "l2")
    return reject("INVALID_RECORD", input.metric, "Metric must be cosine or l2.");
  if (input.normalization != "none" && input.normalization != "l2")
    return reject("INVALID_RECORD", input.normalization, "Normalization must be none or l2.");
  if (input.k == 0) return reject("INVALID_RECORD", input.query_id, "Neighborhood size k must be at least 1.");
  if (input.temporal_mode == "non_temporal") {
    if (!input.as_of && !blank(input.non_temporal_justification) && input.overlap_policy == "none" &&
        !input.fitted_available_at && !input.query.available_at && !input.query.window_start)
      ;
    else return reject("TEMPORAL_VIOLATION", input.operation_run_id,
                       "A non-temporal call has no as-of cutoff, fitted time, query time, query window, or overlap policy.");
  } else if (input.temporal_mode == "retrospective") {
    if (input.as_of) return reject("TEMPORAL_VIOLATION", input.operation_run_id, "Retrospective mode does not carry an as-of cutoff.");
  } else if (input.temporal_mode == "as_of") {
    if (!input.as_of) return reject("TEMPORAL_VIOLATION", input.operation_run_id, "As-of mode requires a cutoff.");
    if (!input.query.available_at || *input.query.available_at > *input.as_of)
      return reject("TEMPORAL_VIOLATION", input.query_id, "The query must be available at or before the cutoff.");
    if (!input.fitted_available_at || *input.fitted_available_at > *input.as_of)
      return reject("STALE_FITTED_ARTIFACT", input.fitted_transform_id,
                    "The fitted artifact must have an availability time at or before the cutoff.");
  } else return reject("TEMPORAL_VIOLATION", input.temporal_mode, "Temporal mode must be non_temporal, retrospective, or as_of.");
  if (!window_ok(input.query)) return reject("INVALID_RECORD", input.query_id, "The query window is inverted or incomplete.");
  if (input.query.representation_kind != input.representation_kind)
    return reject("INCOMPATIBLE_ROLE_OR_SPACE", input.query_id, "The query representation kind must match the SpaceRef.");
  if (input.query.values.size() != input.dimensions)
    return reject("DIMENSION_MISMATCH", input.query_id, "The query length must equal the SpaceRef dimensions.");

  std::unordered_map<std::string, const VectorPoint*> by_source;
  for (const VectorPoint& member : input.corpus) {
    if (blank(member.source_id) || blank(member.parent.id))
      return reject("INVALID_RECORD", member.source_id, "Each corpus member needs a source id and parent artifact.");
    if (!by_source.emplace(member.source_id, &member).second)
      return reject("DUPLICATE_SOURCE", member.source_id, "Corpus source ids must be unique.");
    if (member.representation_kind != input.representation_kind)
      return reject("INCOMPATIBLE_ROLE_OR_SPACE", member.source_id, "A corpus member representation kind disagrees with the SpaceRef.");
    if (member.values.size() != input.dimensions)
      return reject("DIMENSION_MISMATCH", member.source_id, "A corpus member length disagrees with the SpaceRef.");
    if (!window_ok(member)) return reject("INVALID_RECORD", member.source_id, "A corpus window is inverted or incomplete.");
    for (double component : member.values)
      if (!std::isfinite(component)) return reject("NONFINITE_VALUE", member.source_id, "Corpus values must be finite.");
    if (input.temporal_mode == "non_temporal" && (member.available_at || member.window_start))
      return reject("TEMPORAL_VIOLATION", member.source_id, "A non-temporal corpus member carries time or a window.");
    if (input.temporal_mode == "as_of" && !member.available_at)
      return reject("TEMPORAL_VIOLATION", member.source_id, "As-of corpus members need an availability time.");
  }
  for (double component : input.query.values)
    if (!std::isfinite(component)) return reject("NONFINITE_VALUE", input.query_id, "Query values must be finite.");
  if (input.overlap_policy == "fail" || input.overlap_policy == "exclude") {
    if (!input.query.window_start)
      return reject("SOURCE_WINDOW_OVERLAP", input.query_id, "Overlap handling requires a complete query window.");
  } else if (input.overlap_policy != "none") {
    return reject("INVALID_RECORD", input.overlap_policy, "Overlap policy must be none, exclude, or fail.");
  }
  if (input.overlap_policy == "fail") {
    for (const VectorPoint& member : input.corpus)
      if (overlaps(input.query, member))
        return reject("SOURCE_WINDOW_OVERLAP", member.source_id, "A corpus window overlaps the query window.");
  }

  retrieval::Corpus corpus;
  corpus.binding = {input.space_id, input.snapshot_id,
                    input.metric == "cosine" ? retrieval::Metric::cosine : retrieval::Metric::l2,
                    input.normalization == "l2" ? retrieval::Normalization::l2 : retrieval::Normalization::none,
                    input.dimensions};
  std::vector<SkipRow> held_out;
  for (const VectorPoint& member : input.corpus) {
    if (input.temporal_mode == "as_of" && member.available_at && *member.available_at > *input.as_of) {
      held_out.push_back({member.source_id, member.parent.id, "temporal_violation"});
      continue;
    }
    if (input.overlap_policy == "exclude" && overlaps(input.query, member)) {
      held_out.push_back({member.source_id, member.parent.id, "source_window_overlap"});
      continue;
    }
    corpus.records.push_back({member.source_id, member.values});
  }
  retrieval::Query query;
  query.binding = corpus.binding;
  query.query_id = input.query_id;
  query.source_id = input.query.source_id;
  query.values = input.query.values;
  retrieval::ExclusionPolicy exclusions;
  exclusions.exclude_self = input.exclude_self;
  exclusions.source_ids = input.excluded_source_ids;
  const retrieval::SearchResult found = retrieval::search(corpus, query, input.k, exclusions);
  outcome.admitted = true;
  outcome.eligible_count = found.eligible_count;
  if (found.outcome == retrieval::Outcome::failed) {
    outcome.outcome = "failed";
    outcome.failed = 1;
    outcome.diagnostics.push_back({"INVALID_RECORD", input.query_id, "Exact search rejected the saved vectors."});
    return outcome;
  }
  if (found.outcome == retrieval::Outcome::abstained) {
    outcome.outcome = "abstained";
    outcome.abstained = 1;
  } else if (found.outcome == retrieval::Outcome::empty) {
    outcome.outcome = "empty";
    outcome.succeeded = 1;
  } else {
    outcome.outcome = "ranked";
    outcome.succeeded = 1;
    for (const retrieval::Hit& hit : found.hits) {
      const VectorPoint* member = by_source.at(hit.source_id);
      outcome.hits.push_back({hit.source_id, member->parent.id, member->representation_kind,
                              hit.rank, hit.distance, hit.similarity});
    }
  }
  outcome.skipped = std::move(held_out);
  for (const retrieval::Skipped& skip : found.skipped) {
    const VectorPoint* member = by_source.at(skip.source_id);
    const char* reason = "excluded_source";
    if (skip.reason == retrieval::SkipReason::self_match) reason = "self_match";
    else if (skip.reason == retrieval::SkipReason::undefined_geometry) reason = "undefined_geometry";
    outcome.skipped.push_back({skip.source_id, member->parent.id, reason});
  }
  return outcome;
}

void quoted(std::ostream& output, std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  output << '"';
  for (const unsigned char code : value) {
    if (code == '"' || code == '\\') output << '\\' << static_cast<char>(code);
    else if (code < 0x20) output << "\\u00" << hex[code >> 4U] << hex[code & 15U];
    else output << static_cast<char>(code);
  }
  output << '"';
}
void number(std::ostream& output, double value) {
  if (!std::isfinite(value)) malformed("Unexpected nonfinite operation result");
  std::array<char, 64> bytes{};
  const auto converted = std::to_chars(bytes.data(), bytes.data() + bytes.size(), value,
                                      std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (converted.ec != std::errc{}) malformed("Could not encode operation result");
  output.write(bytes.data(), converted.ptr - bytes.data());
}
void write_result(std::ostream& output, const Request& input, const Outcome& value) {
  output << "{\"admitted\":" << (value.admitted ? "true" : "false") << ",\"engineering_only\":true,\"admission_diagnostics\":[";
  for (std::size_t index = 0; index < value.diagnostics.size(); ++index) {
    if (index != 0) output << ',';
    output << "{\"code\":"; quoted(output, value.diagnostics[index].code);
    output << ",\"record_id\":"; quoted(output, value.diagnostics[index].record_id);
    output << ",\"message\":"; quoted(output, value.diagnostics[index].message); output << '}';
  }
  output << "],\"operation_run_id\":"; quoted(output, input.operation_run_id);
  output << ",\"claim_level\":"; quoted(output, input.claim_level);
  output << ",\"profile\":"; quoted(output, input.profile);
  output << ",\"space_id\":"; quoted(output, input.space_id);
  output << ",\"snapshot_id\":"; quoted(output, input.snapshot_id);
  output << ",\"metric\":"; quoted(output, input.metric);
  output << ",\"normalization\":"; quoted(output, input.normalization);
  output << ",\"code_sha256\":"; quoted(output, input.code_sha256);
  output << ",\"configuration_sha256\":"; quoted(output, input.configuration_sha256);
  output << ",\"fitted_transform_id\":"; quoted(output, input.fitted_transform_id);
  output << ",\"outcome\":"; quoted(output, value.outcome);
  output << ",\"hits\":[";
  for (std::size_t index = 0; index < value.hits.size(); ++index) {
    if (index != 0) output << ',';
    const HitRow& hit = value.hits[index];
    output << "{\"source_id\":"; quoted(output, hit.source_id);
    output << ",\"parent_id\":"; quoted(output, hit.parent_id);
    output << ",\"representation_kind\":"; quoted(output, hit.representation_kind);
    output << ",\"rank\":" << hit.rank << ",\"distance\":"; number(output, hit.distance);
    output << ",\"similarity\":"; number(output, hit.similarity); output << '}';
  }
  output << "],\"skipped\":[";
  for (std::size_t index = 0; index < value.skipped.size(); ++index) {
    if (index != 0) output << ',';
    output << "{\"source_id\":"; quoted(output, value.skipped[index].source_id);
    output << ",\"parent_id\":"; quoted(output, value.skipped[index].parent_id);
    output << ",\"reason\":"; quoted(output, value.skipped[index].reason); output << '}';
  }
  output << "],\"eligible_count\":" << value.eligible_count;
  output << ",\"summary\":{\"requested\":" << value.requested << ",\"succeeded\":" << value.succeeded
         << ",\"abstained\":" << value.abstained << ",\"failed\":" << value.failed
         << ",\"unresolved\":" << value.unresolved << ",\"success_rate\":";
  if (!value.admitted) output << "null";
  else number(output, static_cast<double>(value.succeeded) / static_cast<double>(value.requested));
  output << "}}\n";
}

std::string stdin_bytes() {
  std::string bytes;
  std::array<char, 65536> buffer{};
  while (std::cin) {
    std::cin.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = std::cin.gcount();
    if (static_cast<std::size_t>(count) > maximum_request - bytes.size()) malformed("Private request exceeds 64 MiB");
    bytes.append(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!std::cin.eof()) malformed("Could not read private request");
  return bytes;
}
}  // namespace

int main(int argc, char**) {
  try {
    if (argc != 1) malformed("Private bridge accepts request bytes on stdin only");
    Reader reader(stdin_bytes());
    const Request input = request(reader);
    write_result(std::cout, input, evaluate(input));
    if (!std::cout) malformed("Could not write private result");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "retrieval private bridge: " << error.what() << '\n';
    return 2;
  }
}
