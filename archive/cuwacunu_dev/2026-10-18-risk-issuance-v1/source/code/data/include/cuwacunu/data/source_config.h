// SPDX-License-Identifier: MIT
#pragma once
#include <string>
#include "cuwacunu/data/instrument_signature.h"
#include "cuwacunu/data/interval.h"

namespace cuwacunu::data {
// Plain source and retrieval fields extracted from the old source contract.
// No graph, parser, registry service or runtime is required.
struct source_form_t {
  std::string instrument;
  interval_type_e interval;
  std::string record_type;
  std::string market_type;
  std::string venue;
  std::string base_asset;
  std::string quote_asset;
  std::string source_kind{"real"};
  std::string source;
  [[nodiscard]] instrument_signature_t instrument_signature() const {
    return {.symbol = instrument, .record_type = record_type,
            .market_type = market_type, .venue = venue,
            .base_asset = base_asset, .quote_asset = quote_asset};
  }
};
struct channel_form_t {
  interval_type_e interval;
  std::string active;
  std::string record_type;
  std::string input_length;
  std::string future_length;
  std::string channel_weight;
  std::string normalization_policy;
};
} // namespace cuwacunu::data
