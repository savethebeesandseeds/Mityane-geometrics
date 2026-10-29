// SPDX-License-Identifier: MIT
#include "embedding/encoders/timeframe_channel.h"

#include <iostream>
#include <string>
#include <vector>

namespace tf = embedding::encoders::timeframe_channel;

static int failures = 0;

static void expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    ++failures;
  }
}

int main() {
  expect(tf::kKlineCount == 16, "sixteen kline intervals");
  expect(tf::find(4) && std::string(tf::find(4)->key) == "1m" && tf::find(4)->step_ms == 60000,
         "1m is torch ordinal 4");
  expect(tf::find(9) && std::string(tf::find(9)->key) == "1h" && tf::find(9)->step_ms == 3600000,
         "1h is torch ordinal 9");
  expect(tf::find(15) && std::string(tf::find(15)->key) == "1d" && tf::find(15)->step_ms == 86400000,
         "1d is torch ordinal 15");
  expect(tf::find(17) && std::string(tf::find(17)->key) == "1w" && tf::find(17)->step_ms == 604800000,
         "1w is torch ordinal 17");
  expect(tf::find(11) && std::string(tf::find(11)->key) == "4h", "11 is 4h");
  expect(tf::find(18) && std::string(tf::find(18)->key) == "1M" && tf::find(18)->step_ms == 0,
         "1M has no fixed step");
  expect(tf::find(0) == nullptr && tf::find(1) == nullptr && tf::find(2) == nullptr,
         "utility ordinals are not channels");
  expect(tf::find(25) == nullptr && tf::find(32) == nullptr, "old fixture ids 25 and 32 are not intervals");

  std::string error;
  const int64_t minutes_hours_weeks[] = {4, 9, 17};
  expect(tf::timeframe_ids_ok(minutes_hours_weeks, 3, &error), "1m, 1h, 1w are a legal axis");
  const int64_t minutes_hours_days[] = {4, 9, 15};
  expect(tf::timeframe_ids_ok(minutes_hours_days, 3, &error), "1m, 1h, 1d are a legal axis of size 3");
  expect(tf::admit_configured_ids(minutes_hours_days, 3, &error), "1m, 1h, 1d stay admitted");
  const int64_t withdrawn[] = {0, 1, 2};
  expect(!tf::timeframe_ids_ok(withdrawn, 3, &error), "new settings refuse the instrument binding");
  expect(error.find("withdrawn instrument binding") != std::string::npos, "refusal names the binding");
  expect(tf::is_withdrawn_instrument_binding(withdrawn, 3), "recorded {0,1,2} still has a name");
  const int64_t reversed[] = {2, 0, 1};
  expect(tf::admit_configured_ids(reversed, 3, &error), "a recorded checkpoint may open any order of {0,1,2}");
  expect(!tf::admit_configured_ids(std::vector<int64_t>{0}, &error), "a lone 0 is not the binding");
  expect(!tf::admit_configured_ids(std::vector<int64_t>{11, 18, 25}, &error), "25 is refused");
  expect(tf::timeframe_ids_ok(std::vector<int64_t>{11, 18, 4, 9}, &error), "engineering ids 4h, 1M, 1m, 1h");
  expect(!tf::timeframe_ids_ok(std::vector<int64_t>{11, 11}, &error), "duplicate timeframe refused");
  expect(!tf::timeframe_ids_ok(std::vector<int64_t>{}, &error), "empty axis refused");

  if (failures != 0) {
    std::cerr << failures << " failed\n";
    return 1;
  }
  std::cout << "timeframe_channel_ok\n";
  return 0;
}
