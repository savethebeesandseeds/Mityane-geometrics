// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Artificial all-past data contract. No market data, encoder or readout fitting.
#include "cuwacunu/training/market_data.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace {
namespace md = cuwacunu::training::market_data;
namespace enc = cuwacunu::encoder::primary;
namespace data = cuwacunu::data;
namespace fs = std::filesystem;
constexpr int64_t base = 1704067259999;
int64_t key(int64_t tick) { return base+tick*md::sampling_interval_ms; }
void check(bool value, const char *message) { TORCH_CHECK(value,"[market data contract] ",message); }
template<class F> void rejects(F action, const char *message) {
  bool rejected=false;
  try { action(); } catch (const std::exception &) { rejected=true; }
  check(rejected,message);
}
struct Fixture {
  fs::path path;
  Fixture() {
    const auto pattern=(fs::temp_directory_path()/"market-data-contract-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(),pattern.end()); bytes.push_back('\0');
    const auto created=::mkdtemp(bytes.data()); check(created != nullptr,"private temporary fixture"); path=created;
  }
  ~Fixture() { std::error_code error; fs::remove_all(path,error); }
};
data::kline_t candle(int channel, int64_t tick, double adjustment=0.) {
  const auto price=1000.+100*channel+.125*tick+adjustment;
  const auto volume=50.+channel+.25*tick;
  data::kline_t row{};
  row.open_time=key(tick)-59999; row.close_time=key(tick);
  row.open_price=price; row.high_price=price+.25; row.low_price=price-.25; row.close_price=price+.03125;
  row.volume=volume; row.quote_asset_volume=volume*price; row.number_of_trades=10+tick%10;
  row.taker_buy_base_volume=volume*.4; row.taker_buy_quote_volume=row.taker_buy_base_volume*price;
  return row;
}
void write_channel(const fs::path &path, int channel) {
  std::ofstream csv(path); check(bool(csv),"fixture CSV open");
  for (int64_t tick=0; tick<900; ++tick) {
    if (channel == 1 && tick == 450) continue;
    auto row=candle(channel,tick);
    if (channel == 2 && tick == 490) row.volume=std::numeric_limits<double>::quiet_NaN();
    row.to_csv(csv); csv << '\n';
    if (channel == 0 && tick == 480) { candle(channel,tick,11.).to_csv(csv); csv << '\n'; }
  }
  csv.close(); check(bool(csv),"fixture CSV write");
}
void shared_endpoints_and_roles(const std::vector<fs::path> &files) {
  auto short_config=enc::default_config(); short_config.sampling_interval=60000.; short_config.channel_ids.clear();
  auto long_config=short_config; long_config.history_length=256;
  md::Dataset short_data,long_data;
  for (const auto &file:files) {
    short_data.add_dataset(file.string(),128,0,"none",false);
    long_data.add_dataset(file.string(),256,0,"none",false);
  }
  const auto begin=key(256),end=key(767);
  const auto anchors=md::uniform_anchors(begin,end,256,2,true);
  check(anchors == std::vector<int64_t>({key(511),key(767)}),"two adjacent complete H256 windows tile the declared role");
  const auto short_input=md::load_batch(short_data,anchors,begin,end,short_config);
  const auto long_input=md::load_batch(long_data,anchors,begin,end,long_config);
  check(short_input.data.device().is_cpu() && long_input.data.device().is_cpu() &&
            short_input.data.sizes() == torch::IntArrayRef({2,3,128,9}) &&
            long_input.data.sizes() == torch::IntArrayRef({2,3,256,9}),"both configured histories stay raw CPU BCHF");
  check(torch::equal(short_input.endpoints,long_input.endpoints) &&
            torch::equal(short_input.endpoints,torch::tensor(anchors,torch::kFloat64)) &&
            torch::equal(short_input.data,long_input.data.slice(2,128,256)) &&
            torch::equal(short_input.observed,long_input.observed.slice(2,128,256)),
        "common endpoints retain exactly identical H128 suffix values and masks");
  check(torch::equal(short_input.channel_ids,torch::tensor({0,1,2},torch::kInt64)),
        "empty configured IDs resolve to declared semantic defaults");
  check(!short_input.observed[0][1][66].any().item<bool>() &&
            short_input.data[0][1][66].eq(0).all().item<bool>(),"gap at tick450 keeps its actual timestamp and becomes an invalid zero row");
  check(short_input.data[0][0][96][3].item<double>() == candle(0,480,11.).close_price,
        "last duplicate candle is retained at the same endpoint in both histories");
  check(!short_input.observed[0][2][106][4].item<bool>() &&
            short_input.observed[0][2][106][3].item<bool>() &&
            short_input.data[0][2][106][4].item<double>() == 0.,
        "one nonfinite activity coordinate masks only itself across both histories");
  auto first_sample=short_data.get_by_anchor_key(anchors.front());
  check(first_sample.future_features.sizes() == torch::IntArrayRef({3,0,9}) &&
            first_sample.future_mask.sizes() == torch::IntArrayRef({3,0}) &&
            first_sample.future_keys.sizes() == torch::IntArrayRef({3,0}),"history loader physically returns no future rows");
  const auto early=key(383);
  check(md::load_batch(short_data,{early},begin,end,short_config).endpoints.item<double>() == early,
        "first role-contained H128 window is legal");
  rejects([&] { md::load_batch(long_data,{early},begin,end,long_config); },
          "an H128-legal endpoint crossing the H256 role boundary is rejected");
  rejects([&] { md::load_batch(long_data,{end+60000},begin,end,long_config); },"an existing source endpoint beyond the role end is rejected");
  rejects([&] { md::load_batch(long_data,{},begin,end,long_config); },"an empty batch is rejected");
  rejects([&] { md::load_batch(long_data,anchors,begin,end,short_config); },"dataset/model history mismatch is rejected");
  auto shuffled=md::load_batch(long_data,{anchors[1],anchors[0],anchors[0]},begin,end,long_config);
  check(torch::equal(shuffled.endpoints,torch::tensor({anchors[1],anchors[0],anchors[0]},torch::kFloat64)) &&
            torch::equal(shuffled.data[1],shuffled.data[2]),"TRAIN sampling order and intentional repeated anchors remain exact");
  md::Dataset future;
  for (const auto &file:files) future.add_dataset(file.string(),128,1,"none",false);
  rejects([&] { md::load_batch(future,anchors,begin,end,short_config); },"all-past study helper refuses a nonzero future horizon");
}
void selection_bounds_and_source_order(const fs::path &directory) {
  const auto begin=key(0),end=key(899);
  const auto dense=md::uniform_anchors(begin,end,128,31,false);
  check(dense.size() == 31 && dense.front() == key(127) && dense.back() == end,
        "dense selection spans the first complete window through the final endpoint");
  for (size_t j=1; j<dense.size(); ++j) check(dense[j]>dense[j-1] && (dense[j]-base)%60000 == 0,
                                           "selected endpoints remain distinct and minute-aligned");
  rejects([&] { md::uniform_anchors(begin,key(126),128,2,false); },"insufficient complete history is rejected");
  rejects([&] { md::uniform_anchors(begin,end,256,4,true); },"too many nonoverlapping H256 windows are rejected");
  rejects([&] { md::uniform_anchors(begin,end,128,1,false); },"a count without a defined span policy is rejected");
  rejects([&] { md::uniform_anchors(begin,end+1,128,2,false); },"off-grid role boundary is rejected");
  const auto maximum=std::numeric_limits<int64_t>::max();
  rejects([&] { md::uniform_anchors(maximum-60000,maximum,3,2,false); },"timestamp arithmetic overflow is rejected before allocation");
  check(md::slice({1,2,3},1,std::numeric_limits<size_t>::max()) == std::vector<int64_t>({2,3}) &&
            md::slice({1,2,3},3,1).empty(),"batch slicing handles end offsets and overflowing requested counts");
  rejects([&] { md::slice({1,2,3},4,1); },"out-of-range slice offset is rejected");
  const auto path=directory/"late-out-of-order.csv";
  std::ofstream csv(path);
  for (int64_t tick=0; tick<80; ++tick) { candle(0,tick).to_csv(csv); csv << '\n'; }
  candle(0,50).to_csv(csv); csv << '\n'; csv.close(); check(bool(csv),"late-order fixture write");
  rejects([&] { data::sanitize_csv_into_binary_file<data::kline_t>(path.string(),"none",false,2); },
          "late source timestamp regression after bootstrap is rejected rather than sorted");
  check(!fs::exists(data::raw_binary_for_csv(path.string())),"rejected source order publishes no derived cache");
}
} // namespace

int main() try {
  torch::set_num_threads(1); Fixture fixture;
  std::vector<fs::path> files;
  for (int channel=0; channel<3; ++channel) {
    files.push_back(fixture.path/("channel-"+std::to_string(channel)+".csv")); write_channel(files.back(),channel);
  }
  shared_endpoints_and_roles(files); selection_bounds_and_source_order(fixture.path);
  std::cout << "PASS market data contract: real mmap raw9 H128/H256 common endpoints and exact suffix/masks, "
      "history-specific chronological guards, gap/last-duplicate/nonfinite support, future0, late source-order rejection, "
      "strict selection/slice bounds; CPU arrays only, no encoder/scaler/head/market quality execution\n";
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
