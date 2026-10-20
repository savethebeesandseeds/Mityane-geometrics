// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cache88.h"
#include <clocale>
#include <iostream>
#include <sstream>
#include <vector>
namespace cd = mityane::data::cpb_v15;
namespace {
const std::string row = "1706745600000,1.25,2.5,1.125,2,10,1706745659999,20,3,4,5,0";
void rejects(const std::string& text, int64_t rows = 1) {
  bool refused = false;
  try { std::istringstream stream(text); cd::convert_csv(stream, 1706745600000LL, rows, [](const auto&){}); }
  catch (const std::exception&) { refused = true; }
  cd::require(refused, "negative artificial CSV fixture was accepted");
}
std::string replace(std::string text, const std::string& before, const std::string& after) {
  const auto offset = text.find(before); cd::require(offset != std::string::npos, "fixture token missing");
  text.replace(offset, before.size(), after); return text;
}
}
int main() {
  try {
    cd::require(std::setlocale(LC_NUMERIC, "C") != nullptr, "fixture C locale");
    std::vector<cd::Cache88> records;
    std::istringstream stream(row + "\r\n");
    const auto result = cd::convert_csv(stream, 1706745600000LL, 1, [&](const auto& value){records.push_back(value);});
    cd::require(result.rows == 1 && result.cache_bytes == 88 && result.last_close_ms == 1706745659999LL, "conversion chronology/count");
    const auto& record = records.at(0);
    cd::require(record.open_price == 1.25 && record.high_price == 2.5 && record.low_price == 1.125 && record.close_price == 2
      && record.volume == 10 && record.quote_asset_volume == 20 && record.number_of_trades == 3
      && record.taker_buy_base_volume == 4 && record.taker_buy_quote_volume == 5, "wire field order");
    std::array<unsigned char,88> bytes{}; std::memcpy(bytes.data(), &record, bytes.size());
    int64_t closing{}; std::memcpy(&closing, bytes.data() + 48, sizeof closing);
    cd::require(closing == 1706745659999LL && std::bit_cast<uint64_t>(record.open_price) == 0x3ff4000000000000ULL, "wire offsets and exact binary64");
    for (const auto& text : std::vector<std::string>{"", row + "\n" + row, row.substr(0,row.size()-2), row + ",0", row + ",",
      replace(row,"1.25","NaN"), replace(row,"1.25","1.25junk"), replace(row,"1.25","0x1p0"), replace(row,"1.25","1e999"),
      replace(row,"1.25","0"), replace(row,",10,",",-1,"), replace(row,",3,",",3.5,"), replace(row,",3,",",2147483648,"),
      replace(row,"1706745659999","1706745660000"), replace(row,"1706745600000","1706745600001"), "header," + row,
      replace(row,",0",",NaN"), row + "\n\n", replace(row,",3,",", 3,")}) rejects(text);
    rejects(row, 2);
    std::istringstream empty;
    bool geometry_refused = false;
    try { cd::convert_csv(empty, 1706745600001LL, 1, [](const auto&){}); } catch (...) { geometry_refused = true; }
    cd::require(geometry_refused, "off-grid geometry refused");
    std::cout << "Artificial CSV/CACHE88 fixtures passed; no market files opened.\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
