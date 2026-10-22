// Copyright (c) 2026 Waajacu. See LICENSE at repository root.
// Study-owned read-only CACHE88 -> generic observation windows. No encoder.
#include "embedding/data/observation_windows.h"
#include "cuwacunu/artifacts/v1/sidecar.h"
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>

namespace {
namespace fs=std::filesystem;
namespace ow=embedding::data::observation_windows;
namespace artifacts=cuwacunu::artifacts::v1;
constexpr int64_t interval=60000,history=128,train_begin=1704067259999LL,
    train_end=1705881599999LL,validation_begin=1705881659999LL,
    validation_end=1706486399999LL,expected_rows=44640,expected_bytes=3928320;
const std::array<std::string,3> assets{"BTCUSDT","ETHUSDT","SOLUSDT"};
const std::array<std::string,3> expected_hashes{
  "67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58",
  "0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa",
  "364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7"};
const std::array<std::string,9> features{"open_price","high_price","low_price","close_price",
  "volume","quote_asset_volume","number_of_trades","taker_buy_base_volume","taker_buy_quote_volume"};
void check(bool value,const std::string &message){TORCH_CHECK(value,"[V17 windows] ",message);}
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';
  for(const unsigned char c:value){
    if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;
  }
  out<<'"';return out.str();
}
void write_text(const fs::path &path,const std::string &value){
  check(!fs::exists(path),"preserve existing output: "+path.string());
  std::ofstream out(path,std::ios::binary);check(bool(out),"open exclusive text output");
  out<<value<<'\n';out.close();check(bool(out),"complete text output");
}
struct Args {
  std::string mode;fs::path output;std::vector<fs::path> caches;std::vector<std::string> hashes;
  Args(int argc,char **argv){
    check(argc>1&&(argc-1)%2==0,"--mode diagnostic|train|probes --cache PATH --cache-sha256 SHA (each three times, BTC/ETH/SOL) --output NEWLEAF");
    for(int i=1;i<argc;i+=2){const std::string key=argv[i],value=argv[i+1];
      if(key=="--mode"){check(mode.empty(),"duplicate mode");mode=value;}
      else if(key=="--output"){check(output.empty(),"duplicate output");output=value;}
      else if(key=="--cache")caches.emplace_back(value);
      else if(key=="--cache-sha256")hashes.push_back(value);
      else check(false,"unknown option: "+key);
    }
    check(mode=="diagnostic"||mode=="train"||mode=="probes","one supported mode required");
    check(caches.size()==3&&hashes.size()==3&&!output.empty(),"three ordered cache/hash pairs and output required");
    output=fs::absolute(output).lexically_normal();
    check(!fs::exists(output)&&fs::is_directory(output.parent_path()),"new output leaf in an existing parent required");
    std::set<fs::path> unique;
    for(size_t c=0;c<3;++c){
      check(artifacts::is_sha256(hashes[c])&&hashes[c]==expected_hashes[c],"cache pin/order must match declared January BTC/ETH/SOL sources");
      caches[c]=fs::canonical(caches[c]);
      check(fs::is_regular_file(caches[c])&&fs::file_size(caches[c])==expected_bytes&&unique.insert(caches[c]).second,
          "three distinct complete original January caches required");
      check(artifacts::sha256_file(caches[c])==hashes[c],"cache SHA256 mismatch before read");
    }
  }
  void verify()const{
    for(size_t c=0;c<3;++c)check(fs::file_size(caches[c])==expected_bytes&&artifacts::sha256_file(caches[c])==hashes[c],
        "bound cache changed: "+caches[c].string());
  }
};
#pragma pack(push,1)
struct Record {
  int64_t open_time;double open_price,high_price,low_price,close_price,volume;
  int64_t close_time;double quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume;
};
#pragma pack(pop)
static_assert(sizeof(Record)==88&&offsetof(Record,close_time)==48&&
    std::endian::native==std::endian::little&&std::is_trivially_copyable_v<Record>);
struct Channel {
  std::vector<char> bytes;int64_t first{0};
  Channel(const fs::path &path,const std::string &pin){
    std::ifstream in(path,std::ios::binary);check(bool(in),"open cache");
    bytes.resize(expected_bytes);in.read(bytes.data(),expected_bytes);
    check(in.gcount()==expected_bytes&&!in.bad(),"read complete cache");
    check(artifacts::sha256_bytes(std::string_view(bytes.data(),bytes.size()))==pin,"loaded cache buffer differs from bound bytes");
    first=close_at(0);check(first==train_begin,"original January first close required");
    // Whole-cache metadata validation reads only close-time bytes. Observation
    // fields are decoded only inside selected, role-admitted retained windows.
    for(int64_t i=0;i<expected_rows;++i)check(close_at(i)==first+i*interval,"exact original minute lattice required");
  }
  int64_t close_at(int64_t index)const{
    check(index>=0&&index<expected_rows,"cache metadata index");int64_t close;
    std::memcpy(&close,bytes.data()+88*index+48,sizeof(close));return close;
  }
  Record at_close(int64_t close)const{
    check(close>=first&&(close-first)%interval==0,"exact cache close key required");
    const auto index=(close-first)/interval;check(index<expected_rows,"cache observation index");
    Record record;std::memcpy(&record,bytes.data()+88*index,sizeof(record));
    check(record.close_time==close,"observation close-time association");return record;
  }
};
std::vector<int64_t> anchors(int64_t begin,int64_t end,int64_t selection_history,int64_t count,bool nonoverlap){
  check(begin>=0&&end>=begin&&(end-begin)%interval==0&&selection_history>0&&count>=2,"selection geometry");
  const auto stride=nonoverlap?selection_history*interval:interval;
  const auto first=begin+(selection_history-1)*interval;
  check(first<=end,"complete selected role support");const auto available=(end-first)/stride+1;
  check(count<=available,"enough selected windows");std::vector<int64_t> rows;
  for(int64_t i=0;i<count;++i)rows.push_back(first+static_cast<int64_t>(static_cast<__int128>(i)*(available-1)/(count-1))*stride);
  check(std::set<int64_t>(rows.begin(),rows.end()).size()==rows.size(),"unique source row IDs");return rows;
}
bool same_bits(const torch::Tensor &left,const torch::Tensor &right){
  if(left.device()!=right.device()||left.scalar_type()!=right.scalar_type()||left.sizes()!=right.sizes())return false;
  const auto a=left.contiguous(),b=right.contiguous();return std::memcmp(a.const_data_ptr(),b.const_data_ptr(),a.numel()*a.element_size())==0;
}
struct Output {
  std::string file,role,split;std::vector<int64_t> rows;int64_t begin,end,selection_history;
  int64_t observed_cells{0},genuine_f64_cells{0};fs::path path;
};
Output make(const Args &args,const std::vector<Channel> &channels,const std::string &source,
    const std::string &file,const std::string &role,int64_t begin,int64_t end,
    int64_t selection_history,int64_t count,bool nonoverlap){
  Output out{file,role,"",anchors(begin,end,selection_history,count,nonoverlap),begin,end,selection_history,0,0,args.output/file};
  const auto row_digest=artifacts::digest_fields("cpb-v17-window-row-ids-v1",[&]{std::vector<std::string> values;
    for(const auto row:out.rows)values.push_back(std::to_string(row));return values;}());
  out.split="cpb-v17-balance-v1;role="+role+";begin="+std::to_string(begin)+";end="+std::to_string(end)+
      ";selection-history="+std::to_string(selection_history)+";retained128;"+(nonoverlap?"nonoverlap":"overlap")+
      ";rows="+std::to_string(count)+";row-sha256="+row_digest;
  ow::Dataset data;
  data.values=torch::zeros({count,3,history,9},torch::kFloat64);
  data.observed=torch::zeros(data.values.sizes(),torch::kBool);
  data.timestamps=torch::zeros({count,history},torch::kFloat64);
  data.channel_ids=torch::tensor({0,1,2},torch::kInt64);
  data.cutoffs=torch::tensor(out.rows,torch::kInt64).to(torch::kFloat64);
  data.row_ids=torch::tensor(out.rows,torch::kInt64);data.sampling_interval=double(interval);
  data.source_identity=source;data.split_identity=out.split;data.role=role;
  auto x=data.values.accessor<double,4>();auto mask=data.observed.accessor<bool,4>();auto times=data.timestamps.accessor<double,2>();
  for(int64_t b=0;b<count;++b){const auto cutoff=out.rows[b];
    check(cutoff<=end&&cutoff-(selection_history-1)*interval>=begin,"complete original selected history within role");
    for(int64_t t=0;t<history;++t){const auto close=cutoff-(history-1-t)*interval;
      check(close>=begin&&close<=end,"retained observation belongs to declared role");times[b][t]=double(close);
      for(int64_t c=0;c<3;++c){const auto record=channels[c].at_close(close);
        const std::array<double,9> values{record.open_price,record.high_price,record.low_price,record.close_price,
          record.volume,record.quote_asset_volume,record.number_of_trades,record.taker_buy_base_volume,record.taker_buy_quote_volume};
        for(int64_t f=0;f<9;++f){const auto value=values[f];const bool observed=record.open_time!=INT64_MIN&&std::isfinite(value);
          check(!observed||std::isfinite(static_cast<float>(value)),"observed crop overflows historical F32 parent cast");
          mask[b][c][t][f]=observed;x[b][c][t][f]=observed?value:0.;
          if(observed){++out.observed_cells;check(std::bit_cast<uint64_t>(x[b][c][t][f])==std::bit_cast<uint64_t>(value),"genuine F64 bits retained");
            if(value!=static_cast<double>(static_cast<float>(value)))++out.genuine_f64_cells;}
        }
      }
    }
  }
  ow::write(out.path,data);const auto restored=ow::read(out.path);
  check(restored.source_identity==source&&restored.split_identity==out.split&&restored.role==role&&restored.sampling_interval==double(interval),
      "typed archive identity reload");
  for(const auto &pair:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{data.values,restored.values},{data.observed,restored.observed},
      {data.timestamps,restored.timestamps},{data.channel_ids,restored.channel_ids},{data.cutoffs,restored.cutoffs},{data.row_ids,restored.row_ids}})
    check(same_bits(pair.first,pair.second),"exact typed archive tensor reload");
  return out;
}
int execute(int argc,char **argv){
  if(argc==2&&std::string(argv[1])=="--help"){
    std::cout<<"cpb_v17_windows --mode diagnostic|train|probes --cache PATH --cache-sha256 SHA (three each, ordered BTC/ETH/SOL) --output NEWLEAF\n"
      <<"diagnostic:64 TRAIN H128; train:1024 TRAIN H128; probes:112 TRAIN/32 VALIDATION H256 anchors retaining128. No model/fit/training.\n";return 0;
  }
  const Args args(argc,argv);torch::set_num_threads(1);torch::set_num_interop_threads(1);
  std::vector<Channel> channels;for(size_t c=0;c<3;++c)channels.emplace_back(args.caches[c],args.hashes[c]);args.verify();
  const auto source="cpb-v17-CACHE88-Jan2024-BTC-ETH-SOL-9features-"+artifacts::digest_fields("cpb-v17-window-source-v1",
      {assets[0],args.hashes[0],assets[1],args.hashes[1],assets[2],args.hashes[2],"F64-values;bool-observed;ids0,1,2;minute-close"});
  check(fs::create_directory(args.output),"exclusive output leaf");
  std::vector<Output> outputs;
  if(args.mode=="diagnostic")outputs.push_back(make(args,channels,source,"diagnostic-TRAIN.pt","TRAIN",train_begin,train_end,128,64,false));
  else if(args.mode=="train")outputs.push_back(make(args,channels,source,"calibration-TRAIN.pt","TRAIN",train_begin,train_end,128,1024,false));
  else{
    outputs.push_back(make(args,channels,source,"probe-TRAIN.pt","TRAIN",train_begin,train_end,256,112,true));
    outputs.push_back(make(args,channels,source,"probe-VALIDATION.pt","VALIDATION",validation_begin,validation_end,256,32,true));
  }
  args.verify();const auto executable=fs::canonical("/proc/self/exe");
  std::ostringstream json;json<<"{\"kind\":\"cpb_v17_balance_window_preparation_v1\",\"status\":\"passed\",\"mode\":"<<quote(args.mode)
      <<",\"source_identity\":"<<quote(source)<<",\"dataset_kind\":"<<quote(ow::kKind)<<",\"retained_history\":128,\"channels\":3,\"features\":[";
  for(size_t f=0;f<features.size();++f){if(f)json<<',';json<<quote(features[f]);}
  json<<"],\"preprocessing_fits\":0,\"encoder_calls\":0,\"optimizer_updates\":0,\"actual_role_membership_verified\":true,\"inputs\":[";
  for(size_t c=0;c<3;++c){if(c)json<<',';json<<"{\"asset\":"<<quote(assets[c])<<",\"semantic_id\":"<<c<<",\"path\":"<<quote(args.caches[c].generic_string())
      <<",\"bytes\":"<<expected_bytes<<",\"sha256\":"<<quote(args.hashes[c])<<'}';}
  json<<"],\"executable\":{\"path\":"<<quote(executable.generic_string())<<",\"bytes\":"<<fs::file_size(executable)
      <<",\"sha256\":"<<quote(artifacts::sha256_file(executable))<<"},\"artifacts\":[";
  for(size_t i=0;i<outputs.size();++i){const auto &out=outputs[i];if(i)json<<',';
    json<<"{\"file\":"<<quote(out.file)<<",\"role\":"<<quote(out.role)<<",\"split_identity\":"<<quote(out.split)<<",\"rows\":"<<out.rows.size()
        <<",\"role_begin\":"<<out.begin<<",\"role_end\":"<<out.end<<",\"selection_history\":"<<out.selection_history
        <<",\"retained_history\":128,\"observed_cells\":"<<out.observed_cells<<",\"genuine_f64_cells\":"<<out.genuine_f64_cells
        <<",\"bytes\":"<<fs::file_size(out.path)<<",\"sha256\":"<<quote(artifacts::sha256_file(out.path))<<",\"row_ids\":[";
    for(size_t r=0;r<out.rows.size();++r){if(r)json<<',';json<<out.rows[r];}json<<"]}";
  }
  json<<"]}";args.verify();write_text(args.output/"preparation.json",json.str());std::cout<<json.str()<<'\n';return 0;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
