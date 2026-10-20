#include "mityane/saved_tensor/v1/reader.h"

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace S = mityane::saved_tensor::v1;
namespace {
using Bytes = std::vector<std::uint8_t>;
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
void rejects(const std::function<void()>& action, const char* why) {
  try { action(); } catch (const S::DecodeError&) { return; } throw std::runtime_error(why);
}
void word(Bytes& out, std::uint64_t value, unsigned bytes) { for (unsigned i = 0; i < bytes; ++i) out.push_back(static_cast<std::uint8_t>(value >> (i * 8))); }
void literal(Bytes& out, const std::string& text) { out.insert(out.end(), text.begin(), text.end()); }
std::uint32_t crc(const Bytes& bytes) {
  std::uint32_t value = 0xffffffff;
  for (auto byte : bytes) { value ^= byte; for (int i = 0; i < 8; ++i) value = (value >> 1) ^ ((value & 1) ? 0xedb88320 : 0); }
  return ~value;
}
void unicode(Bytes& out, const std::string& text) { out.push_back('X'); word(out, text.size(), 4); literal(out, text); }
void integer(Bytes& out, std::int64_t value) { out.push_back(0x8a); out.push_back(8); word(out, std::bit_cast<std::uint64_t>(value), 8); }
void dimensions(Bytes& out, const std::vector<std::int64_t>& values) { out.push_back('('); for (auto value : values) integer(out, value); out.push_back('t'); }
void tensor(Bytes& out, const std::string& key, const std::string& dtype, const std::vector<std::int64_t>& shape,
            const std::vector<std::int64_t>& stride, std::int64_t offset, std::int64_t storage_size,
            const std::string& device = "cpu") {
  literal(out, "ctorch._utils\n_rebuild_tensor_v2\n(");
  out.push_back('('); unicode(out, "storage"); literal(out, "ctorch\n" + dtype + "\n"); unicode(out, key); unicode(out, device);
  integer(out, storage_size); out.push_back('t'); out.push_back('Q'); integer(out, offset); dimensions(out, shape); dimensions(out, stride);
  out.push_back(0x89); literal(out, "ccollections\nOrderedDict\n)R"); out.push_back('t'); out.push_back('R');
}
Bytes pickle(const std::vector<std::int64_t>& shape = {2, 2}, const std::vector<std::int64_t>& strides = {1, 2},
             std::int64_t offset = 0, const std::string& dtype = "DoubleStorage", const std::string& device = "cpu",
             bool alias = false, std::int64_t storage_size = 4) {
  Bytes out{0x80, 2, '}'}; unicode(out, "x"); tensor(out, "0", dtype, shape, strides, offset, storage_size, device); out.push_back('s');
  if (alias) { unicode(out, "excluded"); tensor(out, "0", dtype, {4}, {1}, 0, storage_size); out.push_back('s'); }
  out.push_back('.'); return out;
}
struct Member { std::string name; Bytes bytes; };
// This fixture writer emits only records needed by these tests. It does not use
// the decoder, Torch, Python, a pickle serializer or a ZIP package.
Bytes zip(const std::vector<Member>& members, bool wide = false, bool descriptors = false, bool classic_end_values = false) {
  Bytes out, central;
  const std::uint16_t flags = descriptors ? 8 : 0;
  for (const auto& member : members) {
    const auto offset = out.size(); Bytes extra;
    if (wide) { word(extra, 1, 2); word(extra, 16, 2); word(extra, member.bytes.size(), 8); word(extra, member.bytes.size(), 8); }
    word(out, 0x04034b50, 4); word(out, wide ? 45 : 20, 2); word(out, flags, 2); word(out, 0, 2); word(out, 0, 4);
    word(out, descriptors ? 0 : crc(member.bytes), 4);
    word(out, wide ? UINT32_MAX : (descriptors ? 0 : member.bytes.size()), 4); word(out, wide ? UINT32_MAX : (descriptors ? 0 : member.bytes.size()), 4);
    word(out, member.name.size(), 2); word(out, extra.size(), 2); literal(out, member.name); out.insert(out.end(), extra.begin(), extra.end());
    out.insert(out.end(), member.bytes.begin(), member.bytes.end());
    if (descriptors) { word(out, 0x08074b50, 4); word(out, crc(member.bytes), 4); word(out, member.bytes.size(), wide ? 8 : 4); word(out, member.bytes.size(), wide ? 8 : 4); }
    extra.clear();
    if (wide) { word(extra, 1, 2); word(extra, 24, 2); word(extra, member.bytes.size(), 8); word(extra, member.bytes.size(), 8); word(extra, offset, 8); }
    word(central, 0x02014b50, 4); word(central, wide ? 45 : 20, 2); word(central, wide ? 45 : 20, 2); word(central, flags, 2); word(central, 0, 2); word(central, 0, 4);
    word(central, crc(member.bytes), 4); word(central, wide ? UINT32_MAX : member.bytes.size(), 4); word(central, wide ? UINT32_MAX : member.bytes.size(), 4);
    word(central, member.name.size(), 2); word(central, extra.size(), 2); word(central, 0, 2); word(central, 0, 2); word(central, 0, 2); word(central, 0, 4); word(central, wide ? UINT32_MAX : offset, 4);
    literal(central, member.name); central.insert(central.end(), extra.begin(), extra.end());
  }
  const auto start = out.size(); out.insert(out.end(), central.begin(), central.end());
  if (wide) {
    const auto zpos = out.size(); word(out, 0x06064b50, 4); word(out, 44, 8); word(out, 45, 2); word(out, 45, 2); word(out, 0, 4); word(out, 0, 4);
    word(out, members.size(), 8); word(out, members.size(), 8); word(out, central.size(), 8); word(out, start, 8);
    word(out, 0x07064b50, 4); word(out, 0, 4); word(out, zpos, 8); word(out, 1, 4);
  }
  const bool sentinel = wide && !classic_end_values;
  word(out, 0x06054b50, 4); word(out, 0, 2); word(out, 0, 2); word(out, sentinel ? 65535 : members.size(), 2); word(out, sentinel ? 65535 : members.size(), 2);
  word(out, sentinel ? UINT32_MAX : central.size(), 4); word(out, sentinel ? UINT32_MAX : start, 4); word(out, 0, 2); return out;
}
void save(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc); file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); check(bool(file), "fixture write");
}
Bytes doubles(std::initializer_list<double> values) { Bytes out; for (auto value : values) word(out, std::bit_cast<std::uint64_t>(value), 8); return out; }
std::vector<Member> members(Bytes pkl, Bytes values = doubles({10, 20, 30, 40})) { return {{"a/data.pkl", std::move(pkl)}, {"a/byteorder", {'l','i','t','t','l','e'}}, {"a/data/0", std::move(values)}}; }
}
int main(int argc, char** argv) {
  try {
    check(argc == 2, "supply a named fixture directory under /opt/mityane-geometrics");
    const auto directory = std::filesystem::path(argv[1]); std::filesystem::create_directories(directory);
    S::ReadPolicy policy; policy.allowed_root = directory; policy.selected_fields = {"x"};
    const auto path = directory / "crafted.pt";
    for (const bool wide : {false, true}) for (const bool descriptor : {false, true}) {
      save(path, zip(members(pickle()), wide, descriptor)); S::Archive archive(path, policy);
      check(archive.fields() == std::vector<std::string>{"x"}, "field schema");
      check(archive.read("x").floating() == std::vector<double>({10, 30, 20, 40}), "noncontiguous logical order");
    }
    save(path, zip(members(pickle()), true, true, true)); check(S::Archive(path, policy).read("x").raw.size() == 32, "ZIP64 locator with non-sentinel classic end values");
    save(path, zip(members(pickle({4}, {-1}, 3)))); check(S::Archive(path, policy).read("x").floating() == std::vector<double>({40, 30, 20, 10}), "signed negative stride");
    save(path, zip(members(pickle({}, {}, 1)))); check(S::Archive(path, policy).read("x").floating() == std::vector<double>({20}), "scalar offset");
    save(path, zip(members(pickle({0}, {1}, 4)))); check(S::Archive(path, policy).read("x").raw.empty(), "empty tensor");
    Bytes fp32; for (const float value : {1.5f,-0.0f,2.25f,4.0f}) word(fp32, std::bit_cast<std::uint32_t>(value), 4);
    save(path, zip(members(pickle({4}, {1}, 0, "FloatStorage"), fp32)));
    check(S::Archive(path, policy).read("x").floating() == std::vector<double>({1.5,0,2.25,4}), "float32 storage decode");
    Bytes longs; for (const std::int64_t value : std::vector<std::int64_t>{INT64_MIN,INT64_MAX,9007199254740993LL,-1}) word(longs, std::bit_cast<std::uint64_t>(value), 8);
    save(path, zip(members(pickle({4}, {1}, 0, "LongStorage"), longs)));
    check(S::Archive(path, policy).read("x").integers() == std::vector<std::int64_t>({INT64_MIN,INT64_MAX,9007199254740993LL,-1}), "signed64 integers never round through double");
    Bytes module{0x80,2}; literal(module, "c__torch__\nModule\n)"); module.push_back(0x81); module.push_back('q'); module.push_back(0);
    module.push_back('}'); unicode(module, "x"); tensor(module, "0", "DoubleStorage", {4}, {1}, 0, 4); module.push_back('s'); module.push_back('b'); module.push_back('.');
    save(path, zip(members(module))); check(S::Archive(path, policy).read("x").floating() == std::vector<double>({10,20,30,40}), "inert Module/BUILD and memo profile");
    auto payload = pickle(); Bytes framed{0x80,4,0x95}; word(framed, payload.size() - 2, 8); framed.insert(framed.end(), payload.begin() + 2, payload.end());
    save(path, zip(members(framed))); check(S::Archive(path, policy).read("x").raw.size() == 32, "protocol4 FRAME");
    framed[3] = 2; for (std::size_t i = 4; i < 11; ++i) framed[i] = 0;
    save(path, zip(members(framed))); rejects([&] { S::Archive archive(path, policy); }, "opcode crossed pickle FRAME");
    save(path, zip(members(pickle({1}, {1}, 0), doubles({-0.0, 0, 0, 0}))));
    check(S::Archive(path, policy).read("x").raw == doubles({-0.0}), "signed zero bytes preserved");
    save(path, zip(members(pickle({4}, {0})))); rejects([&] { S::Archive(path, policy).read("x"); }, "zero-stride alias accepted");
    save(path, zip(members(pickle({4}, {1}, 1)))); rejects([&] { S::Archive(path, policy).read("x"); }, "out-of-storage view accepted");
    save(path, zip(members(pickle({2}, {INT64_MAX}, 1)))); rejects([&] { S::Archive(path, policy).read("x"); }, "signed address overflow accepted");
    save(path, zip(members(pickle({1}, {1}, 0, "DoubleStorage", "cuda:0")))); rejects([&] { S::Archive archive(path, policy); }, "CUDA storage accepted");
    save(path, zip(members(pickle({4}, {1}, 0, "DoubleStorage", "cpu", true)))); rejects([&] { S::Archive(path, policy).read("x"); }, "excluded-field storage alias accepted");
    auto restricted = policy; restricted.selected_fields = {"absent"}; rejects([&] { S::Archive archive(path, restricted); }, "missing selected field accepted");
    save(path, zip(members(pickle()))); S::Archive archive(path, policy); rejects([&] { archive.read("excluded"); }, "unselected field read accepted");
    auto lazy_pickle = pickle(); lazy_pickle.pop_back(); unicode(lazy_pickle, "excluded");
    tensor(lazy_pickle, "1", "DoubleStorage", {1}, {1}, 0, 1); lazy_pickle.push_back('s'); lazy_pickle.push_back('.');
    auto lazy_members = members(lazy_pickle); const auto excluded_bytes = doubles({12345.125}); lazy_members.push_back({"a/data/1", excluded_bytes});
    auto lazy_zip = zip(lazy_members); auto excluded_at = std::search(lazy_zip.begin(), lazy_zip.end(), excluded_bytes.begin(), excluded_bytes.end());
    check(excluded_at != lazy_zip.end(), "excluded fixture location"); *excluded_at ^= 1; save(path, lazy_zip);
    S::Archive lazy(path, policy); check(lazy.read("x").floating() == std::vector<double>({10,30,20,40}), "unselected corrupt member must not be materialized");
    rejects([&] { lazy.read("excluded"); }, "excluded storage admitted");
    auto test_policy = policy; test_policy.role = S::Role::test; rejects([&] { S::Archive archive(path, test_policy); }, "TEST role accepted");
    auto narrow = policy; narrow.allowed_root = directory / "nested"; std::filesystem::create_directories(narrow.allowed_root); rejects([&] { S::Archive archive(path, narrow); }, "root escape accepted");
    auto bad = members(pickle()); bad.push_back({"a/../escape", {}}); save(path, zip(bad)); rejects([&] { S::Archive archive(path, policy); }, "ZIP traversal accepted");
    bad = members(pickle()); bad.push_back(bad.front()); save(path, zip(bad)); rejects([&] { S::Archive archive(path, policy); }, "duplicate ZIP member accepted");
    bad = members(pickle()); bad[1].bytes = {'b','i','g'}; save(path, zip(bad)); rejects([&] { S::Archive archive(path, policy); }, "big-endian storage accepted");
    bad = members(pickle()); bad.erase(bad.begin() + 1); save(path, zip(bad)); rejects([&] { S::Archive archive(path, policy); }, "missing byteorder implicitly accepted");
    auto legacy = policy; legacy.permit_legacy_missing_byteorder = true; check(S::Archive(path, legacy).read("x").floating().size() == 4, "explicit legacy endian profile");
    bad = members(pickle()); bad[0].bytes = {0x80, 2, 'c'}; literal(bad[0].bytes, "os\nsystem\n."); save(path, zip(bad)); rejects([&] { S::Archive archive(path, policy); }, "arbitrary pickle global accepted");
    bad = members(pickle({1}, {1}, 0, "BoolStorage"), {2,0,0,0}); save(path, zip(bad)); rejects([&] { S::Archive(path, policy).read("x"); }, "noncanonical boolean accepted");
    bad = members(pickle({4}, {1}, 0, "ByteStorage"), {0xf0,0x80,0x80,0x80}); save(path, zip(bad)); rejects([&] { S::Archive(path, policy).read("x").utf8(); }, "overlong UTF-8 accepted");
    auto corrupt = zip(members(pickle())); const auto needle = doubles({10,20,30,40});
    auto where = std::search(corrupt.begin(), corrupt.end(), needle.begin(), needle.end()); check(where != corrupt.end(), "fixture storage location"); *where ^= 1;
    save(path, corrupt); rejects([&] { S::Archive(path, policy).read("x"); }, "corrupt storage CRC accepted");
    corrupt = zip(members(pickle()), false, true); const Bytes signature{0x50,0x4b,0x07,0x08};
    auto descriptor = std::search(corrupt.begin(), corrupt.end(), signature.begin(), signature.end()); check(descriptor != corrupt.end(), "descriptor fixture location");
    *(descriptor + 4) ^= 1; save(path, corrupt); rejects([&] { S::Archive archive(path, policy); }, "corrupt data descriptor accepted");
    corrupt = zip(members(pickle())); corrupt[8] = 8; save(path, corrupt); rejects([&] { S::Archive archive(path, policy); }, "compressed ZIP accepted");
    auto tiny = policy; tiny.limits.tensor_elements = 3; save(path, zip(members(pickle()))); rejects([&] { S::Archive(path, tiny).read("x"); }, "element limit ignored");
    std::cout << "saved tensor v1 crafted archive checks passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
