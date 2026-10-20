#include "mityane/saved_tensor/v1/reader.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <unordered_set>
#include <utility>

namespace mityane::saved_tensor::v1 {
namespace {
using Bytes = std::vector<std::uint8_t>;
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
              std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "saved tensor v1 requires IEEE binary32/binary64");
void require(bool ok, const std::string& why) { if (!ok) throw DecodeError(why); }
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
  require(a <= UINT64_MAX - b, "unsigned offset overflow"); return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
  require(b == 0 || a <= UINT64_MAX / b, "size overflow"); return a * b;
}
std::int64_t signed_add(std::int64_t a, std::int64_t b) {
  require(!((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)), "signed offset overflow");
  return a + b;
}
std::int64_t signed_mul(std::int64_t a, std::int64_t b) {
  // Indices are nonnegative, strides can be signed.
  require(a >= 0, "negative tensor index");
  if (a == 0) return 0;
  require(b >= 0 ? b <= INT64_MAX / a : b >= INT64_MIN / a, "signed stride overflow");
  return a * b;
}
std::uint64_t le(std::span<const std::uint8_t> b, std::size_t p, std::size_t n) {
  require(n <= 8 && p <= b.size() && n <= b.size() - p, "truncated little-endian number");
  std::uint64_t v = 0; for (std::size_t i = 0; i < n; ++i) v |= std::uint64_t(b[p + i]) << (8 * i);
  return v;
}
std::uint32_t crc32(std::span<const std::uint8_t> b) {
  std::uint32_t crc = UINT32_MAX;
  for (auto x : b) { crc ^= x; for (int i = 0; i < 8; ++i) crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U))); }
  return ~crc;
}
bool utf8_valid(std::span<const std::uint8_t> b) {
  for (std::size_t i = 0; i < b.size();) {
    const auto c = b[i++]; if (c < 0x80) continue;
    int count; std::uint32_t cp, minimum;
    if (c >= 0xc2 && c <= 0xdf) { count = 1; cp = c & 31; minimum = 0x80; }
    else if (c >= 0xe0 && c <= 0xef) { count = 2; cp = c & 15; minimum = 0x800; }
    else if (c >= 0xf0 && c <= 0xf4) { count = 3; cp = c & 7; minimum = 0x10000; }
    else return false;
    if (b.size() - i < static_cast<std::size_t>(count)) return false;
    while (count--) { const auto x = b[i++]; if ((x & 0xc0) != 0x80) return false; cp = (cp << 6) | (x & 63); }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
  }
  return true;
}
void safe_member(const std::string& name) {
  require(!name.empty() && name.front() != '/' && name.find('\\') == std::string::npos &&
          name.find(':') == std::string::npos && name.find('\0') == std::string::npos, "unsafe ZIP member name");
  std::size_t p = 0;
  while (p < name.size()) {
    const auto e = name.find('/', p); const auto part = name.substr(p, e - p);
    require(!part.empty() && part != "." && part != "..", "unsafe ZIP member component");
    if (e == std::string::npos) break;
    p = e + 1;
  }
}
struct Entry { std::uint64_t offset, size; std::uint32_t crc; std::uint16_t flags; };
class Zip {
 public:
  Zip(const std::filesystem::path& path, const Limits& limits) : limits_(limits), file_(path, std::ios::binary) {
    require(bool(file_), "cannot open admitted archive");
    file_.seekg(0, std::ios::end); const auto end = file_.tellg();
    require(end >= 22, "truncated ZIP archive"); size_ = static_cast<std::uint64_t>(end);
    require(size_ <= limits_.archive_bytes, "archive size limit");
    // Seek the usual zero-comment EOCD first, avoiding an unnecessary tail read.
    auto tail = read_at(size_ - 22, 22); std::uint64_t end_pos = size_ - 22;
    if (le(tail, 0, 4) != 0x06054b50 || le(tail, 20, 2) != 0) {
      const auto count = std::min<std::uint64_t>(size_, 65557); tail = read_at(size_ - count, count);
      bool found = false;
      for (std::size_t p = tail.size() - 22 + 1; p-- > 0;) {
        if (le(tail, p, 4) == 0x06054b50 && p + 22 + le(tail, p + 20, 2) == tail.size()) {
          end_pos = size_ - count + p; tail = Bytes(tail.begin() + p, tail.end()); found = true; break;
        }
      }
      require(found, "ZIP EOCD not found");
    }
    require(le(tail, 4, 2) == 0 && le(tail, 6, 2) == 0 && le(tail, 8, 2) == le(tail, 10, 2), "multi-disk ZIP unsupported");
    auto count = le(tail, 10, 2), directory_size = le(tail, 12, 4), directory = le(tail, 16, 4);
    auto records_end = end_pos;
    const bool needs_zip64 = count == 65535 || directory_size == UINT32_MAX || directory == UINT32_MAX;
    const auto locator = end_pos >= 20 ? read_at(end_pos - 20, 20) : Bytes{};
    const bool has_zip64 = locator.size() == 20 && le(locator, 0, 4) == 0x07064b50;
    require(!needs_zip64 || has_zip64, "missing ZIP64 locator");
    if (has_zip64) {
      require(le(locator, 0, 4) == 0x07064b50 && le(locator, 4, 4) == 0 && le(locator, 16, 4) == 1, "invalid ZIP64 locator");
      const auto zpos = le(locator, 8, 8); const auto z = read_at(zpos, 56);
      require(le(z, 0, 4) == 0x06064b50 && le(z, 4, 8) == 44 && add(zpos, 56) == end_pos - 20, "unsupported ZIP64 end record");
      require(le(z, 16, 4) == 0 && le(z, 20, 4) == 0 && le(z, 24, 8) == le(z, 32, 8), "ZIP64 multi-disk unsupported");
      require((count == 65535 || count == le(z, 32, 8)) &&
              (directory_size == UINT32_MAX || directory_size == le(z, 40, 8)) &&
              (directory == UINT32_MAX || directory == le(z, 48, 8)), "ZIP32/ZIP64 end records disagree");
      count = le(z, 32, 8); directory_size = le(z, 40, 8); directory = le(z, 48, 8); records_end = zpos;
    }
    require(count <= limits_.members && directory_size <= limits_.member_bytes && add(directory, directory_size) == records_end, "invalid ZIP directory bounds");
    directory_ = directory; const auto bytes = read_at(directory, directory_size); std::size_t p = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (std::uint64_t i = 0; i < count; ++i) {
      require(le(bytes, p, 4) == 0x02014b50, "invalid central directory signature");
      const auto flags = le(bytes, p + 8, 2), method = le(bytes, p + 10, 2);
      require((flags & ~std::uint64_t(0x808)) == 0 && method == 0, "only unencrypted ZIP_STORED members supported");
      auto packed = le(bytes, p + 20, 4), unpacked = le(bytes, p + 24, 4), offset = le(bytes, p + 42, 4);
      const bool central_zip64_sizes = packed == UINT32_MAX || unpacked == UINT32_MAX;
      const auto names = le(bytes, p + 28, 2), extra = le(bytes, p + 30, 2), comment = le(bytes, p + 32, 2);
      require(le(bytes, p + 34, 2) == 0, "multi-disk member unsupported");
      const auto next = add(p, add(46, add(names, add(extra, comment)))); require(next <= bytes.size(), "truncated central directory entry");
      const std::string name(bytes.begin() + p + 46, bytes.begin() + p + 46 + names); safe_member(name);
      bool zip64 = false;
      for (std::size_t q = p + 46 + names; q < p + 46 + names + extra;) {
        const auto tag = le(bytes, q, 2), n = le(bytes, q + 2, 2); q += 4;
        require(q + n <= p + 46 + names + extra, "truncated ZIP extra field");
        if (tag == 1) {
          require(!zip64, "duplicate ZIP64 extra field"); zip64 = true; std::size_t r = q;
          auto take = [&]() { require(r + 8 <= q + n, "truncated ZIP64 member size"); const auto v = le(bytes, r, 8); r += 8; return v; };
          if (unpacked == UINT32_MAX) unpacked = take();
          if (packed == UINT32_MAX) packed = take();
          if (offset == UINT32_MAX) offset = take();
          require(r == q + n, "unsupported ZIP64 member extension");
        }
        q += n;
      }
      require(packed == unpacked && unpacked <= limits_.member_bytes && offset < directory, "invalid stored ZIP member size");
      const auto local = read_at(offset, 30);
      require(le(local, 0, 4) == 0x04034b50 && le(local, 6, 2) == flags && le(local, 8, 2) == 0, "local ZIP header mismatch");
      const auto local_names = le(local, 26, 2), local_extra = le(local, 28, 2);
      const auto local_name = read_at(add(offset, 30), local_names);
      require(std::string(local_name.begin(), local_name.end()) == name, "local ZIP name mismatch");
      const auto payload = add(offset, add(30, add(local_names, local_extra)));
      const auto finish = add(payload, unpacked); require(finish <= directory, "ZIP member overlaps directory");
      const auto expected_crc = static_cast<std::uint32_t>(le(bytes, p + 16, 4));
      auto extent = finish;
      if (!(flags & 8)) {
        auto local_packed = le(local, 18, 4), local_unpacked = le(local, 22, 4);
        if (local_packed == UINT32_MAX || local_unpacked == UINT32_MAX) {
          const auto extras = read_at(add(offset, add(30, local_names)), local_extra); bool found = false;
          for (std::size_t q = 0; q < extras.size();) {
            const auto tag = le(extras, q, 2), n = le(extras, q + 2, 2); q += 4;
            require(q + n <= extras.size(), "truncated local ZIP extra field");
            if (tag == 1) {
              require(!found, "duplicate local ZIP64 sizes"); found = true; std::size_t r = q;
              auto take = [&]() { require(r + 8 <= q + n, "truncated local ZIP64 size"); const auto v = le(extras, r, 8); r += 8; return v; };
              if (local_unpacked == UINT32_MAX) local_unpacked = take();
              if (local_packed == UINT32_MAX) local_packed = take();
              require(r == q + n, "unsupported local ZIP64 extension");
            }
            q += n;
          }
          require(found, "missing local ZIP64 sizes");
        }
        require(le(local, 14, 4) == expected_crc && local_packed == packed && local_unpacked == unpacked, "local ZIP sizes/CRC mismatch");
      } else {
        require(add(finish, 4) <= directory, "missing ZIP data descriptor");
        const auto first = read_at(finish, 4); const bool signature = le(first, 0, 4) == 0x08074b50;
        const auto start = add(finish, signature ? 4 : 0);
        const bool wide = central_zip64_sizes || le(local, 18, 4) == UINT32_MAX || le(local, 22, 4) == UINT32_MAX;
        const std::size_t word = wide ? 8 : 4; extent = add(start, 4 + word * 2);
        require(extent <= directory, "truncated ZIP data descriptor"); const auto descriptor = read_at(start, 4 + word * 2);
        require(le(descriptor, 0, 4) == expected_crc && le(descriptor, 4, word) == packed && le(descriptor, 4 + word, word) == unpacked,
                "ZIP data descriptor mismatch");
      }
      require(entries_.emplace(name, Entry{payload, unpacked, expected_crc, static_cast<std::uint16_t>(flags)}).second, "duplicate ZIP member");
      ranges.emplace_back(offset, extent); p = static_cast<std::size_t>(next);
    }
    require(p == bytes.size(), "trailing central directory data"); std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i) require(ranges[i - 1].second <= ranges[i].first, "overlapping ZIP members");
  }
  Bytes member(const std::string& name, std::uint64_t limit) const {
    const auto it = entries_.find(name); require(it != entries_.end(), "missing ZIP member: " + name);
    require(it->second.size <= limit, "member read limit"); auto bytes = read_at(it->second.offset, it->second.size);
    require(crc32(bytes) == it->second.crc, "ZIP CRC mismatch: " + name); return bytes;
  }
  const std::map<std::string, Entry>& entries() const { return entries_; }
 private:
  Bytes read_at(std::uint64_t p, std::uint64_t n) const {
    require(p <= size_ && n <= size_ - p && n <= limits_.member_bytes && n <= std::numeric_limits<std::size_t>::max(), "ZIP read bounds");
    Bytes b(static_cast<std::size_t>(n)); file_.clear(); file_.seekg(static_cast<std::streamoff>(p));
    if (n) file_.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
    require(bool(file_), "truncated ZIP read"); return b;
  }
  Limits limits_; mutable std::ifstream file_; std::uint64_t size_ = 0, directory_ = 0;
  std::map<std::string, Entry> entries_;
};

struct Node;
using N = std::shared_ptr<Node>;
struct Node {
  enum Kind { none, integer, boolean, text, sequence, mapping, symbol, storage, tensor, module, mark } kind;
  std::int64_t number = 0; std::string string; std::vector<N> items; std::map<std::string, N> fields; TensorInfo info;
  explicit Node(Kind k) : kind(k) {}
};
class Pickle {
 public:
  Pickle(Bytes bytes, const Limits& limits) : bytes_(std::move(bytes)), limits_(limits) {}
  ~Pickle() {
    // Keep every allocated node alive until all edges are cleared. Malformed
    // cyclic/deep graphs must not leak or recursively overflow on rejection.
    for (const auto& node : allocated_) { node->items.clear(); node->fields.clear(); }
  }
  N parse() {
    require(byte() == 0x80, "pickle must declare protocol"); const auto protocol = byte(); require(protocol >= 2 && protocol <= 5, "unsupported pickle protocol");
    for (std::uint64_t operations = 0;; ++operations) {
      require(operations < limits_.opcodes && stack_.size() <= limits_.nodes, "pickle resource limit");
      if (frame_end_ && pos_ == frame_end_) frame_end_ = 0;
      const auto op = byte();
      switch (op) {
        case '.': require(stack_.size() == 1 && pos_ == bytes_.size(), "pickle trailing state/bytes"); return stack_.back();
        case '(': stack_.push_back(make(Node::mark)); break;
        case 'N': stack_.push_back(make(Node::none)); break;
        case 0x88: case 0x89: { auto n = make(Node::boolean); n->number = op == 0x88; stack_.push_back(n); break; }
        case 'K': integer(byte()); break;
        case 'M': integer(static_cast<std::int64_t>(number(2))); break;
        case 'J': integer(std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(number(4)))); break;
        case 0x8a: long_integer(byte()); break;
        case 0x8b: long_integer(number(4)); break;
        case 'X': string(number(4)); break;
        case 0x8c: string(byte()); break;
        case 0x8d: string(number(8)); break;
        case ')': stack_.push_back(make(Node::sequence)); break;
        case ']': stack_.push_back(make(Node::sequence)); break;
        case '}': stack_.push_back(make(Node::mapping)); break;
        case 't': tuple_from_mark(); break;
        case 0x85: tuple(1); break; case 0x86: tuple(2); break; case 0x87: tuple(3); break;
        case 'l': tuple_from_mark(); break;
        case 'd': { const auto values = marked(); auto n = make(Node::mapping); set_pairs(n, values); stack_.push_back(n); break; }
        case 'a': { const auto value = pop(); auto n = top(); require(n->kind == Node::sequence && n != value, "invalid APPEND"); n->items.push_back(value); break; }
        case 'e': { const auto values = marked(); auto n = top(); require(n->kind == Node::sequence, "invalid APPENDS"); for (const auto& v : values) { require(n != v, "cyclic pickle container"); n->items.push_back(v); } break; }
        case 's': { const auto value = pop(), key = pop(); set_pairs(top(), {key, value}); break; }
        case 'u': { const auto values = marked(); set_pairs(top(), values); break; }
        case 'q': memo_put(number(1)); break; case 'r': memo_put(number(4)); break;
        case 'h': memo_get(number(1)); break; case 'j': memo_get(number(4)); break;
        case 0x94: memo_put(memo_.size()); break;
        case 'c': { const auto module = line(), name = line(); symbol(module + "." + name); break; }
        case 0x93: { const auto name = pop(), module = pop(); require(name->kind == Node::text && module->kind == Node::text, "invalid STACK_GLOBAL"); symbol(module->string + "." + name->string); break; }
        case 'Q': persistent(); break;
        case 'R': reduce(); break;
        case 0x81: { const auto args = pop(), cls = pop(); require(is_module(cls) && args->kind == Node::sequence && args->items.empty(), "NEWOBJ outside empty Module"); stack_.push_back(make(Node::module)); break; }
        case 'b': { const auto state = pop(), object = top(); require(object->kind == Node::module && state->kind == Node::mapping && object->fields.empty(), "BUILD outside Module state"); object->fields = state->fields; break; }
        case 0x95: {
          require(protocol >= 4 && !frame_end_, "nested/invalid pickle FRAME"); const auto n = number(8);
          require(n <= bytes_.size() - pos_, "truncated pickle FRAME"); frame_end_ = pos_ + n; break;
        }
        default: throw DecodeError("pickle opcode outside closed saved-tensor subset: " + std::to_string(op));
      }
    }
  }
 private:
  N make(Node::Kind kind) {
    require(++nodes_ <= limits_.nodes, "pickle node limit"); auto node = std::make_shared<Node>(kind);
    allocated_.push_back(node); return node;
  }
  std::uint8_t byte() {
    require(pos_ < bytes_.size() && (!frame_end_ || pos_ < frame_end_), "truncated pickle/frame"); return bytes_[pos_++];
  }
  std::uint64_t number(std::size_t n) {
    require(!frame_end_ || (pos_ <= frame_end_ && n <= frame_end_ - pos_), "pickle integer crosses FRAME");
    const auto v = le(bytes_, pos_, n); pos_ += n; return v;
  }
  N top() { require(!stack_.empty(), "pickle stack underflow"); return stack_.back(); }
  N pop() { auto n = top(); stack_.pop_back(); return n; }
  void integer(std::int64_t v) { auto n = make(Node::integer); n->number = v; stack_.push_back(n); }
  void long_integer(std::uint64_t length) {
    require(length <= 8 && length <= bytes_.size() - pos_, "pickle integer outside signed64");
    if (!length) { integer(0); return; }
    auto value = number(static_cast<std::size_t>(length));
    if (length < 8 && (value & (1ULL << (length * 8 - 1)))) value |= UINT64_MAX << (length * 8);
    integer(std::bit_cast<std::int64_t>(value));
  }
  void string(std::uint64_t n) {
    require(n <= bytes_.size() - pos_ && (!frame_end_ || (pos_ <= frame_end_ && n <= frame_end_ - pos_)), "truncated pickle/frame string"); auto value = make(Node::text);
    const std::span<const std::uint8_t> part(bytes_.data() + pos_, static_cast<std::size_t>(n)); require(utf8_valid(part), "invalid pickle UTF-8");
    value->string.assign(reinterpret_cast<const char*>(part.data()), part.size()); pos_ += n; stack_.push_back(value);
  }
  std::string line() { const auto begin = pos_; while (byte() != '\n') {} return std::string(bytes_.begin() + begin, bytes_.begin() + pos_ - 1); }
  std::vector<N> marked() {
    std::size_t start = stack_.size(); while (start && stack_[start - 1]->kind != Node::mark) --start;
    require(start > 0, "missing pickle MARK"); std::vector<N> values(stack_.begin() + start, stack_.end()); stack_.resize(start - 1); return values;
  }
  void tuple_from_mark() { auto n = make(Node::sequence); n->items = marked(); stack_.push_back(n); }
  void tuple(std::size_t count) {
    require(stack_.size() >= count, "tuple stack underflow"); auto n = make(Node::sequence);
    n->items.assign(stack_.end() - count, stack_.end()); stack_.resize(stack_.size() - count); stack_.push_back(n);
  }
  void set_pairs(const N& n, const std::vector<N>& pairs) {
    require(n->kind == Node::mapping && pairs.size() % 2 == 0, "invalid pickle mapping");
    for (std::size_t i = 0; i < pairs.size(); i += 2) {
      require(pairs[i]->kind == Node::text && n != pairs[i + 1], "invalid/cyclic pickle mapping key/value");
      require(n->fields.emplace(pairs[i]->string, pairs[i + 1]).second, "duplicate pickle mapping key");
    }
  }
  void memo_put(std::uint64_t key) { require(key < limits_.nodes && memo_.emplace(key, top()).second, "duplicate/excess pickle memo"); }
  void memo_get(std::uint64_t key) { const auto it = memo_.find(key); require(it != memo_.end(), "undefined pickle memo"); stack_.push_back(it->second); }
  static bool is_module(const N& n) { return n->kind == Node::symbol && n->string.starts_with("__torch__.") && n->string.ends_with(".Module"); }
  void symbol(const std::string& name) {
    const bool storage = name == "torch.DoubleStorage" || name == "torch.FloatStorage" || name == "torch.LongStorage" || name == "torch.BoolStorage" || name == "torch.ByteStorage";
    auto n = make(Node::symbol); n->string = name;
    require(storage || name == "collections.OrderedDict" || name == "torch._utils._rebuild_tensor" || name == "torch._utils._rebuild_tensor_v2" || is_module(n), "disallowed pickle global: " + name);
    stack_.push_back(n);
  }
  static std::int64_t integer_of(const N& n) { require(n->kind == Node::integer, "expected signed64 pickle integer"); return n->number; }
  static std::vector<std::int64_t> dimensions(const N& n) {
    require(n->kind == Node::sequence, "expected shape/stride tuple"); std::vector<std::int64_t> out;
    for (const auto& v : n->items) out.push_back(integer_of(v));
    return out;
  }
  void persistent() {
    const auto pid = pop(); require(pid->kind == Node::sequence && pid->items.size() == 5, "storage PID must have exactly five fields"); const auto& p = pid->items;
    require(p[0]->kind == Node::text && p[0]->string == "storage" && p[1]->kind == Node::symbol && p[2]->kind == Node::text && p[3]->kind == Node::text && p[3]->string == "cpu", "non-CPU/invalid storage PID");
    auto n = make(Node::storage); const auto& s = p[1]->string;
    if (s == "torch.DoubleStorage") n->info.dtype = DType::float64;
    else if (s == "torch.FloatStorage") n->info.dtype = DType::float32;
    else if (s == "torch.LongStorage") n->info.dtype = DType::int64;
    else if (s == "torch.BoolStorage") n->info.dtype = DType::boolean;
    else if (s == "torch.ByteStorage") n->info.dtype = DType::byte;
    else throw DecodeError("unsupported persistent storage dtype");
    n->info.storage_key = p[2]->string; require(!n->info.storage_key.empty() && n->info.storage_key.find_first_not_of("0123456789") == std::string::npos, "storage key must be decimal digits");
    n->info.storage_elements = integer_of(p[4]); require(n->info.storage_elements >= 0, "negative storage element count"); stack_.push_back(n);
  }
  void reduce() {
    const auto args = pop(), callable = pop(); require(callable->kind == Node::symbol && args->kind == Node::sequence, "invalid REDUCE"); const auto& a = args->items;
    if (callable->string == "collections.OrderedDict") { require(a.empty(), "OrderedDict construction arguments unsupported"); stack_.push_back(make(Node::mapping)); return; }
    const bool v2 = callable->string == "torch._utils._rebuild_tensor_v2";
    require(v2 || callable->string == "torch._utils._rebuild_tensor", "REDUCE outside tensor/OrderedDict");
    require(v2 ? (a.size() == 6 || a.size() == 7) : a.size() == 4, "tensor rebuild argument count");
    require(a[0]->kind == Node::storage, "tensor missing storage PID");
    if (v2) {
      require(a[4]->kind == Node::boolean && !a[4]->number && a[5]->kind == Node::mapping && a[5]->fields.empty(), "requires-grad/hooks unsupported");
      if (a.size() == 7) require(a[6]->kind == Node::none || (a[6]->kind == Node::mapping && a[6]->fields.empty()), "tensor metadata unsupported");
    }
    auto n = make(Node::tensor); n->info = a[0]->info; n->info.storage_offset = integer_of(a[1]);
    n->info.shape = dimensions(a[2]); n->info.strides = dimensions(a[3]); require(n->info.shape.size() == n->info.strides.size() && n->info.shape.size() <= limits_.depth, "tensor shape/stride rank");
    for (const auto d : n->info.shape) require(d >= 0, "negative tensor dimension");
    stack_.push_back(n);
  }
  Bytes bytes_; Limits limits_; std::size_t pos_ = 0, frame_end_ = 0; std::uint64_t nodes_ = 0;
  std::vector<N> stack_, allocated_; std::map<std::uint64_t, N> memo_;
};

std::filesystem::path admitted_path(const std::filesystem::path& path, const ReadPolicy& policy) {
  require(policy.role == Role::training || policy.role == Role::validation || policy.role == Role::engineering_fixture,
          "TEST/unknown artifact role is forbidden by saved-tensor v1");
  require(!policy.allowed_root.empty() && !policy.selected_fields.empty(), "explicit allowed root and selected fields required");
  const auto root = std::filesystem::canonical(policy.allowed_root), resolved = std::filesystem::canonical(path);
  auto r = root.begin(), p = resolved.begin(); for (; r != root.end(); ++r, ++p) require(p != resolved.end() && *r == *p, "archive escapes admitted root");
  require(std::filesystem::is_regular_file(resolved), "archive is not a regular file"); return resolved;
}
}  // namespace

std::size_t element_bytes(DType dtype) {
  switch (dtype) { case DType::float64: case DType::int64: return 8; case DType::float32: return 4; case DType::boolean: case DType::byte: return 1; }
  throw DecodeError("invalid dtype");
}
struct Archive::Impl {
  ReadPolicy policy; Zip zip; std::string prefix; std::map<std::string, TensorInfo> tensors; std::map<std::string, std::size_t> uses;
  Impl(const std::filesystem::path& path, ReadPolicy p) : policy(std::move(p)), zip(admitted_path(path, policy), policy.limits) {
    for (const auto& [name, entry] : zip.entries()) { (void)entry; if (name.ends_with("/data.pkl")) { require(prefix.empty(), "multiple pickle roots"); prefix = name.substr(0, name.size() - 8); } }
    require(!prefix.empty(), "missing data.pkl root");
    for (const auto& [name, entry] : zip.entries()) { (void)entry; require(name.starts_with(prefix), "ZIP members outside single archive root"); }
    if (zip.entries().contains(prefix + "byteorder")) { const auto bytes = zip.member(prefix + "byteorder", 16); require(std::string(bytes.begin(), bytes.end()) == "little", "only little-endian storage supported"); }
    else require(policy.permit_legacy_missing_byteorder, "missing byteorder requires explicit historical profile");
    Pickle parser(zip.member(prefix + "data.pkl", policy.limits.pickle_bytes), policy.limits);
    auto root = parser.parse();
    require(root->kind == Node::module || root->kind == Node::mapping, "archive root must be Module/dictionary");
    std::unordered_set<const Node*> active;
    auto visit = [&](auto&& self, const N& node, const std::string& name, std::size_t depth) -> void {
      require(depth <= policy.limits.depth && active.insert(node.get()).second, "cyclic/deep pickle state");
      if (node->kind == Node::tensor) { require(tensors.emplace(name, node->info).second, "duplicate flattened tensor field"); ++uses[node->info.storage_key]; }
      else if (node->kind == Node::module || node->kind == Node::mapping) {
        for (const auto& [key, child] : node->fields) {
          require(!key.empty() && key.find('.') == std::string::npos, "ambiguous flattened field name");
          self(self, child, name.empty() ? key : name + "." + key, depth + 1);
        }
      } else require(node->kind == Node::none || node->kind == Node::integer || node->kind == Node::boolean || node->kind == Node::text, "non-tensor container state unsupported");
      active.erase(node.get());
    };
    visit(visit, root, "", 0);
    for (const auto& field : policy.selected_fields) require(tensors.contains(field), "selected field missing: " + field);
  }
};
Archive::Archive(const std::filesystem::path& path, ReadPolicy policy) : impl_(std::make_unique<Impl>(path, std::move(policy))) {}
Archive::~Archive() = default;
Archive::Archive(Archive&&) noexcept = default;
Archive& Archive::operator=(Archive&&) noexcept = default;
std::vector<std::string> Archive::fields() const { std::vector<std::string> out; for (const auto& [key, value] : impl_->tensors) { (void)value; out.push_back(key); } return out; }
const TensorInfo& Archive::info(const std::string& field) const { const auto it = impl_->tensors.find(field); require(it != impl_->tensors.end(), "tensor field not found"); return it->second; }
Tensor Archive::read(const std::string& field) const {
  require(impl_->policy.selected_fields.contains(field), "tensor field not admitted for reading"); const auto& t = info(field);
  require(impl_->uses.at(t.storage_key) == 1, "shared/aliased storage requires a future explicit protocol");
  const auto width = element_bytes(t.dtype);
  std::uint64_t count = std::find(t.shape.begin(), t.shape.end(), 0) == t.shape.end() ? 1 : 0;
  if (count) for (const auto n : t.shape) count = mul(count, static_cast<std::uint64_t>(n));
  require(count <= impl_->policy.limits.tensor_elements, "tensor element limit");
  const auto expected = mul(static_cast<std::uint64_t>(t.storage_elements), width);
  const auto entry = impl_->zip.entries().find(impl_->prefix + "data/" + t.storage_key);
  require(entry != impl_->zip.entries().end() && entry->second.size == expected, "storage PID size mismatch");
  std::int64_t low = t.storage_offset, high = t.storage_offset;
  for (std::size_t d = 0; count && d < t.shape.size(); ++d) if (t.shape[d]) {
    const auto span = signed_mul(t.shape[d] - 1, t.strides[d]); if (span < 0) low = signed_add(low, span); else high = signed_add(high, span);
  }
  require(low >= 0 && (count ? high < t.storage_elements : t.storage_offset <= t.storage_elements), "tensor view outside storage");
  const auto storage = impl_->zip.member(entry->first, impl_->policy.limits.member_bytes);
  Tensor out{t.dtype, t.shape, Bytes(static_cast<std::size_t>(mul(count, width)))};
  std::vector<std::int64_t> indices(t.shape.size(), 0); std::unordered_set<std::int64_t> visited;
  for (std::uint64_t i = 0; i < count; ++i) {
    auto address = t.storage_offset; for (std::size_t d = 0; d < indices.size(); ++d) address = signed_add(address, signed_mul(indices[d], t.strides[d]));
    require(visited.insert(address).second, "overlapping/zero-stride tensor view forbidden");
    const auto source = static_cast<std::size_t>(mul(static_cast<std::uint64_t>(address), width));
    std::copy_n(storage.begin() + source, width, out.raw.begin() + static_cast<std::size_t>(i * width));
    for (std::size_t d = indices.size(); d-- > 0;) { if (++indices[d] < t.shape[d]) break; indices[d] = 0; }
  }
  if (t.dtype == DType::boolean) for (const auto v : out.raw) require(v <= 1, "noncanonical boolean tensor");
  return out;
}
std::vector<double> Tensor::floating() const {
  require(dtype == DType::float64 || dtype == DType::float32, "floating access requires Float/Double storage");
  const auto width = element_bytes(dtype); require(raw.size() % width == 0, "truncated floating bytes"); std::vector<double> out;
  for (std::size_t i = 0; i < raw.size(); i += width) out.push_back(dtype == DType::float64 ? std::bit_cast<double>(le(raw, i, 8)) : double(std::bit_cast<float>(static_cast<std::uint32_t>(le(raw, i, 4)))));
  return out;
}
std::vector<std::int64_t> Tensor::integers() const {
  require(dtype == DType::int64 || dtype == DType::boolean || dtype == DType::byte, "integer access requires Long/Bool/Byte storage");
  const auto width = element_bytes(dtype); require(raw.size() % width == 0, "truncated integer bytes"); std::vector<std::int64_t> out;
  for (std::size_t i = 0; i < raw.size(); i += width) { const auto value = le(raw, i, width); require(dtype != DType::boolean || value <= 1, "noncanonical bool"); out.push_back(width == 8 ? std::bit_cast<std::int64_t>(value) : static_cast<std::int64_t>(value)); }
  return out;
}
std::string Tensor::utf8() const {
  require(dtype == DType::byte && shape.size() == 1 && shape[0] >= 0 && static_cast<std::uint64_t>(shape[0]) == raw.size(), "text must be rank-one Byte tensor");
  require(utf8_valid(raw), "text tensor is not valid UTF-8"); return std::string(raw.begin(), raw.end());
}
}  // namespace mityane::saved_tensor::v1
