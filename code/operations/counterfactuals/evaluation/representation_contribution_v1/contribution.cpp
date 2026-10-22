#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::array<const char*, 3> axes{"polynomial", "periodic", "hinge"};
constexpr std::array<const char*, 5> arms{
    "raw_polynomial", "raw_additive", "fourier", "hinge", "lossy"};
constexpr std::array<int, 6> changes{-1024, -512, 0, 512, 1024, 8192};
constexpr std::size_t dimensions = 4;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::string number(double value) {
  require(std::isfinite(value), "nonfinite numeric output refused");
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17) << value;
  return out.str();
}

std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot read " + path.string());
  std::ostringstream bytes;
  bytes << input.rdbuf();
  require(!input.bad(), "read failure for " + path.string());
  return bytes.str();
}

// The Linux development target implements exclusive-create 'x' mode. Even a
// concurrent or interrupted invocation cannot truncate an earlier artifact.
class NewFile {
  std::FILE* file_ = nullptr;
  std::string path_;
public:
  explicit NewFile(const fs::path& path) : path_(path.string()) {
    file_ = std::fopen(path_.c_str(), "wbx");
    require(file_ != nullptr, "exclusive creation failed: " + path_);
  }
  NewFile(const NewFile&) = delete;
  NewFile& operator=(const NewFile&) = delete;
  ~NewFile() { if (file_) std::fclose(file_); }
  void write(std::string_view bytes) {
    require(file_ != nullptr, "write after close: " + path_);
    require(std::fwrite(bytes.data(), 1, bytes.size(), file_) == bytes.size(),
            "write failure: " + path_);
  }
  void close() {
    require(file_ != nullptr, "duplicate close: " + path_);
    auto* file = std::exchange(file_, nullptr);
    require(std::fclose(file) == 0, "close failure: " + path_);
  }
};

std::vector<std::string> fields(std::string_view line, char delimiter = '\t') {
  std::vector<std::string> result;
  std::size_t begin = 0;
  for (;;) {
    const auto end = line.find(delimiter, begin);
    result.emplace_back(line.substr(begin, end == line.npos ? end : end - begin));
    if (end == line.npos) return result;
    begin = end + 1;
  }
}

std::vector<std::string> lines(const std::string& bytes) {
  require(!bytes.empty() && bytes.back() == '\n', "TSV must end with LF");
  require(bytes.find('\0') == bytes.npos && bytes.find('\r') == bytes.npos,
          "TSV must use canonical LF lines without NUL/CR bytes");
  std::vector<std::string> result;
  for (std::size_t begin = 0; begin < bytes.size();) {
    const auto end = bytes.find('\n', begin);
    require(end != begin, "TSV contains an empty row");
    result.push_back(bytes.substr(begin, end - begin));
    begin = end + 1;
  }
  return result;
}

template <typename Integer>
Integer integer(const std::string& value, const std::string& context) {
  Integer result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
              !value.empty() && std::to_string(result) == value,
          "invalid canonical integer for " + context);
  return result;
}

double real(const std::string& value, const std::string& context) {
  require(!value.empty() && value.find_first_not_of("0123456789+-.eE") == value.npos,
          "invalid decimal for " + context);
  char* end = nullptr;
  errno = 0;
  const double result = std::strtod(value.c_str(), &end);
  require(end == value.c_str() + value.size() && errno != ERANGE && std::isfinite(result),
          "invalid finite decimal for " + context);
  return result;
}

struct Unit {
  std::string id;
  int a_num;
  int b_num;
  int noise_sign;
};

struct Population {
  std::string split;
  std::vector<Unit> units;
};

Population parse_population(const std::string& bytes, bool fitting) {
  Population population;
  const auto input = lines(bytes);
  require(input.front() == "unit\ta_num\tb_num\tnoise_sign", "wrong input TSV header");
  require(input.size() > 1 && input.size() <= 8194, "invalid input population size");
  std::set<int> b_values;
  for (std::size_t row = 1; row < input.size(); ++row) {
    const auto item = fields(input[row]);
    require(item.size() == 4, "input row must have four fields");
    const auto separator = item[0].rfind('-');
    require(separator != item[0].npos, "unit ID lacks split/index");
    const auto split = item[0].substr(0, separator);
    const auto index = integer<std::size_t>(item[0].substr(separator + 1), "unit index");
    require(index == row - 1, "unit IDs must be ordered contiguous zero-based indices");
    require(fitting ? split == "training" : split == "development" || split == "assessment",
            "wrong input split for command");
    if (population.split.empty()) population.split = split;
    require(population.split == split, "mixed input splits refused");
    Unit unit{item[0], integer<int>(item[1], "a_num"), integer<int>(item[2], "b_num"),
              integer<int>(item[3], "noise_sign")};
    require(unit.a_num >= -3072 && unit.a_num <= 3072 &&
                unit.b_num >= -4096 && unit.b_num <= 4096,
            "raw coordinates outside generator bounds");
    require(unit.noise_sign == -1 || unit.noise_sign == 1, "noise_sign must be +/-1");
    require(b_values.insert(unit.b_num).second, "duplicate b_num/source unit refused");
    population.units.push_back(std::move(unit));
  }
  return population;
}

using Features = std::array<double, dimensions>;

Features features(std::size_t arm, int a_num, int b_num) {
  const double a = static_cast<double>(a_num) / 4096;
  const double b = static_cast<double>(b_num) / 4096;
  constexpr double pi = std::numbers::pi_v<double>;
  switch (arm) {
    case 0: return {a, b, a * b, a * a};
    case 1: return {a, b, a * a, b * b};
    case 2: return {a, b, std::sin(pi * a), std::cos(pi * a)};
    case 3: return {a, b, std::abs(a), std::abs(b)};
    case 4: return {b, b * b, std::sin(pi * b), std::cos(pi * b)};
    default: throw std::runtime_error("unknown feature arm");
  }
}

std::string coordinates(const Features& values) {
  std::string result;
  for (double value : values) {
    if (!result.empty()) result += ',';
    result += number(value);
  }
  return result;
}

double reference(std::size_t axis, int a_num, int b_num) {
  const double a = static_cast<double>(a_num) / 4096;
  const double b = static_cast<double>(b_num) / 4096;
  constexpr double pi = std::numbers::pi_v<double>;
  switch (axis) {
    case 0: return 1 + 2 * a - b + 1.5 * a * b + 0.5 * a * a;
    case 1: return 1 + 2 * a - b + std::sin(pi * a) + 0.5 * std::cos(pi * a);
    case 2: return 1 + 2 * a - b + 1.5 * std::abs(a) + 0.5 * std::abs(b);
    default: throw std::runtime_error("unknown response axis");
  }
}

struct Head {
  double intercept = 0;
  Features mean{};
  Features scale{};
  Features coefficient{};
};
using Model = std::array<std::array<Head, arms.size()>, axes.size()>;

using Vector = std::array<double, dimensions + 1>;
using Matrix = std::array<Vector, dimensions + 1>;

Vector solve(Matrix matrix, Vector target) {
  for (std::size_t column = 0; column < target.size(); ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < target.size(); ++row)
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
    require(std::isfinite(matrix[pivot][column]) &&
                std::abs(matrix[pivot][column]) > 1e-12,
            "OLS pivot refusal at column " + std::to_string(column));
    std::swap(matrix[pivot], matrix[column]);
    std::swap(target[pivot], target[column]);
    for (std::size_t row = column + 1; row < target.size(); ++row) {
      const double multiplier = matrix[row][column] / matrix[column][column];
      for (std::size_t j = column; j < target.size(); ++j)
        matrix[row][j] -= multiplier * matrix[column][j];
      target[row] -= multiplier * target[column];
    }
  }
  Vector coefficients{};
  for (std::size_t remaining = target.size(); remaining > 0; --remaining) {
    const auto row = remaining - 1;
    double value = target[row];
    for (std::size_t column = row + 1; column < target.size(); ++column)
      value -= matrix[row][column] * coefficients[column];
    coefficients[row] = value / matrix[row][row];
    require(std::isfinite(coefficients[row]), "nonfinite OLS coefficient refused");
  }
  return coefficients;
}

Model fit(const Population& training) {
  require(training.split == "training", "readout fitting is TRAIN-only");
  require(training.units.size() >= dimensions + 1, "OLS requires at least five training units");
  Model result;
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
      auto& head = result[axis][arm];
      std::vector<Features> raw;
      for (const auto& unit : training.units) {
        raw.push_back(features(arm, unit.a_num, unit.b_num));
        for (std::size_t j = 0; j < dimensions; ++j) head.mean[j] += raw.back()[j];
      }
      const auto count = static_cast<double>(training.units.size());
      for (auto& mean : head.mean) mean /= count;
      for (const auto& row : raw)
        for (std::size_t j = 0; j < dimensions; ++j) {
          const double centered = row[j] - head.mean[j];
          head.scale[j] += centered * centered;
        }
      for (auto& scale : head.scale) {
        scale = std::sqrt(scale / count);
        require(std::isfinite(scale) && scale > 0, "degenerate training feature refused");
      }
      Matrix normal{};
      Vector target{};
      for (std::size_t i = 0; i < raw.size(); ++i) {
        Vector row{1};
        for (std::size_t j = 0; j < dimensions; ++j)
          row[j + 1] = (raw[i][j] - head.mean[j]) / head.scale[j];
        const auto& unit = training.units[i];
        const double label = reference(axis, unit.a_num, unit.b_num) +
                             static_cast<double>(unit.noise_sign) / 32;
        for (std::size_t j = 0; j < row.size(); ++j) {
          target[j] += row[j] * label;
          for (std::size_t k = 0; k < row.size(); ++k) normal[j][k] += row[j] * row[k];
        }
      }
      const auto coefficients = solve(normal, target);
      head.intercept = coefficients.front();
      std::copy(coefficients.begin() + 1, coefficients.end(), head.coefficient.begin());
    }
  }
  return result;
}

std::string model_text(const Model& model) {
  std::string result = "axis\tarm\tfeature\tmean\tscale\tcoefficient\n";
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
      const auto prefix = std::string(axes[axis]) + '\t' + arms[arm] + '\t';
      const auto& head = model[axis][arm];
      result += prefix + "intercept\t0\t1\t" + number(head.intercept) + '\n';
      for (std::size_t feature = 0; feature < dimensions; ++feature)
        result += prefix + std::to_string(feature) + '\t' + number(head.mean[feature]) +
                  '\t' + number(head.scale[feature]) + '\t' +
                  number(head.coefficient[feature]) + '\n';
    }
  }
  return result;
}

Model load_model(const std::string& bytes) {
  const auto input = lines(bytes);
  require(input.front() == "axis\tarm\tfeature\tmean\tscale\tcoefficient",
          "wrong model TSV header");
  std::size_t row = 1;
  Model result;
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
      auto& head = result[axis][arm];
      for (std::size_t item = 0; item <= dimensions; ++item) {
        require(row < input.size(), "truncated model rows");
        const auto cells = fields(input[row++]);
        require(cells.size() == 6 && cells[0] == axes[axis] && cells[1] == arms[arm] &&
                    cells[2] == (item == 0 ? "intercept" : std::to_string(item - 1)),
                "wrong/duplicate/out-of-order model axis, arm or feature");
        const double mean = real(cells[3], "mean");
        const double scale = real(cells[4], "scale");
        const double coefficient = real(cells[5], "coefficient");
        require(scale > 0, "model scale must be positive");
        if (item == 0) {
          require(mean == 0 && scale == 1, "intercept mean/scale must be 0/1");
          head.intercept = coefficient;
        } else {
          head.mean[item - 1] = mean;
          head.scale[item - 1] = scale;
          head.coefficient[item - 1] = coefficient;
        }
      }
    }
  }
  require(row == input.size(), "unexpected model rows");
  return result;
}

double predict(const Head& head, const Features& raw) {
  double response = head.intercept;
  for (std::size_t i = 0; i < dimensions; ++i)
    response += head.coefficient[i] * ((raw[i] - head.mean[i]) / head.scale[i]);
  require(std::isfinite(response), "nonfinite fitted response refused");
  return response;
}

bool supported(int a_num, int b_num) {
  return std::abs(a_num) <= 4096 && std::abs(b_num) <= 4096;
}

std::string evaluate(const Population& population, const Model& model) {
  require(population.split == "development" || population.split == "assessment",
          "evaluation requires development or assessment input");
  std::string result = "axis\tarm\tunit\tchange_num\ta_num\tb_num\tstate\treason\t"
                       "base_prediction\talternative_prediction\tdelta\ttruth\tz_base\tz_alt\n";
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
      for (const auto& unit : population.units) {
        for (const auto change : changes) {
          const auto prefix = std::string(axes[axis]) + '\t' + arms[arm] + '\t' + unit.id +
                              '\t' + std::to_string(change) + '\t' +
                              std::to_string(unit.a_num) + '\t' + std::to_string(unit.b_num) + '\t';
          const auto alternative_a = unit.a_num + change;
          if (!supported(unit.a_num, unit.b_num) || !supported(alternative_a, unit.b_num)) {
            result += prefix + "abstained\tout_of_support\tNA\tNA\tNA\tNA\tNA\tNA\n";
            continue;
          }
          const auto base = features(arm, unit.a_num, unit.b_num);
          const auto alternative = features(arm, alternative_a, unit.b_num);
          const auto base_response = predict(model[axis][arm], base);
          const auto alternative_response = predict(model[axis][arm], alternative);
          // These are deterministic responses of the fitted readout, directly
          // subtracted on the response scale. This is not a causal estimator.
          const auto delta = alternative_response - base_response;
          const auto truth = reference(axis, alternative_a, unit.b_num) -
                             reference(axis, unit.a_num, unit.b_num);
          result += prefix + "succeeded\tNONE\t" + number(base_response) + '\t' +
                    number(alternative_response) + '\t' + number(delta) + '\t' +
                    number(truth) + '\t' + coordinates(base) + '\t' +
                    coordinates(alternative) + '\n';
        }
      }
    }
  }
  return result;
}

void selftest() {
  constexpr double pi = std::numbers::pi_v<double>;
  const std::array<Features, arms.size()> expected{{
      {0.5, -0.25, -0.125, 0.25},
      {0.5, -0.25, 0.25, 0.0625},
      {0.5, -0.25, 1, 0},
      {0.5, -0.25, 0.5, 0.25},
      {-0.25, 0.0625, -std::sqrt(0.5), std::sqrt(0.5)}}};
  for (std::size_t arm = 0; arm < arms.size(); ++arm) {
    const auto actual = features(arm, 2048, -1024);
    for (std::size_t j = 0; j < dimensions; ++j)
      require(std::abs(actual[j] - expected[arm][j]) < 1e-14,
              "selftest known feature value mismatch");
  }
  require(std::abs(reference(0, 2048, -1024) - 2.1875) < 1e-14,
          "selftest polynomial truth mismatch");
  require(std::abs(reference(1, 2048, -1024) - 3.25) < 1e-14,
          "selftest periodic truth mismatch");
  require(std::abs(reference(2, 2048, -1024) - 3.125) < 1e-14,
          "selftest hinge truth mismatch");
  require(std::abs(std::sin(pi)) < 1e-14 && supported(4096, -4096) &&
              !supported(4097, 0), "selftest support boundary mismatch");
  std::string engineering = "unit\ta_num\tb_num\tnoise_sign\n";
  for (int i = 0; i < 96; ++i) {
    const auto a = (i * 997 + 131) % 6145 - 3072;
    const auto b = (i * 79 + 17) % 8193 - 4096;
    engineering += "training-" + std::to_string(i) + '\t' + std::to_string(a) + '\t' +
                   std::to_string(b) + '\t' + std::to_string(i % 2 ? -1 : 1) + '\n';
  }
  const auto model = fit(parse_population(engineering, true));
  const auto text = model_text(model);
  const auto reloaded = load_model(text);
  require(model_text(reloaded) == text, "selftest model round-trip mismatch");
  const auto population = parse_population(
      "unit\ta_num\tb_num\tnoise_sign\n"
      "development-0\t0\t0\t1\n"
      "development-1\t3072\t1\t-1\n", false);
  const auto output = lines(evaluate(population, reloaded));
  require(output.size() == 181, "selftest roster size mismatch");
  std::size_t succeeded = 0;
  std::size_t abstained = 0;
  for (std::size_t row = 1; row < output.size(); ++row) {
    const auto item = fields(output[row]);
    require(item.size() == 14, "selftest pair field count mismatch");
    if (item[6] == "succeeded") {
      ++succeeded;
      require(item[7] == "NONE", "selftest success reason mismatch");
      if (item[3] == "0")
        require(real(item[10], "zero delta") == 0 && real(item[11], "zero truth") == 0,
                "selftest no-change subtraction mismatch");
    } else {
      ++abstained;
      require(item[6] == "abstained" && item[7] == "out_of_support",
              "selftest refusal mismatch");
      for (std::size_t column = 8; column < item.size(); ++column)
        require(item[column] == "NA", "selftest refusal leaked numeric result");
    }
  }
  require(succeeded == 150 && abstained == 30, "selftest coverage mismatch");
  const auto must_refuse = [](const auto& operation, const char* description) {
    bool refused = false;
    try { operation(); } catch (const std::runtime_error&) { refused = true; }
    require(refused, std::string("selftest did not refuse ") + description);
  };
  must_refuse([] {
    parse_population("unit\ta_num\tb_num\tnoise_sign\ntraining-00\t0\t0\t1\n", true);
  }, "noncanonical source ID");
  must_refuse([] {
    parse_population("unit\ta_num\tb_num\tnoise_sign\ntraining-0\t0\t0\t1\n"
                     "training-1\t1\t0\t-1\n", true);
  }, "duplicate source b coordinate");
  must_refuse([] {
    parse_population("unit\ta_num\tb_num\tnoise_sign\ndevelopment-0\t0\t0\t1\n", true);
  }, "evaluation population for fitting");
  must_refuse([] {
    parse_population("unit\ta_num\tb_num\tnoise_sign\nassessment-0\t3073\t0\t1\n", false);
  }, "out-of-generator source coordinate");
  must_refuse([] {
    Matrix singular{};
    Vector target{};
    solve(singular, target);
  }, "singular readout");
  must_refuse([&] { load_model(text + "polynomial\traw_polynomial\t0\t0\t1\t0\n"); },
              "duplicate model row");
  std::cout << "selftest passed: 5 families, 3 axes, 15 TRAIN-only heads, 180 pairs\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::locale::global(std::locale::classic());
    if (argc == 2 && std::string_view(argv[1]) == "selftest") {
      selftest();
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "fit") {
      const auto population = parse_population(read_file(argv[2]), true);
      const auto bytes = model_text(fit(population));
      NewFile output(argv[3]);
      output.write(bytes);
      output.close();
      return 0;
    }
    if (argc == 5 && std::string_view(argv[1]) == "evaluate") {
      const auto population = parse_population(read_file(argv[2]), false);
      const auto model = load_model(read_file(argv[3]));
      const auto bytes = evaluate(population, model);
      NewFile output(argv[4]);
      output.write(bytes);
      output.close();
      return 0;
    }
    throw std::runtime_error("usage: contribution selftest | fit TRAIN.tsv MODEL.tsv | "
                             "evaluate INPUT.tsv MODEL.tsv PAIRS.tsv");
  } catch (const std::exception& error) {
    std::cerr << "refused: " << error.what() << '\n';
    return 1;
  }
}
