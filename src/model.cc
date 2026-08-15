#include "parallel_phylogenetics/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace parallel_phylogenetics {
namespace {

using Matrix = std::array<double, 16>;

constexpr int kGammaMaximumIterations = 512;
constexpr double kGammaTolerance = 2e-14;

constexpr std::size_t Index(std::size_t row, std::size_t column) {
  return row * 4 + column;
}

void ValidateProbabilityVector(const std::array<double, 4> &values,
                               const char *name, bool require_positive) {
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  // Public callers commonly obtain frequencies from FP32 model files even
  // though construction is performed in double precision.
  constexpr double kProbabilitySumTolerance = 1e-6;
  if (!std::isfinite(sum) || std::abs(sum - 1.0) > kProbabilitySumTolerance ||
      std::any_of(values.begin(), values.end(), [&](double value) {
        return !std::isfinite(value) ||
               (require_positive ? value <= 0.0 : value < 0.0);
      })) {
    throw std::invalid_argument(
        std::string(name) + " must be finite, " +
        (require_positive ? "positive" : "nonnegative") + ", and sum to one");
  }
}

void ValidateModel(const NucleotideModel &model) {
  ValidateProbabilityVector(model.equilibrium_frequencies,
                            "equilibrium frequencies", true);
  ValidateProbabilityVector(model.root_frequencies, "root frequencies", false);
  if (std::any_of(
          model.exchangeabilities.begin(), model.exchangeabilities.end(),
          [](double value) { return !std::isfinite(value) || value < 0.0; })) {
    throw std::invalid_argument(
        "exchangeabilities must be finite and nonnegative");
  }
  if (!(model.substitution_rate >= 0.0) ||
      !std::isfinite(model.substitution_rate)) {
    throw std::invalid_argument(
        "the substitution rate must be finite and nonnegative");
  }
}

Matrix Identity() {
  Matrix result{};
  for (std::size_t state = 0; state < 4; ++state)
    result[Index(state, state)] = 1.0;
  return result;
}

Matrix Multiply(const Matrix &left, const Matrix &right) {
  Matrix result{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < 4; ++inner)
        value += left[Index(row, inner)] * right[Index(inner, column)];
      result[Index(row, column)] = value;
    }
  }
  return result;
}

Matrix UniformizedExponential(const Matrix &generator, double time) {
  if (time == 0.0)
    return Identity();

  double uniformization_rate = 0.0;
  for (std::size_t state = 0; state < 4; ++state) {
    uniformization_rate =
        std::max(uniformization_rate, -generator[Index(state, state)]);
  }
  if (uniformization_rate == 0.0)
    return Identity();

  // Scaling keeps the Poisson mean at most one half. The resulting positive
  // stochastic exponential is then squared back to the requested time. This
  // avoids both short-edge cancellation and underflow for long branches.
  const double full_mean = uniformization_rate * time;
  int squarings = 0;
  double mean = full_mean;
  while (mean > 0.5) {
    mean *= 0.5;
    ++squarings;
  }

  Matrix jump = Identity();
  for (std::size_t index = 0; index < jump.size(); ++index)
    jump[index] += generator[index] / uniformization_rate;

  Matrix power = Identity();
  Matrix result{};
  double weight = std::exp(-mean);
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = weight * power[index];

  constexpr int kMaximumTerms = 256;
  for (int term = 1; term < kMaximumTerms; ++term) {
    power = Multiply(power, jump);
    weight *= mean / static_cast<double>(term);
    for (std::size_t index = 0; index < result.size(); ++index)
      result[index] += weight * power[index];
    if (weight <= 0.25 * std::numeric_limits<double>::epsilon())
      break;
    if (term + 1 == kMaximumTerms)
      throw std::runtime_error(
          "nucleotide matrix exponential did not converge");
  }

  for (int square = 0; square < squarings; ++square)
    result = Multiply(result, result);

  // Roundoff can create tiny negative entries and row-sum drift. Project only
  // values at numerical-noise scale; a material violation is an implementation
  // error and must not be hidden.
  for (std::size_t row = 0; row < 4; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < 4; ++column) {
      double &value = result[Index(row, column)];
      if (value < 0.0) {
        if (value < -1e-12)
          throw std::runtime_error(
              "nucleotide matrix exponential produced a negative probability");
        value = 0.0;
      }
      sum += value;
    }
    if (!(sum > 0.0) || !std::isfinite(sum))
      throw std::runtime_error(
          "nucleotide matrix exponential produced an invalid row");
    for (std::size_t column = 0; column < 4; ++column)
      result[Index(row, column)] /= sum;
  }
  return result;
}

double RegularizedGammaP(double shape, double value) {
  if (!(shape > 0.0) || !std::isfinite(shape) || !(value >= 0.0) ||
      !std::isfinite(value)) {
    throw std::invalid_argument("invalid regularized-gamma arguments");
  }
  if (value == 0.0)
    return 0.0;
  const double log_scale = shape * std::log(value) - value - std::lgamma(shape);
  if (value < shape + 1.0) {
    double term = 1.0 / shape;
    double sum = term;
    double denominator = shape;
    for (int iteration = 1; iteration <= kGammaMaximumIterations; ++iteration) {
      denominator += 1.0;
      term *= value / denominator;
      sum += term;
      if (std::abs(term) <= std::abs(sum) * kGammaTolerance)
        return std::clamp(sum * std::exp(log_scale), 0.0, 1.0);
    }
  } else {
    constexpr double kFloor = std::numeric_limits<double>::min() /
                              std::numeric_limits<double>::epsilon();
    double b = value + 1.0 - shape;
    double c = 1.0 / kFloor;
    double d = 1.0 / b;
    double fraction = d;
    for (int iteration = 1; iteration <= kGammaMaximumIterations; ++iteration) {
      const double index = static_cast<double>(iteration);
      const double coefficient = -index * (index - shape);
      b += 2.0;
      d = coefficient * d + b;
      if (std::abs(d) < kFloor)
        d = kFloor;
      c = b + coefficient / c;
      if (std::abs(c) < kFloor)
        c = kFloor;
      d = 1.0 / d;
      const double delta = d * c;
      fraction *= delta;
      if (std::abs(delta - 1.0) <= kGammaTolerance) {
        const double upper = std::exp(log_scale) * fraction;
        return std::clamp(1.0 - upper, 0.0, 1.0);
      }
    }
  }
  throw std::runtime_error("regularized-gamma evaluation did not converge");
}

double GammaQuantile(double shape, double probability) {
  if (!(probability > 0.0 && probability < 1.0))
    throw std::invalid_argument("gamma quantile probability must lie in (0,1)");
  double lower = 0.0;
  double upper = std::max(1.0, shape);
  while (RegularizedGammaP(shape, upper) < probability) {
    upper *= 2.0;
    if (!std::isfinite(upper))
      throw std::runtime_error("gamma quantile could not be bracketed");
  }
  for (int iteration = 0; iteration < 256; ++iteration) {
    const double midpoint = 0.5 * (lower + upper);
    if (RegularizedGammaP(shape, midpoint) < probability)
      lower = midpoint;
    else
      upper = midpoint;
    if (upper - lower <=
        4.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, upper)) {
      break;
    }
  }
  return 0.5 * (lower + upper);
}

} // namespace

RateMixture::RateMixture(std::vector<double> rates, std::vector<double> weights)
    : rates_(std::move(rates)), weights_(std::move(weights)) {
  ValidateRateMixture(view());
}

RateMixture DiscreteGammaRateMixture(double shape, std::size_t categories) {
  if (!(shape > 0.0) || !std::isfinite(shape))
    throw std::invalid_argument("gamma shape must be finite and positive");
  if (categories == 0)
    throw std::invalid_argument(
        "a gamma mixture requires at least one category");
  std::vector<double> boundaries(categories + 1,
                                 std::numeric_limits<double>::infinity());
  boundaries.front() = 0.0;
  for (std::size_t category = 1; category < categories; ++category) {
    boundaries[category] = GammaQuantile(
        shape, static_cast<double>(category) / static_cast<double>(categories));
  }

  std::vector<double> rates(categories);
  std::vector<double> weights(categories,
                              1.0 / static_cast<double>(categories));
  double previous_moment = 0.0;
  for (std::size_t category = 0; category < categories; ++category) {
    const double next_moment =
        category + 1 == categories
            ? 1.0
            : RegularizedGammaP(shape + 1.0, boundaries[category + 1]);
    rates[category] =
        static_cast<double>(categories) * (next_moment - previous_moment);
    previous_moment = next_moment;
  }
  // Remove the last few ulps of numerical integration drift.
  const double mean =
      std::inner_product(rates.begin(), rates.end(), weights.begin(), 0.0);
  for (double &rate : rates)
    rate /= mean;
  return RateMixture(std::move(rates), std::move(weights));
}

std::size_t RateCategoryCount(RateMixtureView mixture) {
  ValidateRateMixture(mixture);
  return mixture.rates.empty() ? 1 : mixture.rates.size();
}

double RateCategoryRate(RateMixtureView mixture, std::size_t category) {
  const std::size_t count = RateCategoryCount(mixture);
  if (category >= count)
    throw std::out_of_range("rate category is out of range");
  return mixture.rates.empty() ? 1.0 : mixture.rates[category];
}

double RateCategoryWeight(RateMixtureView mixture, std::size_t category) {
  const std::size_t count = RateCategoryCount(mixture);
  if (category >= count)
    throw std::out_of_range("rate category is out of range");
  return mixture.weights.empty() ? 1.0 : mixture.weights[category];
}

void ValidateRateMixture(RateMixtureView mixture) {
  if (mixture.rates.empty() && mixture.weights.empty())
    return;
  if (mixture.rates.empty() || mixture.rates.size() != mixture.weights.size())
    throw std::invalid_argument(
        "rate-mixture rates and weights must be nonempty and equally sized");
  double weight_sum = 0.0;
  for (std::size_t category = 0; category < mixture.rates.size(); ++category) {
    if (!(mixture.rates[category] > 0.0) ||
        !std::isfinite(mixture.rates[category])) {
      throw std::invalid_argument(
          "rate-mixture rates must be finite and positive");
    }
    if (!(mixture.weights[category] > 0.0) ||
        !std::isfinite(mixture.weights[category])) {
      throw std::invalid_argument(
          "rate-mixture weights must be finite and positive");
    }
    weight_sum += mixture.weights[category];
  }
  if (!std::isfinite(weight_sum) || std::abs(weight_sum - 1.0) > 1e-12)
    throw std::invalid_argument("rate-mixture weights must sum to one");
}

NucleotideModel JukesCantorModel(double substitution_rate) {
  NucleotideModel result;
  result.substitution_rate = substitution_rate;
  ValidateModel(result);
  return result;
}

NucleotideModel HasegawaKishinoYanoModel(std::array<double, 4> frequencies,
                                         double kappa,
                                         double substitution_rate) {
  if (!(kappa >= 0.0) || !std::isfinite(kappa))
    throw std::invalid_argument("HKY kappa must be finite and nonnegative");
  // AC, AG, AT, CG, CT, GT: the purine and pyrimidine transitions are AG/CT.
  return GeneralTimeReversibleModel(
      frequencies, {1.0, kappa, 1.0, 1.0, kappa, 1.0}, substitution_rate);
}

NucleotideModel
GeneralTimeReversibleModel(std::array<double, 4> frequencies,
                           std::array<double, 6> exchangeabilities,
                           double substitution_rate) {
  NucleotideModel result{
      .equilibrium_frequencies = frequencies,
      .root_frequencies = frequencies,
      .exchangeabilities = exchangeabilities,
      .substitution_rate = substitution_rate,
  };
  ValidateModel(result);
  static_cast<void>(NormalizedRateMatrix(result));
  return result;
}

std::array<double, 16> NormalizedRateMatrix(const NucleotideModel &model) {
  ValidateModel(model);
  Matrix result{};
  constexpr std::array<std::pair<std::size_t, std::size_t>, 6> pairs{{
      {0, 1},
      {0, 2},
      {0, 3},
      {1, 2},
      {1, 3},
      {2, 3},
  }};
  for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
    const auto [first, second] = pairs[pair];
    const double exchangeability = model.exchangeabilities[pair];
    result[Index(first, second)] =
        exchangeability * model.equilibrium_frequencies[second];
    result[Index(second, first)] =
        exchangeability * model.equilibrium_frequencies[first];
  }
  for (std::size_t row = 0; row < 4; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < 4; ++column) {
      if (column != row)
        sum += result[Index(row, column)];
    }
    result[Index(row, row)] = -sum;
  }
  double expected_rate = 0.0;
  for (std::size_t state = 0; state < 4; ++state) {
    expected_rate -=
        model.equilibrium_frequencies[state] * result[Index(state, state)];
  }
  if (!(expected_rate > 0.0) || !std::isfinite(expected_rate)) {
    throw std::invalid_argument(
        "the nucleotide model must have a positive expected substitution rate");
  }
  for (double &value : result)
    value /= expected_rate;
  return result;
}

std::array<Scalar, 16> NucleotideTransition(const NucleotideModel &model,
                                            double branch_length) {
  ValidateModel(model);
  if (!(branch_length >= 0.0) || !std::isfinite(branch_length)) {
    throw std::invalid_argument(
        "the branch length must be finite and nonnegative");
  }
  Matrix generator = NormalizedRateMatrix(model);
  const double evolutionary_time = model.substitution_rate * branch_length;
  if (!std::isfinite(evolutionary_time)) {
    throw std::overflow_error(
        "the product of substitution rate and branch length must be finite");
  }
  const Matrix transition =
      UniformizedExponential(generator, evolutionary_time);
  std::array<Scalar, 16> result{};
  std::transform(transition.begin(), transition.end(), result.begin(),
                 [](double value) { return static_cast<Scalar>(value); });
  return result;
}

} // namespace parallel_phylogenetics
