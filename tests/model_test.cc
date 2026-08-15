#include "parallel_phylogenetics/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {

using parallel_phylogenetics::NucleotideModel;
using parallel_phylogenetics::Scalar;

void Check(bool condition, const char *description) {
  if (!condition) {
    std::cerr << "check failed: " << description << '\n';
    std::exit(EXIT_FAILURE);
  }
}

bool Near(double first, double second, double tolerance = 2e-6) {
  return std::abs(first - second) <=
         tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

std::array<double, 16> Multiply(const std::array<Scalar, 16> &left,
                                const std::array<Scalar, 16> &right) {
  std::array<double, 16> result{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      for (std::size_t inner = 0; inner < 4; ++inner) {
        result[row * 4 + column] +=
            static_cast<double>(left[row * 4 + inner]) *
            static_cast<double>(right[inner * 4 + column]);
      }
    }
  }
  return result;
}

void CheckTransition(const NucleotideModel &model, double branch_length) {
  const auto transition =
      parallel_phylogenetics::NucleotideTransition(model, branch_length);
  for (std::size_t parent = 0; parent < 4; ++parent) {
    double sum = 0.0;
    for (std::size_t child = 0; child < 4; ++child) {
      const double value = transition[parent * 4 + child];
      Check(std::isfinite(value) && value >= 0.0,
            "transition probabilities must be finite and nonnegative");
      sum += value;
      Check(Near(model.equilibrium_frequencies[parent] * value,
                 model.equilibrium_frequencies[child] *
                     transition[child * 4 + parent]),
            "transition matrix must satisfy detailed balance");
    }
    Check(Near(sum, 1.0), "transition rows must sum to one");
  }
}

} // namespace

int main() {
  const auto gamma = parallel_phylogenetics::DiscreteGammaRateMixture(0.5, 4);
  const std::array<double, 4> expected_gamma{
      0.0333877533835995,
      0.251915917593438,
      0.820268481973650,
      2.89442784704931,
  };
  Check(gamma.rates().size() == expected_gamma.size(),
        "discrete gamma must return the requested category count");
  for (std::size_t category = 0; category < expected_gamma.size(); ++category) {
    Check(Near(gamma.rates()[category], expected_gamma[category], 2e-7),
          "discrete-gamma conditional means must match the reference");
    Check(Near(gamma.weights()[category], 0.25, 1e-14),
          "discrete-gamma categories must have equal probability");
  }
  for (const double shape : {0.05, 1.0, 20.0}) {
    const auto stress_gamma =
        parallel_phylogenetics::DiscreteGammaRateMixture(shape, 8);
    double mean = 0.0;
    for (std::size_t category = 0; category < stress_gamma.rates().size();
         ++category) {
      Check(std::isfinite(stress_gamma.rates()[category]) &&
                stress_gamma.rates()[category] > 0.0,
            "discrete-Gamma rates must be finite and positive");
      mean += stress_gamma.rates()[category] * stress_gamma.weights()[category];
    }
    Check(Near(mean, 1.0, 1e-12), "discrete-Gamma rates must retain mean one");
  }

  const auto jc = parallel_phylogenetics::JukesCantorModel(1.7);
  const auto jc_transition =
      parallel_phylogenetics::NucleotideTransition(jc, 0.23);
  const double different = -0.25 * std::expm1(-4.0 * 1.7 * 0.23 / 3.0);
  for (std::size_t parent = 0; parent < 4; ++parent) {
    for (std::size_t child = 0; child < 4; ++child) {
      Check(Near(jc_transition[parent * 4 + child],
                 parent == child ? 1.0 - 3.0 * different : different),
            "general transition builder must reproduce JC69");
    }
  }

  const std::array<double, 4> frequencies{0.31, 0.19, 0.27, 0.23};
  const auto hky =
      parallel_phylogenetics::HasegawaKishinoYanoModel(frequencies, 4.2, 0.8);
  const auto gtr = parallel_phylogenetics::GeneralTimeReversibleModel(
      frequencies, {1.2, 3.1, 0.7, 1.8, 4.6, 0.9}, 1.3);
  for (const double length : {0.0, 1e-12, 1e-5, 0.2, 10.0}) {
    CheckTransition(hky, length);
    CheckTransition(gtr, length);
  }

  const auto first = parallel_phylogenetics::NucleotideTransition(gtr, 0.17);
  const auto second = parallel_phylogenetics::NucleotideTransition(gtr, 0.29);
  const auto combined = parallel_phylogenetics::NucleotideTransition(gtr, 0.46);
  const auto composed = Multiply(first, second);
  for (std::size_t index = 0; index < combined.size(); ++index) {
    Check(Near(composed[index], combined[index], 4e-6),
          "transition matrices must obey the Markov semigroup law");
  }
  for (std::size_t child = 0; child < 4; ++child) {
    double stationary = 0.0;
    for (std::size_t parent = 0; parent < 4; ++parent) {
      stationary += frequencies[parent] * combined[parent * 4 + child];
    }
    Check(Near(stationary, frequencies[child], 2e-6),
          "the equilibrium frequencies must be stationary");
  }

  const auto identity = parallel_phylogenetics::NucleotideTransition(gtr, 0.0);
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      Check(identity[row * 4 + column] ==
                static_cast<Scalar>(row == column ? 1.0 : 0.0),
            "zero-length transition must be exact identity");
    }
  }

  const auto short_transition =
      parallel_phylogenetics::NucleotideTransition(gtr, 1e-12);
  Check(std::any_of(short_transition.begin(), short_transition.end(),
                    [](Scalar value) { return value > 0.0 && value < 1e-10; }),
        "short-edge off-diagonal probabilities must survive FP32");

  const auto rate_matrix = parallel_phylogenetics::NormalizedRateMatrix(gtr);
  double expected_rate = 0.0;
  for (std::size_t state = 0; state < 4; ++state)
    expected_rate -= frequencies[state] * rate_matrix[state * 4 + state];
  Check(Near(expected_rate, 1.0, 1e-12),
        "GTR rate matrix must have expected rate one");
  const double derivative_step = std::is_same_v<Scalar, float> ? 1e-3 : 1e-6;
  const auto derivative_transition =
      parallel_phylogenetics::NucleotideTransition(gtr, derivative_step);
  for (std::size_t index = 0; index < rate_matrix.size(); ++index) {
    const double identity_value = index / 4 == index % 4 ? 1.0 : 0.0;
    const double derivative =
        (derivative_transition[index] - identity_value) / derivative_step;
    Check(Near(derivative, gtr.substitution_rate * rate_matrix[index],
               std::is_same_v<Scalar, float> ? 2e-3 : 2e-6),
          "the transition derivative at zero must equal the rate generator");
  }

  bool rejected = false;
  try {
    static_cast<void>(parallel_phylogenetics::GeneralTimeReversibleModel(
        {0.0, 0.2, 0.3, 0.5}, {1, 1, 1, 1, 1, 1}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Check(rejected, "zero stationary frequency must be rejected");

  rejected = false;
  try {
    static_cast<void>(
        parallel_phylogenetics::RateMixture({0.0, 2.0}, {0.6, 0.5}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Check(rejected, "rate-mixture weights not summing to one must be rejected");

  rejected = false;
  try {
    static_cast<void>(
        parallel_phylogenetics::RateMixture({0.0, 1.25}, {0.2, 0.8}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Check(rejected, "zero-rate mixture categories must be rejected");

  rejected = false;
  try {
    static_cast<void>(parallel_phylogenetics::NucleotideTransition(
        parallel_phylogenetics::JukesCantorModel(
            std::numeric_limits<double>::max()),
        2.0));
  } catch (const std::overflow_error &) {
    rejected = true;
  }
  Check(rejected, "overflowing evolutionary time must be rejected");
}
