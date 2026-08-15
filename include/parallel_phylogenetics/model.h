#ifndef PARALLEL_PHYLOGENETICS_MODEL_H_
#define PARALLEL_PHYLOGENETICS_MODEL_H_

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "tree_hmm/scalar.h"

namespace parallel_phylogenetics {

using Scalar = tree_hmm::Scalar;

// A stationary, time-reversible four-state nucleotide model. Frequencies use
// A/C/G/T order. Exchangeabilities use AC/AG/AT/CG/CT/GT order. The root
// distribution is explicit so a rooted analysis can deliberately differ from
// the stationary distribution; the standard constructors set them equal.
struct NucleotideModel {
  std::array<double, 4> equilibrium_frequencies{0.25, 0.25, 0.25, 0.25};
  std::array<double, 4> root_frequencies{0.25, 0.25, 0.25, 0.25};
  std::array<double, 6> exchangeabilities{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double substitution_rate = 1.0;
};

// A finite mixture of positive branch-rate multipliers. Empty spans denote
// the single-category rate-one model, which keeps ordinary model views small
// and allocation-free.
struct RateMixtureView {
  std::span<const double> rates;
  std::span<const double> weights;
};

class RateMixture {
public:
  RateMixture(std::vector<double> rates, std::vector<double> weights);

  std::span<const double> rates() const { return rates_; }
  std::span<const double> weights() const { return weights_; }
  RateMixtureView view() const { return {rates_, weights_}; }

private:
  std::vector<double> rates_;
  std::vector<double> weights_;
};

// Equal-probability, conditional-mean discretization of a mean-one gamma
// distribution with the requested shape. This is the conventional +Gamma
// construction used for among-site rate heterogeneity.
RateMixture DiscreteGammaRateMixture(double shape, std::size_t categories);

std::size_t RateCategoryCount(RateMixtureView mixture);
double RateCategoryRate(RateMixtureView mixture, std::size_t category);
double RateCategoryWeight(RateMixtureView mixture, std::size_t category);
void ValidateRateMixture(RateMixtureView mixture);

NucleotideModel JukesCantorModel(double substitution_rate = 1.0);

NucleotideModel HasegawaKishinoYanoModel(std::array<double, 4> frequencies,
                                         double kappa,
                                         double substitution_rate = 1.0);

NucleotideModel
GeneralTimeReversibleModel(std::array<double, 4> frequencies,
                           std::array<double, 6> exchangeabilities,
                           double substitution_rate = 1.0);

// Validates all probabilities and rates and returns the expected-rate-one
// infinitesimal generator Q in row-major A/C/G/T order. The model's overall
// substitution rate is deliberately not included.
std::array<double, 16> NormalizedRateMatrix(const NucleotideModel &model);

// Returns exp(Q * substitution_rate * branch_length). Construction is always
// performed in double precision and only then cast to the configured tree-HMM
// scalar type.
std::array<Scalar, 16> NucleotideTransition(const NucleotideModel &model,
                                            double branch_length);

} // namespace parallel_phylogenetics

#endif // PARALLEL_PHYLOGENETICS_MODEL_H_
