#ifndef PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_
#define PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "btrc/plan.h"

namespace parallel_phylogenetics {

// Bit mask of nucleotides compatible with an observation. The ambiguity
// symbols follow the IUPAC nucleotide code.
enum class Nucleotide : std::uint8_t {
  kA = 0b0001,
  kC = 0b0010,
  kG = 0b0100,
  kT = 0b1000,
  kR = 0b0101,
  kY = 0b1010,
  kS = 0b0110,
  kW = 0b1001,
  kK = 0b1100,
  kM = 0b0011,
  kB = 0b1110,
  kD = 0b1101,
  kH = 0b1011,
  kV = 0b0111,
  kUnknown = 0b1111,
};

constexpr bool AllowsState(Nucleotide observation, std::size_t state) {
  return state < 4 && (static_cast<std::uint8_t>(observation) &
                       (std::uint8_t{1} << state)) != 0;
}

struct SiteModelView {
  const btrc::Plan &plan;
  std::span<const double> branch_lengths;
  std::span<const Nucleotide> observations;
  std::array<double, 4> root_frequencies{0.25, 0.25, 0.25, 0.25};
  double substitution_rate = 1.0;
};

struct SitePosterior {
  double likelihood = 0.0;
  // [node, A/C/G/T], including observed tips and ancestral nodes.
  std::vector<double> ancestral_states;
  // [edge, parent nucleotide, child nucleotide].
  std::vector<double> substitutions;
};

std::array<double, 16> JukesCantorTransition(double branch_length,
                                             double rate = 1.0);

double SiteLikelihood(SiteModelView model);
double SiteLogLikelihood(SiteModelView model);
SitePosterior AncestralPosterior(SiteModelView model);

} // namespace parallel_phylogenetics

#endif // PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_
