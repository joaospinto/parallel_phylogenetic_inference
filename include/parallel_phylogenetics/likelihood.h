#ifndef PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_
#define PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "btrc/plan.h"
#include "tree_hmm/scalar.h"

namespace parallel_phylogenetics {

using Scalar = tree_hmm::Scalar;

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
  std::span<const Scalar> branch_lengths;
  std::span<const Nucleotide> observations;
  std::array<Scalar, 4> root_frequencies{0.25, 0.25, 0.25, 0.25};
  Scalar substitution_rate = 1.0;
};

struct SitePosterior {
  Scalar likelihood = 0.0;
  // [node, A/C/G/T], including observed tips and ancestral nodes.
  std::vector<Scalar> ancestral_states;
  // [edge, parent nucleotide, child nucleotide].
  std::vector<Scalar> substitutions;
};

std::array<Scalar, 16> JukesCantorTransition(Scalar branch_length,
                                             Scalar rate = 1.0);

Scalar SiteLikelihood(SiteModelView model);
Scalar SiteLogLikelihood(SiteModelView model);
SitePosterior AncestralPosterior(SiteModelView model);

} // namespace parallel_phylogenetics

#endif // PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_
