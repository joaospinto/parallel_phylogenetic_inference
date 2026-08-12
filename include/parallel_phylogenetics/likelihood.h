#ifndef PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_
#define PARALLEL_PHYLOGENETICS_LIKELIHOOD_H_

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "btrc/plan.h"

namespace parallel_phylogenetics {

enum class Nucleotide : std::int8_t {
  kA = 0,
  kC = 1,
  kG = 2,
  kT = 3,
  kUnknown = -1,
};

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
