#include "parallel_phylogenetics/likelihood.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "tree_hmm/inference.h"

namespace parallel_phylogenetics {
namespace {

struct Potentials {
  std::vector<double> nodes;
  std::vector<double> edges;
};

Potentials BuildPotentials(SiteModelView model) {
  if (model.branch_lengths.size() != model.plan.num_edges())
    throw std::invalid_argument("one branch length is required per plan edge");
  if (model.observations.size() != model.plan.num_nodes())
    throw std::invalid_argument("one observation marker is required per node");
  if (!(model.substitution_rate >= 0.0) ||
      !std::isfinite(model.substitution_rate)) {
    throw std::invalid_argument(
        "the substitution rate must be finite and nonnegative");
  }
  const double frequency_sum = std::accumulate(
      model.root_frequencies.begin(), model.root_frequencies.end(), 0.0);
  if (!std::isfinite(frequency_sum) || std::abs(frequency_sum - 1.0) > 1e-12 ||
      std::any_of(model.root_frequencies.begin(), model.root_frequencies.end(),
                  [](double value) { return value < 0.0; })) {
    throw std::invalid_argument(
        "root frequencies must be nonnegative and sum to one");
  }

  Potentials result;
  result.nodes.assign(model.plan.num_nodes() * 4, 1.0);
  std::copy(model.root_frequencies.begin(), model.root_frequencies.end(),
            result.nodes.begin() + model.plan.root() * 4);
  for (std::size_t node = 0; node < model.plan.num_nodes(); ++node) {
    const Nucleotide observation = model.observations[node];
    if (observation == Nucleotide::kUnknown)
      continue;
    const int state = static_cast<int>(observation);
    if (state < 0 || state >= 4)
      throw std::invalid_argument("invalid nucleotide observation");
    double *potential = result.nodes.data() + node * 4;
    for (int candidate = 0; candidate < 4; ++candidate) {
      if (candidate != state)
        potential[candidate] = 0.0;
    }
  }

  result.edges.reserve(model.plan.num_edges() * 16);
  for (const double length : model.branch_lengths) {
    const auto transition =
        JukesCantorTransition(length, model.substitution_rate);
    result.edges.insert(result.edges.end(), transition.begin(),
                        transition.end());
  }
  return result;
}

} // namespace

std::array<double, 16> JukesCantorTransition(double branch_length,
                                             double rate) {
  if (!(branch_length >= 0.0) || !std::isfinite(branch_length) ||
      !(rate >= 0.0) || !std::isfinite(rate)) {
    throw std::invalid_argument(
        "Jukes-Cantor branch length and rate must be finite and nonnegative");
  }
  const double decay = std::exp(-4.0 * rate * branch_length / 3.0);
  const double same = 0.25 + 0.75 * decay;
  const double different = 0.25 - 0.25 * decay;
  std::array<double, 16> result{};
  for (std::size_t parent = 0; parent < 4; ++parent) {
    for (std::size_t child = 0; child < 4; ++child)
      result[parent * 4 + child] = parent == child ? same : different;
  }
  return result;
}

double SiteLikelihood(SiteModelView model) {
  const Potentials potentials = BuildPotentials(model);
  return tree_hmm::PartitionFunction(
      {model.plan, 4, potentials.nodes, potentials.edges});
}

double SiteLogLikelihood(SiteModelView model) {
  const Potentials potentials = BuildPotentials(model);
  return tree_hmm::LogPartitionFunction(
      {model.plan, 4, potentials.nodes, potentials.edges});
}

SitePosterior AncestralPosterior(SiteModelView model) {
  const Potentials potentials = BuildPotentials(model);
  tree_hmm::Marginals marginals = tree_hmm::PosteriorMarginals(
      {model.plan, 4, potentials.nodes, potentials.edges});
  return {
      .likelihood = marginals.partition,
      .ancestral_states = std::move(marginals.nodes),
      .substitutions = std::move(marginals.edges),
  };
}

} // namespace parallel_phylogenetics
