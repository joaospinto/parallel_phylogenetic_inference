#include "parallel_phylogenetics/likelihood.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "tree_hmm/inference.h"

namespace parallel_phylogenetics {
namespace {

struct Potentials {
  std::vector<Scalar> nodes;
  std::vector<Scalar> edges;
};

Potentials BuildPotentials(SiteModelView model) {
  if (model.branch_lengths.size() != model.plan.num_edges())
    throw std::invalid_argument("one branch length is required per plan edge");
  if (model.observations.size() != model.plan.num_nodes())
    throw std::invalid_argument("one observation marker is required per node");
  ValidateRateMixture(model.rate_mixture);
  if (RateCategoryCount(model.rate_mixture) != 1)
    throw std::invalid_argument(
        "site factor construction requires one selected rate category");
  // NucleotideTransition validates the complete model. Construct the first
  // matrix eagerly so an empty tree still receives the same validation.
  if (model.plan.num_edges() == 0)
    static_cast<void>(NucleotideTransition(model.nucleotide_model, 0.0));

  Potentials result;
  result.nodes.assign(model.plan.num_nodes() * 4, 1.0);
  std::transform(model.nucleotide_model.root_frequencies.begin(),
                 model.nucleotide_model.root_frequencies.end(),
                 result.nodes.begin() + model.plan.root() * 4,
                 [](double value) { return static_cast<Scalar>(value); });
  for (std::size_t node = 0; node < model.plan.num_nodes(); ++node) {
    const Nucleotide observation = model.observations[node];
    if (observation == Nucleotide::kUnknown)
      continue;
    const std::uint8_t mask = static_cast<std::uint8_t>(observation);
    if (mask == 0 || mask > static_cast<std::uint8_t>(Nucleotide::kUnknown))
      throw std::invalid_argument("invalid nucleotide observation");
    Scalar *potential = result.nodes.data() + node * 4;
    for (int candidate = 0; candidate < 4; ++candidate) {
      if (!AllowsState(observation, candidate))
        potential[candidate] = 0.0;
    }
  }

  result.edges.reserve(model.plan.num_edges() * 16);
  for (const Scalar length : model.branch_lengths) {
    const auto transition =
        NucleotideTransition(model.nucleotide_model, length);
    result.edges.insert(result.edges.end(), transition.begin(),
                        transition.end());
  }
  return result;
}

} // namespace

SiteModelView SelectRateCategory(SiteModelView model, std::size_t category) {
  ValidateRateMixture(model.rate_mixture);
  model.nucleotide_model.substitution_rate *=
      RateCategoryRate(model.rate_mixture, category);
  model.rate_mixture = {};
  return model;
}

std::array<Scalar, 16> JukesCantorTransition(Scalar branch_length,
                                             Scalar rate) {
  return NucleotideTransition(JukesCantorModel(rate), branch_length);
}

Scalar SiteLikelihood(SiteModelView model) {
  return std::exp(SiteLogLikelihood(model));
}

Scalar SiteLogLikelihood(SiteModelView model) {
  ValidateRateMixture(model.rate_mixture);
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  Scalar result = -std::numeric_limits<Scalar>::infinity();
  for (std::size_t category = 0; category < categories; ++category) {
    const Potentials potentials =
        BuildPotentials(SelectRateCategory(model, category));
    const Scalar category_log = tree_hmm::LogPartitionFunction(
        {model.plan, 4, potentials.nodes, potentials.edges});
    const Scalar weighted =
        category_log +
        std::log(RateCategoryWeight(model.rate_mixture, category));
    if (result == -std::numeric_limits<Scalar>::infinity())
      result = weighted;
    else {
      const Scalar maximum = std::max(result, weighted);
      result = maximum + std::log(std::exp(result - maximum) +
                                  std::exp(weighted - maximum));
    }
  }
  return result;
}

SitePosterior AncestralPosterior(SiteModelView model) {
  ValidateRateMixture(model.rate_mixture);
  const Scalar log_likelihood = SiteLogLikelihood(model);
  SitePosterior result{
      .likelihood = std::exp(log_likelihood),
      .ancestral_states =
          std::vector<Scalar>(model.plan.num_nodes() * 4, Scalar{0}),
      .substitutions =
          std::vector<Scalar>(model.plan.num_edges() * 16, Scalar{0}),
      .rate_categories =
          std::vector<Scalar>(RateCategoryCount(model.rate_mixture), Scalar{0}),
  };
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  for (std::size_t category = 0; category < categories; ++category) {
    const Potentials potentials =
        BuildPotentials(SelectRateCategory(model, category));
    tree_hmm::Marginals marginals = tree_hmm::PosteriorMarginals(
        {model.plan, 4, potentials.nodes, potentials.edges});
    const Scalar category_log = tree_hmm::LogPartitionFunction(
        {model.plan, 4, potentials.nodes, potentials.edges});
    const Scalar posterior_weight =
        std::exp(category_log +
                 std::log(RateCategoryWeight(model.rate_mixture, category)) -
                 log_likelihood);
    result.rate_categories[category] = posterior_weight;
    for (std::size_t index = 0; index < result.ancestral_states.size(); ++index)
      result.ancestral_states[index] +=
          posterior_weight * marginals.nodes[index];
    for (std::size_t index = 0; index < result.substitutions.size(); ++index)
      result.substitutions[index] += posterior_weight * marginals.edges[index];
  }
  return result;
}

} // namespace parallel_phylogenetics
