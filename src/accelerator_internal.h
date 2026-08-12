#ifndef PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_
#define PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "parallel_phylogenetics/alignment.h"

namespace parallel_phylogenetics::internal {

inline AlignmentModelView SiteRange(AlignmentModelView model,
                                    std::size_t first_site,
                                    std::size_t site_count) {
  if (model.sites == 0 || model.observations.size() % model.sites != 0 ||
      model.observations.size() / model.sites !=
          model.observation_nodes.size() ||
      first_site > model.sites || site_count > model.sites - first_site) {
    throw std::invalid_argument("invalid phylogenetic alignment site range");
  }
  return {
      model.plan,
      site_count,
      model.branch_lengths,
      model.observation_nodes,
      model.observations.subspan(first_site * model.observation_nodes.size(),
                                 site_count * model.observation_nodes.size()),
      model.root_frequencies,
      model.substitution_rate};
}

inline tree_hmm::MutableBatchedModelView
BatchPrefix(tree_hmm::MutableBatchedModelView destination, std::size_t batch) {
  if (batch == 0 || batch > destination.batch ||
      destination.node_potentials.size() % destination.batch != 0) {
    throw std::invalid_argument("invalid dense accelerator batch prefix");
  }
  const std::size_t node_values =
      batch * (destination.node_potentials.size() / destination.batch);
  return {destination.plan, destination.states, batch,
          destination.node_potentials.first(node_values),
          destination.edge_potentials};
}

inline tree_hmm::MutableBatchedCategoricalModelView
BatchPrefix(tree_hmm::MutableBatchedCategoricalModelView destination,
            std::size_t batch) {
  if (batch == 0 || batch > destination.batch ||
      destination.observations.size() % destination.batch != 0) {
    throw std::invalid_argument("invalid categorical accelerator batch prefix");
  }
  const std::size_t observation_values =
      batch * (destination.observations.size() / destination.batch);
  return {destination.plan,
          destination.states,
          batch,
          destination.categories,
          destination.observation_nodes,
          destination.observations.first(observation_values),
          destination.root_potential,
          destination.emission_potentials,
          destination.edge_potentials};
}

template <class Destination, class Evaluate>
std::span<const float>
LogLikelihoodsPrepared(AlignmentModelView model, std::size_t batch_capacity,
                       Destination destination, Evaluate &&evaluate,
                       std::span<float> output) {
  if (batch_capacity == 0 || destination.batch != batch_capacity ||
      output.size() != model.sites) {
    throw std::invalid_argument(
        "phylogenetic accelerator workspace has the wrong capacity");
  }
  for (std::size_t first_site = 0; first_site < model.sites;
       first_site += batch_capacity) {
    const std::size_t count =
        std::min(batch_capacity, model.sites - first_site);
    const auto factors = Prepare(SiteRange(model, first_site, count),
                                 BatchPrefix(destination, count));
    const tree_hmm::PartitionView result = evaluate(factors);
    if (result.values.size() != count)
      throw std::runtime_error("tree-HMM backend returned a wrong batch size");
    std::copy(result.values.begin(), result.values.end(),
              output.begin() + first_site);
  }
  return output;
}

} // namespace parallel_phylogenetics::internal

#endif // PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_
