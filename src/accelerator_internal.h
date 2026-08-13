#ifndef PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_
#define PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "parallel_phylogenetics/alignment.h"
#include "src/alignment_internal.h"

namespace parallel_phylogenetics::internal {

enum class PreparedOperation {
  kLikelihood,
  kMaximum,
  kSampling,
  kMarginals,
};

template <class BackendWorkspace> struct WorkspaceStorage {
  const btrc::Plan *plan = nullptr;
  std::size_t sites = 0;
  std::size_t batch_capacity = 0;
  PreparedOperation operation = PreparedOperation::kLikelihood;
  std::vector<btrc::Index> observation_nodes;
  std::vector<Scalar> output;
  BackendWorkspace tree_hmm;
};

inline std::size_t BatchCapacity(AlignmentModelView model,
                                 std::size_t requested) {
  if (model.sites == 0)
    throw std::invalid_argument("an alignment must contain at least one site");
  const std::size_t capacity = requested == 0 ? model.sites : requested;
  if (capacity > model.sites)
    throw std::invalid_argument(
        "site batch capacity cannot exceed the alignment site count");
  return capacity;
}

inline void ValidatePrepared(AlignmentModelView model, const btrc::Plan *plan,
                             std::size_t batch_capacity,
                             std::span<const btrc::Index> observation_nodes,
                             PreparedOperation reserved_operation,
                             PreparedOperation operation,
                             bool require_single_batch = true) {
  if (plan != &model.plan || model.sites == 0 ||
      (require_single_batch && model.sites > batch_capacity) ||
      reserved_operation != operation ||
      observation_nodes.size() != model.observation_nodes.size() ||
      !std::equal(observation_nodes.begin(), observation_nodes.end(),
                  model.observation_nodes.begin(),
                  model.observation_nodes.end())) {
    throw std::invalid_argument(
        "prepared accelerator inference requires the corresponding Reserve "
        "operation for this alignment shape and batch capacity");
  }
}

template <class BackendWorkspace, class Reserve>
void ReserveOperation(WorkspaceStorage<BackendWorkspace> &storage,
                      AlignmentModelView model, std::size_t requested_capacity,
                      PreparedOperation operation, Reserve &&reserve) {
  const std::size_t batch = BatchCapacity(model, requested_capacity);
  reserve(batch);
  storage.plan = &model.plan;
  storage.sites = model.sites;
  storage.batch_capacity = batch;
  storage.operation = operation;
  storage.observation_nodes.assign(model.observation_nodes.begin(),
                                   model.observation_nodes.end());
  if (operation == PreparedOperation::kLikelihood)
    storage.output.resize(model.sites);
  else
    storage.output.clear();
}

template <class BackendWorkspace>
void ValidatePrepared(AlignmentModelView model,
                      const WorkspaceStorage<BackendWorkspace> &storage,
                      PreparedOperation operation,
                      bool require_single_batch = true) {
  ValidatePrepared(model, storage.plan, storage.batch_capacity,
                   storage.observation_nodes, storage.operation, operation,
                   require_single_batch);
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
std::span<const Scalar>
LogLikelihoodsPrepared(AlignmentModelView model, std::size_t batch_capacity,
                       Destination destination, Evaluate &&evaluate,
                       std::span<Scalar> output) {
  if (batch_capacity == 0 || destination.batch != batch_capacity ||
      output.size() != model.sites) {
    throw std::invalid_argument(
        "phylogenetic accelerator workspace has the wrong capacity");
  }
  PrepareCategoricalShared(model, destination);
  for (std::size_t first_site = 0; first_site < model.sites;
       first_site += batch_capacity) {
    const std::size_t count =
        std::min(batch_capacity, model.sites - first_site);
    const auto factors = PrepareCategoricalObservations(
        SelectSites(model, first_site, count), BatchPrefix(destination, count));
    const tree_hmm::PartitionView result = evaluate(factors);
    if (result.values.size() != count)
      throw std::runtime_error("tree-HMM backend returned a wrong batch size");
    std::copy(result.values.begin(), result.values.end(),
              output.begin() + first_site);
  }
  return output;
}

template <class Destination, class Evaluate>
AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView model,
                                                Destination destination,
                                                Evaluate &&evaluate) {
  const auto factors = Prepare(model, BatchPrefix(destination, model.sites));
  const tree_hmm::BatchedMaximumAssignmentView result = evaluate(factors);
  if (result.log_weights.size() != model.sites ||
      result.states.size() != model.sites * model.plan.num_nodes()) {
    throw std::runtime_error("tree-HMM backend returned a wrong MAP shape");
  }
  return {result.log_weights, result.states};
}

template <class Destination, class Uniforms, class Evaluate>
AlignmentPosteriorSampleView PosteriorSamplePrepared(
    AlignmentModelView model, std::span<const Scalar> variates,
    Destination destination, Uniforms &&uniforms, Evaluate &&evaluate) {
  const std::size_t expected_uniforms = model.sites * model.plan.num_nodes();
  if (variates.size() != expected_uniforms)
    throw std::invalid_argument(
        "posterior sampling requires one uniform variate per site and node");
  const auto factors = Prepare(model, BatchPrefix(destination, model.sites));
  std::span<Scalar> staged_uniforms = uniforms(model.sites);
  std::copy(variates.begin(), variates.end(), staged_uniforms.begin());
  const tree_hmm::BatchedPosteriorSampleView result =
      evaluate(factors, staged_uniforms);
  if (result.states.size() != expected_uniforms) {
    throw std::runtime_error(
        "tree-HMM backend returned a wrong posterior-sample shape");
  }
  return {result.states};
}

template <class Destination, class Evaluate>
AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView model,
                                                  Destination destination,
                                                  Evaluate &&evaluate) {
  const auto factors = Prepare(model, BatchPrefix(destination, model.sites));
  const tree_hmm::BatchedMarginalView result = evaluate(factors);
  const std::size_t expected_nodes = model.sites * model.plan.num_nodes() * 4;
  const std::size_t expected_edges = model.sites * model.plan.num_edges() * 16;
  if (result.log_partitions.size() != model.sites ||
      result.nodes.size() != expected_nodes ||
      result.edges.size() != expected_edges) {
    throw std::runtime_error(
        "tree-HMM backend returned a wrong posterior-marginal shape");
  }
  return {result.log_partitions, result.nodes, result.edges};
}

} // namespace parallel_phylogenetics::internal

#endif // PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_
