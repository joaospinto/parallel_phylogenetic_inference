#ifndef PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_
#define PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
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
  std::size_t rate_categories = 1;
  PreparedOperation operation = PreparedOperation::kLikelihood;
  std::vector<btrc::Index> observation_nodes;
  std::vector<Scalar> output;
  std::vector<Scalar> node_output;
  std::vector<Scalar> edge_output;
  std::vector<Scalar> category_output;
  std::vector<std::uint32_t> state_output;
  std::vector<std::uint32_t> selected_categories;
  PreparedTimings timings;
  std::vector<BackendWorkspace> tree_hmms;
  // Mixture sampling needs p(category | observations) before drawing the
  // category-conditional ancestral state. These likelihood-only workspaces
  // keep both prepared input sets resident alongside the sampling workspaces.
  std::vector<BackendWorkspace> sampling_likelihood_tree_hmms;
};

inline void AccumulateTimings(PreparedTimings &destination,
                              const tree_hmm::AcceleratorTimings &source) {
  destination.backend.upload_ms += source.upload_ms;
  destination.backend.kernel_ms += source.kernel_ms;
  destination.backend.download_ms += source.download_ms;
  destination.backend.wall_ms += source.wall_ms;
  ++destination.site_batches;
}

using Clock = std::chrono::steady_clock;

inline void ResetTimings(PreparedTimings &timings) { timings = {}; }

inline void FinishTimings(Clock::time_point start, PreparedTimings &timings) {
  timings.evaluation_wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline Scalar LogAdd(Scalar first, Scalar second) {
  if (first == -std::numeric_limits<Scalar>::infinity())
    return second;
  if (second == -std::numeric_limits<Scalar>::infinity())
    return first;
  const Scalar maximum = std::max(first, second);
  return maximum +
         std::log(std::exp(first - maximum) + std::exp(second - maximum));
}

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
                             std::size_t rate_categories,
                             std::span<const btrc::Index> observation_nodes,
                             PreparedOperation reserved_operation,
                             PreparedOperation operation,
                             bool require_single_batch = true) {
  if (plan != &model.plan || model.sites == 0 ||
      (require_single_batch && model.sites > batch_capacity) ||
      RateCategoryCount(model.rate_mixture) != rate_categories ||
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
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  storage.tree_hmms.clear();
  storage.tree_hmms.resize(categories);
  for (BackendWorkspace &tree_hmm : storage.tree_hmms)
    reserve(tree_hmm, batch);
  storage.sampling_likelihood_tree_hmms.clear();
  storage.plan = &model.plan;
  storage.sites = model.sites;
  storage.batch_capacity = batch;
  storage.rate_categories = categories;
  storage.operation = operation;
  storage.timings = {};
  storage.observation_nodes.assign(model.observation_nodes.begin(),
                                   model.observation_nodes.end());
  storage.output.clear();
  storage.node_output.clear();
  storage.edge_output.clear();
  storage.category_output.clear();
  storage.state_output.clear();
  storage.selected_categories.clear();
  switch (operation) {
  case PreparedOperation::kLikelihood:
    storage.output.resize(model.sites);
    break;
  case PreparedOperation::kMaximum:
    storage.output.resize(model.sites);
    storage.state_output.resize(model.sites * model.plan.num_nodes());
    storage.selected_categories.resize(model.sites);
    break;
  case PreparedOperation::kSampling:
    storage.category_output.resize(categories * model.sites);
    storage.state_output.resize(model.sites * model.plan.num_nodes());
    storage.selected_categories.resize(model.sites);
    break;
  case PreparedOperation::kMarginals:
    storage.output.resize(model.sites);
    storage.node_output.resize(model.sites * model.plan.num_nodes() * 4);
    storage.edge_output.resize(model.sites * model.plan.num_edges() * 16);
    storage.category_output.resize(categories * model.sites);
    break;
  }
}

template <class BackendWorkspace>
void ValidatePrepared(AlignmentModelView model,
                      const WorkspaceStorage<BackendWorkspace> &storage,
                      PreparedOperation operation,
                      bool require_single_batch = true) {
  ValidatePrepared(model, storage.plan, storage.batch_capacity,
                   storage.rate_categories, storage.observation_nodes,
                   storage.operation, operation, require_single_batch);
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

template <class Workspaces, class Inputs, class Evaluate>
std::span<const Scalar>
LogLikelihoodsPrepared(AlignmentModelView model, std::size_t batch_capacity,
                       Workspaces &workspaces, Inputs &&inputs,
                       Evaluate &&evaluate, std::span<Scalar> output,
                       InputUpdate update, PreparedTimings &timings) {
  const Clock::time_point start = Clock::now();
  ResetTimings(timings);
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  if (batch_capacity == 0 || workspaces.size() != categories ||
      output.size() != model.sites) {
    throw std::invalid_argument(
        "phylogenetic accelerator workspace has the wrong capacity");
  }
  if (update != InputUpdate::kAll && model.sites != batch_capacity) {
    throw std::invalid_argument(
        "resident phylogenetic inputs require one unchunked site batch");
  }
  std::fill(output.begin(), output.end(),
            -std::numeric_limits<Scalar>::infinity());
  for (std::size_t category = 0; category < categories; ++category) {
    const AlignmentModelView category_model =
        SelectRateCategory(model, category);
    tree_hmm::MutableBatchedCategoricalModelView destination =
        inputs(workspaces[category]);
    if (destination.batch != batch_capacity)
      throw std::invalid_argument(
          "phylogenetic accelerator category workspace has wrong capacity");
    if (update != InputUpdate::kNone)
      PrepareCategoricalShared(category_model, destination);
    const Scalar log_weight = static_cast<Scalar>(
        std::log(RateCategoryWeight(model.rate_mixture, category)));
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += batch_capacity) {
      const std::size_t count =
          std::min(batch_capacity, model.sites - first_site);
      const tree_hmm::MutableBatchedCategoricalModelView batch_destination =
          BatchPrefix(destination, count);
      const tree_hmm::BatchedCategoricalModelView factors =
          update == InputUpdate::kAll
              ? PrepareCategoricalObservations(
                    SelectSites(category_model, first_site, count),
                    batch_destination)
              : static_cast<tree_hmm::BatchedCategoricalModelView>(
                    batch_destination);
      const tree_hmm::PartitionView result =
          evaluate(factors, workspaces[category], update);
      AccumulateTimings(timings, result.timings);
      if (result.values.size() != count) {
        throw std::runtime_error(
            "tree-HMM backend returned a wrong batch size");
      }
      for (std::size_t index = 0; index < count; ++index) {
        output[first_site + index] = LogAdd(output[first_site + index],
                                            result.values[index] + log_weight);
      }
    }
  }
  FinishTimings(start, timings);
  return output;
}

inline tree_hmm::BatchedCategoricalModelView
PrepareForUpdate(AlignmentModelView model,
                 tree_hmm::MutableBatchedCategoricalModelView destination,
                 InputUpdate update) {
  if (update != InputUpdate::kNone)
    PrepareCategoricalShared(model, destination);
  if (update == InputUpdate::kAll)
    return PrepareCategoricalObservations(model, destination);
  return destination;
}

template <class Workspaces, class Inputs, class Evaluate>
AlignmentMaximumView
MaximumAPosterioriPrepared(AlignmentModelView model, Workspaces &workspaces,
                           Inputs &&inputs, Evaluate &&evaluate,
                           std::span<Scalar> log_weights,
                           std::span<std::uint32_t> states,
                           std::span<std::uint32_t> selected_categories,
                           InputUpdate update, PreparedTimings &timings) {
  const Clock::time_point start = Clock::now();
  ResetTimings(timings);
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  const std::size_t state_values = model.sites * model.plan.num_nodes();
  if (workspaces.size() != categories || log_weights.size() != model.sites ||
      states.size() != state_values ||
      selected_categories.size() != model.sites) {
    throw std::invalid_argument(
        "phylogenetic MAP workspace has the wrong mixture shape");
  }
  std::fill(log_weights.begin(), log_weights.end(),
            -std::numeric_limits<Scalar>::infinity());
  for (std::size_t category = 0; category < categories; ++category) {
    const AlignmentModelView category_model =
        SelectRateCategory(model, category);
    const auto factors = PrepareForUpdate(
        category_model, BatchPrefix(inputs(workspaces[category]), model.sites),
        update);
    const tree_hmm::BatchedMaximumAssignmentView result =
        evaluate(factors, workspaces[category], update);
    AccumulateTimings(timings, result.timings);
    if (result.log_weights.size() != model.sites ||
        result.states.size() != state_values) {
      throw std::runtime_error("tree-HMM backend returned a wrong MAP shape");
    }
    const Scalar log_weight = static_cast<Scalar>(
        std::log(RateCategoryWeight(model.rate_mixture, category)));
    for (std::size_t site = 0; site < model.sites; ++site) {
      const Scalar candidate = result.log_weights[site] + log_weight;
      if (candidate > log_weights[site]) {
        log_weights[site] = candidate;
        selected_categories[site] = static_cast<std::uint32_t>(category);
        const std::size_t offset = site * model.plan.num_nodes();
        std::transform(result.states.begin() + offset,
                       result.states.begin() + offset + model.plan.num_nodes(),
                       states.begin() + offset, [](std::size_t state) {
                         return static_cast<std::uint32_t>(state);
                       });
      }
    }
  }
  FinishTimings(start, timings);
  return {log_weights, states, selected_categories};
}

template <class Workspaces, class LikelihoodWorkspaces, class Inputs,
          class Uniforms, class EvaluateLikelihood, class EvaluateSample>
AlignmentPosteriorSampleView PosteriorSamplePrepared(
    AlignmentModelView model, std::span<const Scalar> variates,
    Workspaces &workspaces, LikelihoodWorkspaces &likelihood_workspaces,
    Inputs &&inputs, Uniforms &&uniforms,
    EvaluateLikelihood &&evaluate_likelihood, EvaluateSample &&evaluate_sample,
    std::span<Scalar> category_log_values, std::span<std::uint32_t> states,
    std::span<std::uint32_t> selected_categories, InputUpdate update,
    PreparedTimings &timings) {
  const Clock::time_point start = Clock::now();
  ResetTimings(timings);
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  const std::size_t node_uniforms = model.sites * model.plan.num_nodes();
  const std::size_t expected_uniforms =
      node_uniforms + (categories > 1 ? model.sites : 0);
  if (variates.size() != expected_uniforms)
    throw std::invalid_argument(
        categories > 1
            ? "mixture posterior sampling requires one uniform variate per "
              "site and node followed by one rate-category variate per site"
            : "posterior sampling requires one uniform variate per site and "
              "node");
  if (workspaces.size() != categories || states.size() != node_uniforms ||
      selected_categories.size() != model.sites ||
      category_log_values.size() != categories * model.sites ||
      (categories > 1 && likelihood_workspaces.size() != categories)) {
    throw std::invalid_argument(
        "phylogenetic sampling workspace has the wrong mixture shape");
  }

  if (categories == 1) {
    std::fill(selected_categories.begin(), selected_categories.end(), 0);
  } else {
    for (std::size_t category = 0; category < categories; ++category) {
      const AlignmentModelView category_model =
          SelectRateCategory(model, category);
      const auto factors = PrepareForUpdate(
          category_model,
          BatchPrefix(inputs(likelihood_workspaces[category]), model.sites),
          update);
      const tree_hmm::PartitionView result =
          evaluate_likelihood(factors, likelihood_workspaces[category], update);
      AccumulateTimings(timings, result.timings);
      if (result.values.size() != model.sites)
        throw std::runtime_error(
            "tree-HMM backend returned a wrong category-likelihood shape");
      const Scalar log_weight = static_cast<Scalar>(
          std::log(RateCategoryWeight(model.rate_mixture, category)));
      for (std::size_t site = 0; site < model.sites; ++site) {
        category_log_values[site * categories + category] =
            result.values[site] + log_weight;
      }
    }
    for (std::size_t site = 0; site < model.sites; ++site) {
      Scalar total = -std::numeric_limits<Scalar>::infinity();
      for (std::size_t category = 0; category < categories; ++category) {
        total =
            LogAdd(total, category_log_values[site * categories + category]);
      }
      const Scalar draw = variates[node_uniforms + site];
      if (!(draw >= Scalar{0} && draw < Scalar{1}))
        throw std::invalid_argument(
            "posterior-sampling variates must lie in [0,1)");
      Scalar cumulative = 0.0;
      std::size_t selected = categories - 1;
      for (std::size_t category = 0; category + 1 < categories; ++category) {
        cumulative +=
            std::exp(category_log_values[site * categories + category] - total);
        if (draw < cumulative) {
          selected = category;
          break;
        }
      }
      selected_categories[site] = static_cast<std::uint32_t>(selected);
    }
  }

  for (std::size_t category = 0; category < categories; ++category) {
    const AlignmentModelView category_model =
        SelectRateCategory(model, category);
    const auto factors = PrepareForUpdate(
        category_model, BatchPrefix(inputs(workspaces[category]), model.sites),
        update);
    std::span<Scalar> staged_uniforms =
        uniforms(workspaces[category], model.sites);
    std::copy(variates.begin(), variates.begin() + node_uniforms,
              staged_uniforms.begin());
    const tree_hmm::BatchedPosteriorSampleView result =
        evaluate_sample(factors, staged_uniforms, workspaces[category], update);
    AccumulateTimings(timings, result.timings);
    if (result.states.size() != node_uniforms) {
      throw std::runtime_error(
          "tree-HMM backend returned a wrong posterior-sample shape");
    }
    for (std::size_t site = 0; site < model.sites; ++site) {
      if (selected_categories[site] != category)
        continue;
      const std::size_t offset = site * model.plan.num_nodes();
      std::transform(result.states.begin() + offset,
                     result.states.begin() + offset + model.plan.num_nodes(),
                     states.begin() + offset, [](std::size_t state) {
                       return static_cast<std::uint32_t>(state);
                     });
    }
  }
  FinishTimings(start, timings);
  return {states, selected_categories};
}

template <class Workspaces, class Inputs, class Evaluate>
AlignmentPosteriorView PosteriorMarginalsPrepared(
    AlignmentModelView model, Workspaces &workspaces, Inputs &&inputs,
    Evaluate &&evaluate, std::span<Scalar> log_values,
    std::span<Scalar> node_values, std::span<Scalar> edge_values,
    std::span<Scalar> category_values, InputUpdate update,
    PreparedTimings &timings) {
  const Clock::time_point start = Clock::now();
  ResetTimings(timings);
  const std::size_t categories = RateCategoryCount(model.rate_mixture);
  const std::size_t expected_nodes = model.sites * model.plan.num_nodes() * 4;
  const std::size_t expected_edges = model.sites * model.plan.num_edges() * 16;
  if (workspaces.size() != categories || log_values.size() != model.sites ||
      node_values.size() != expected_nodes ||
      edge_values.size() != expected_edges ||
      category_values.size() != categories * model.sites) {
    throw std::invalid_argument(
        "phylogenetic marginal workspace has the wrong mixture shape");
  }
  std::fill(log_values.begin(), log_values.end(),
            -std::numeric_limits<Scalar>::infinity());
  std::fill(node_values.begin(), node_values.end(), Scalar{0});
  std::fill(edge_values.begin(), edge_values.end(), Scalar{0});
  for (std::size_t category = 0; category < categories; ++category) {
    const AlignmentModelView category_model =
        SelectRateCategory(model, category);
    const auto factors = PrepareForUpdate(
        category_model, BatchPrefix(inputs(workspaces[category]), model.sites),
        update);
    const tree_hmm::BatchedMarginalView result =
        evaluate(factors, workspaces[category], update);
    AccumulateTimings(timings, result.timings);
    if (result.log_partitions.size() != model.sites ||
        result.nodes.size() != expected_nodes ||
        result.edges.size() != expected_edges) {
      throw std::runtime_error(
          "tree-HMM backend returned a wrong posterior-marginal shape");
    }
    const Scalar log_weight = static_cast<Scalar>(
        std::log(RateCategoryWeight(model.rate_mixture, category)));
    for (std::size_t site = 0; site < model.sites; ++site) {
      const Scalar weighted = result.log_partitions[site] + log_weight;
      category_values[site * categories + category] = weighted;
      const Scalar combined = LogAdd(log_values[site], weighted);
      const Scalar previous_weight =
          log_values[site] == -std::numeric_limits<Scalar>::infinity()
              ? Scalar{0}
              : std::exp(log_values[site] - combined);
      const Scalar category_weight = std::exp(weighted - combined);
      const std::size_t node_offset = site * model.plan.num_nodes() * 4;
      const std::size_t edge_offset = site * model.plan.num_edges() * 16;
      for (std::size_t index = 0; index < model.plan.num_nodes() * 4; ++index) {
        node_values[node_offset + index] =
            previous_weight * node_values[node_offset + index] +
            category_weight * result.nodes[node_offset + index];
      }
      for (std::size_t index = 0; index < model.plan.num_edges() * 16;
           ++index) {
        edge_values[edge_offset + index] =
            previous_weight * edge_values[edge_offset + index] +
            category_weight * result.edges[edge_offset + index];
      }
      log_values[site] = combined;
    }
  }
  for (std::size_t category = 0; category < categories; ++category) {
    for (std::size_t site = 0; site < model.sites; ++site) {
      category_values[site * categories + category] = std::exp(
          category_values[site * categories + category] - log_values[site]);
    }
  }
  FinishTimings(start, timings);
  return {log_values, node_values, edge_values, category_values};
}

} // namespace parallel_phylogenetics::internal

#endif // PARALLEL_PHYLOGENETICS_ACCELERATOR_INTERNAL_H_
