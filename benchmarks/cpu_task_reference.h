#ifndef PARALLEL_PHYLOGENETICS_CPU_TASK_REFERENCE_H_
#define PARALLEL_PHYLOGENETICS_CPU_TASK_REFERENCE_H_

#include "parallel_phylogenetics/alignment.h"

#include "tree_hmm/inference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace parallel_phylogenetics::benchmark {

enum class InferenceTask {
  kLikelihood,
  kMaximum,
  kSampling,
  kMarginals,
};

// A preallocated, batched application wrapper around the reusable scalar CPU
// tree-HMM algebra. Dense factors for every site are retained so kFactors and
// kNone have the same application-level meaning as for the accelerator APIs.
// The generic CPU algebra itself consumes one dense site at a time.
class CpuTaskReference {
public:
  void Reserve(AlignmentModelView model, InferenceTask task) {
    if (model.sites == 0)
      throw std::invalid_argument(
          "an alignment must contain at least one site");
    plan_ = &model.plan;
    sites_ = model.sites;
    observation_nodes_.assign(model.observation_nodes.begin(),
                              model.observation_nodes.end());
    task_ = task;
    nodes_.resize(CheckedProduct({sites_, model.plan.num_nodes(), 4},
                                 "CPU task node factors"));
    edges_.resize(
        CheckedProduct({model.plan.num_edges(), 16}, "CPU task edge factors"));
    tree_hmm_.Reserve(model.plan, 4);

    log_values_.clear();
    node_values_.clear();
    edge_values_.clear();
    states_.clear();
    switch (task_) {
    case InferenceTask::kLikelihood:
      log_values_.resize(sites_);
      break;
    case InferenceTask::kMaximum:
      log_values_.resize(sites_);
      states_.resize(CheckedProduct({sites_, model.plan.num_nodes()},
                                    "CPU MAP assignments"));
      break;
    case InferenceTask::kSampling:
      states_.resize(CheckedProduct({sites_, model.plan.num_nodes()},
                                    "CPU posterior samples"));
      break;
    case InferenceTask::kMarginals:
      log_values_.resize(sites_);
      node_values_.resize(CheckedProduct({sites_, model.plan.num_nodes(), 4},
                                         "CPU node marginals"));
      edge_values_.resize(CheckedProduct({sites_, model.plan.num_edges(), 16},
                                         "CPU edge marginals"));
      break;
    }
  }

  void Evaluate(AlignmentModelView model, InputUpdate update,
                std::span<const Scalar> uniforms = {}) {
    Validate(model, uniforms);
    if (update == InputUpdate::kAll)
      PrepareAll(model);
    else if (update == InputUpdate::kFactors)
      PrepareSharedFactors(model);

    const std::size_t node_stride = model.plan.num_nodes() * 4;
    const std::size_t state_stride = model.plan.num_nodes();
    const std::size_t edge_stride = model.plan.num_edges() * 16;
    for (std::size_t site = 0; site < sites_; ++site) {
      const tree_hmm::ModelView factors{model.plan, 4,
                                        std::span<const Scalar>(nodes_).subspan(
                                            site * node_stride, node_stride),
                                        edges_};
      switch (task_) {
      case InferenceTask::kLikelihood:
        log_values_[site] =
            tree_hmm::LogPartitionFunctionPrepared(factors, tree_hmm_);
        break;
      case InferenceTask::kMaximum: {
        const tree_hmm::MaximumAssignmentView result =
            tree_hmm::MaximumAPosterioriPrepared(factors, tree_hmm_);
        log_values_[site] = result.log_weight;
        std::copy(result.states.begin(), result.states.end(),
                  states_.begin() + site * state_stride);
        break;
      }
      case InferenceTask::kSampling: {
        const std::span<const std::size_t> result =
            tree_hmm::PosteriorSamplePrepared(
                factors, uniforms.subspan(site * state_stride, state_stride),
                tree_hmm_);
        std::copy(result.begin(), result.end(),
                  states_.begin() + site * state_stride);
        break;
      }
      case InferenceTask::kMarginals: {
        const tree_hmm::MarginalView result =
            tree_hmm::PosteriorMarginalsPrepared(factors, tree_hmm_);
        log_values_[site] = result.log_partition;
        std::copy(result.nodes.begin(), result.nodes.end(),
                  node_values_.begin() + site * node_stride);
        std::copy(result.edges.begin(), result.edges.end(),
                  edge_values_.begin() + site * edge_stride);
        break;
      }
      }
    }
  }

  std::span<const Scalar> log_values() const { return log_values_; }
  std::span<const Scalar> node_values() const { return node_values_; }
  std::span<const Scalar> edge_values() const { return edge_values_; }
  std::span<const std::size_t> states() const { return states_; }

private:
  static std::size_t CheckedProduct(std::initializer_list<std::size_t> values,
                                    const char *description) {
    std::size_t result = 1;
    for (const std::size_t value : values) {
      if (value != 0 &&
          result > std::numeric_limits<std::size_t>::max() / value) {
        throw std::length_error(std::string(description) + " overflows size_t");
      }
      result *= value;
    }
    return result;
  }

  void Validate(AlignmentModelView model,
                std::span<const Scalar> uniforms) const {
    if (plan_ != &model.plan || sites_ != model.sites ||
        observation_nodes_.size() != model.observation_nodes.size() ||
        !std::equal(observation_nodes_.begin(), observation_nodes_.end(),
                    model.observation_nodes.begin())) {
      throw std::invalid_argument(
          "CPU task reference requires Reserve for this alignment shape");
    }
    const std::size_t expected_uniforms =
        task_ == InferenceTask::kSampling
            ? CheckedProduct({sites_, model.plan.num_nodes()},
                             "CPU posterior-sampling variates")
            : 0;
    if (uniforms.size() != expected_uniforms) {
      throw std::invalid_argument(
          task_ == InferenceTask::kSampling
              ? "posterior sampling requires one variate per site and node"
              : "uniform variates apply only to posterior sampling");
    }
  }

  void PrepareSharedFactors(AlignmentModelView model) {
    for (std::size_t edge = 0; edge < model.plan.num_edges(); ++edge) {
      const auto transition = JukesCantorTransition(model.branch_lengths[edge],
                                                    model.substitution_rate);
      std::copy(transition.begin(), transition.end(),
                edges_.begin() + edge * 16);
    }

    const std::size_t node_stride = model.plan.num_nodes() * 4;
    for (std::size_t site = 0; site < sites_; ++site) {
      Scalar *root = nodes_.data() + site * node_stride + model.plan.root() * 4;
      std::copy(model.root_frequencies.begin(), model.root_frequencies.end(),
                root);
    }
    // An observed root combines its emission factor with the refreshed prior.
    const auto root_observation =
        std::lower_bound(model.observation_nodes.begin(),
                         model.observation_nodes.end(), model.plan.root());
    if (root_observation == model.observation_nodes.end() ||
        *root_observation != model.plan.root()) {
      return;
    }
    const std::size_t observation_index = static_cast<std::size_t>(
        root_observation - model.observation_nodes.begin());
    for (std::size_t site = 0; site < sites_; ++site) {
      Mask(nodes_.data() + site * node_stride + model.plan.root() * 4,
           model.observations[site * model.observation_nodes.size() +
                              observation_index]);
    }
  }

  void PrepareAll(AlignmentModelView model) {
    std::fill(nodes_.begin(), nodes_.end(), Scalar{1});
    PrepareSharedFactors(model);
    const std::size_t node_stride = model.plan.num_nodes() * 4;
    for (std::size_t site = 0; site < sites_; ++site) {
      for (std::size_t index = 0; index < model.observation_nodes.size();
           ++index) {
        const btrc::Index node = model.observation_nodes[index];
        if (node == model.plan.root())
          continue;
        Mask(nodes_.data() + site * node_stride + node * 4,
             model.observations[site * model.observation_nodes.size() + index]);
      }
    }
  }

  static void Mask(Scalar *factor, Nucleotide observation) {
    const std::uint8_t mask = static_cast<std::uint8_t>(observation);
    if (mask == 0 || mask > static_cast<std::uint8_t>(Nucleotide::kUnknown))
      throw std::invalid_argument("invalid nucleotide observation");
    for (std::size_t state = 0; state < 4; ++state) {
      if (!AllowsState(observation, state))
        factor[state] = Scalar{0};
    }
  }

  const btrc::Plan *plan_ = nullptr;
  std::size_t sites_ = 0;
  InferenceTask task_ = InferenceTask::kLikelihood;
  std::vector<btrc::Index> observation_nodes_;
  std::vector<Scalar> nodes_;
  std::vector<Scalar> edges_;
  std::vector<Scalar> log_values_;
  std::vector<Scalar> node_values_;
  std::vector<Scalar> edge_values_;
  std::vector<std::size_t> states_;
  tree_hmm::Workspace tree_hmm_;
};

} // namespace parallel_phylogenetics::benchmark

#endif // PARALLEL_PHYLOGENETICS_CPU_TASK_REFERENCE_H_
