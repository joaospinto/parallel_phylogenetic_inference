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
    rate_categories_ = RateCategoryCount(model.rate_mixture);
    task_ = task;
    nodes_.resize(CheckedProduct({sites_, model.plan.num_nodes(), 4},
                                 "CPU task node factors"));
    edges_.resize(CheckedProduct({rate_categories_, model.plan.num_edges(), 16},
                                 "CPU task edge factors"));
    tree_hmm_.Reserve(model.plan, 4);

    log_values_.clear();
    node_values_.clear();
    edge_values_.clear();
    states_.clear();
    category_values_.clear();
    selected_categories_.clear();
    switch (task_) {
    case InferenceTask::kLikelihood:
      log_values_.resize(sites_);
      break;
    case InferenceTask::kMaximum:
      log_values_.resize(sites_);
      states_.resize(CheckedProduct({sites_, model.plan.num_nodes()},
                                    "CPU MAP assignments"));
      selected_categories_.resize(sites_);
      break;
    case InferenceTask::kSampling:
      states_.resize(CheckedProduct({sites_, model.plan.num_nodes()},
                                    "CPU posterior samples"));
      category_values_.resize(rate_categories_ * sites_);
      selected_categories_.resize(sites_);
      break;
    case InferenceTask::kMarginals:
      log_values_.resize(sites_);
      node_values_.resize(CheckedProduct({sites_, model.plan.num_nodes(), 4},
                                         "CPU node marginals"));
      edge_values_.resize(CheckedProduct({sites_, model.plan.num_edges(), 16},
                                         "CPU edge marginals"));
      category_values_.resize(rate_categories_ * sites_);
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
    const auto factors = [&](std::size_t site, std::size_t category) {
      return tree_hmm::ModelView{model.plan, 4,
                                 std::span<const Scalar>(nodes_).subspan(
                                     site * node_stride, node_stride),
                                 std::span<const Scalar>(edges_).subspan(
                                     category * edge_stride, edge_stride)};
    };

    if (task_ == InferenceTask::kSampling) {
      for (std::size_t site = 0; site < sites_; ++site) {
        Scalar total = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t category = 0; category < rate_categories_;
             ++category) {
          const Scalar weighted =
              tree_hmm::LogPartitionFunctionPrepared(factors(site, category),
                                                     tree_hmm_) +
              static_cast<Scalar>(
                  std::log(RateCategoryWeight(model.rate_mixture, category)));
          category_values_[site * rate_categories_ + category] = weighted;
          total = LogAdd(total, weighted);
        }
        std::size_t selected = 0;
        if (rate_categories_ > 1) {
          const Scalar draw = uniforms[sites_ * state_stride + site];
          if (!(draw >= Scalar{0} && draw < Scalar{1})) {
            throw std::invalid_argument(
                "posterior-sampling variates must lie in [0,1)");
          }
          Scalar cumulative = 0.0;
          selected = rate_categories_ - 1;
          for (std::size_t category = 0; category + 1 < rate_categories_;
               ++category) {
            cumulative += std::exp(
                category_values_[site * rate_categories_ + category] - total);
            if (draw < cumulative) {
              selected = category;
              break;
            }
          }
        }
        selected_categories_[site] = selected;
        const std::span<const std::size_t> result =
            tree_hmm::PosteriorSamplePrepared(
                factors(site, selected),
                uniforms.subspan(site * state_stride, state_stride), tree_hmm_);
        std::copy(result.begin(), result.end(),
                  states_.begin() + site * state_stride);
      }
      return;
    }

    std::fill(log_values_.begin(), log_values_.end(),
              -std::numeric_limits<Scalar>::infinity());
    if (task_ == InferenceTask::kMarginals) {
      std::fill(node_values_.begin(), node_values_.end(), Scalar{0});
      std::fill(edge_values_.begin(), edge_values_.end(), Scalar{0});
    }
    for (std::size_t category = 0; category < rate_categories_; ++category) {
      const Scalar log_weight = static_cast<Scalar>(
          std::log(RateCategoryWeight(model.rate_mixture, category)));
      for (std::size_t site = 0; site < sites_; ++site) {
        if (task_ == InferenceTask::kLikelihood) {
          const Scalar weighted = tree_hmm::LogPartitionFunctionPrepared(
                                      factors(site, category), tree_hmm_) +
                                  log_weight;
          log_values_[site] = LogAdd(log_values_[site], weighted);
        } else if (task_ == InferenceTask::kMaximum) {
          const tree_hmm::MaximumAssignmentView result =
              tree_hmm::MaximumAPosterioriPrepared(factors(site, category),
                                                   tree_hmm_);
          const Scalar weighted = result.log_weight + log_weight;
          if (weighted > log_values_[site]) {
            log_values_[site] = weighted;
            selected_categories_[site] = category;
            std::copy(result.states.begin(), result.states.end(),
                      states_.begin() + site * state_stride);
          }
        } else {
          const tree_hmm::MarginalView result =
              tree_hmm::PosteriorMarginalsPrepared(factors(site, category),
                                                   tree_hmm_);
          const Scalar weighted = result.log_partition + log_weight;
          category_values_[site * rate_categories_ + category] = weighted;
          const Scalar combined = LogAdd(log_values_[site], weighted);
          const Scalar previous_weight =
              log_values_[site] == -std::numeric_limits<Scalar>::infinity()
                  ? Scalar{0}
                  : std::exp(log_values_[site] - combined);
          const Scalar category_weight = std::exp(weighted - combined);
          for (std::size_t index = 0; index < node_stride; ++index) {
            node_values_[site * node_stride + index] =
                previous_weight * node_values_[site * node_stride + index] +
                category_weight * result.nodes[index];
          }
          for (std::size_t index = 0; index < edge_stride; ++index) {
            edge_values_[site * edge_stride + index] =
                previous_weight * edge_values_[site * edge_stride + index] +
                category_weight * result.edges[index];
          }
          log_values_[site] = combined;
        }
      }
    }
    if (task_ == InferenceTask::kMarginals) {
      for (std::size_t site = 0; site < sites_; ++site) {
        for (std::size_t category = 0; category < rate_categories_;
             ++category) {
          category_values_[site * rate_categories_ + category] =
              std::exp(category_values_[site * rate_categories_ + category] -
                       log_values_[site]);
        }
      }
    }
  }

  std::span<const Scalar> log_values() const { return log_values_; }
  std::span<const Scalar> node_values() const { return node_values_; }
  std::span<const Scalar> edge_values() const { return edge_values_; }
  std::span<const std::size_t> states() const { return states_; }
  std::span<const Scalar> category_values() const { return category_values_; }
  std::span<const std::size_t> selected_categories() const {
    return selected_categories_;
  }

private:
  static Scalar LogAdd(Scalar first, Scalar second) {
    if (first == -std::numeric_limits<Scalar>::infinity())
      return second;
    if (second == -std::numeric_limits<Scalar>::infinity())
      return first;
    const Scalar maximum = std::max(first, second);
    return maximum +
           std::log(std::exp(first - maximum) + std::exp(second - maximum));
  }

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
        rate_categories_ != RateCategoryCount(model.rate_mixture) ||
        observation_nodes_.size() != model.observation_nodes.size() ||
        !std::equal(observation_nodes_.begin(), observation_nodes_.end(),
                    model.observation_nodes.begin())) {
      throw std::invalid_argument(
          "CPU task reference requires Reserve for this alignment shape");
    }
    const std::size_t expected_uniforms =
        task_ == InferenceTask::kSampling
            ? CheckedProduct(
                  {sites_, model.plan.num_nodes() + (rate_categories_ > 1
                                                         ? std::size_t{1}
                                                         : std::size_t{0})},
                  "CPU posterior-sampling variates")
            : 0;
    if (uniforms.size() != expected_uniforms) {
      throw std::invalid_argument(
          task_ == InferenceTask::kSampling
              ? "posterior sampling requires node variates and, for a rate "
                "mixture, one additional category variate per site"
              : "uniform variates apply only to posterior sampling");
    }
  }

  void PrepareSharedFactors(AlignmentModelView model) {
    const std::size_t edge_stride = model.plan.num_edges() * 16;
    for (std::size_t category = 0; category < rate_categories_; ++category) {
      const AlignmentModelView category_model =
          SelectRateCategory(model, category);
      for (std::size_t edge = 0; edge < model.plan.num_edges(); ++edge) {
        const auto transition = NucleotideTransition(
            category_model.nucleotide_model, model.branch_lengths[edge]);
        std::copy(transition.begin(), transition.end(),
                  edges_.begin() + category * edge_stride + edge * 16);
      }
    }

    const std::size_t node_stride = model.plan.num_nodes() * 4;
    for (std::size_t site = 0; site < sites_; ++site) {
      Scalar *root = nodes_.data() + site * node_stride + model.plan.root() * 4;
      std::transform(model.nucleotide_model.root_frequencies.begin(),
                     model.nucleotide_model.root_frequencies.end(), root,
                     [](double value) { return static_cast<Scalar>(value); });
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
  std::size_t rate_categories_ = 1;
  InferenceTask task_ = InferenceTask::kLikelihood;
  std::vector<btrc::Index> observation_nodes_;
  std::vector<Scalar> nodes_;
  std::vector<Scalar> edges_;
  std::vector<Scalar> log_values_;
  std::vector<Scalar> node_values_;
  std::vector<Scalar> edge_values_;
  std::vector<std::size_t> states_;
  std::vector<Scalar> category_values_;
  std::vector<std::size_t> selected_categories_;
  tree_hmm::Workspace tree_hmm_;
};

} // namespace parallel_phylogenetics::benchmark

#endif // PARALLEL_PHYLOGENETICS_CPU_TASK_REFERENCE_H_
