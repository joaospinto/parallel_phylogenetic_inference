#include "benchmarks/cpu_task_reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckImpl(bool condition, int line) {
  if (!condition)
    throw std::runtime_error("CPU task reference test failed at line " +
                             std::to_string(line));
}

#define Check(condition) CheckImpl((condition), __LINE__)

bool Near(parallel_phylogenetics::Scalar left,
          parallel_phylogenetics::Scalar right) {
  const auto scale = std::max(
      {parallel_phylogenetics::Scalar{1}, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
         parallel_phylogenetics::Scalar{64} *
             std::numeric_limits<parallel_phylogenetics::Scalar>::epsilon() *
             scale;
}

parallel_phylogenetics::Scalar
AssignmentLogWeight(parallel_phylogenetics::SiteModelView model,
                    const std::array<std::size_t, 3> &states) {
  using parallel_phylogenetics::AllowsState;
  using parallel_phylogenetics::NucleotideTransition;
  using parallel_phylogenetics::Scalar;
  Scalar result = static_cast<Scalar>(
      std::log(model.nucleotide_model.root_frequencies[states[0]]));
  for (std::size_t node = 0; node < states.size(); ++node) {
    if (!AllowsState(model.observations[node], states[node]))
      return -std::numeric_limits<Scalar>::infinity();
  }
  for (std::size_t edge = 0; edge < model.plan.num_edges(); ++edge) {
    const auto transition = NucleotideTransition(model.nucleotide_model,
                                                 model.branch_lengths[edge]);
    const std::size_t parent = model.plan.edge_parents()[edge];
    const std::size_t child = model.plan.edge_children()[edge];
    result += std::log(transition[states[parent] * 4 + states[child]]);
  }
  return result;
}

} // namespace

int main() {
  using namespace parallel_phylogenetics;
  using namespace parallel_phylogenetics::benchmark;
  const std::vector<std::int64_t> parents{-1, 0, 0};
  const btrc::Plan plan = btrc::MakePlan(parents);
  std::vector<Scalar> lengths{Scalar{0.1}, Scalar{0.2}};
  const std::vector<btrc::Index> observation_nodes{1, 2};
  const std::vector<Nucleotide> observations{Nucleotide::kA, Nucleotide::kC,
                                             Nucleotide::kG, Nucleotide::kT,
                                             Nucleotide::kR, Nucleotide::kY};
  const NucleotideModel nucleotide_model =
      HasegawaKishinoYanoModel({0.31, 0.19, 0.27, 0.23}, 4.0, 0.8);
  const RateMixture gamma = DiscreteGammaRateMixture(0.7, 3);
  const AlignmentModelView model{plan,         3,
                                 lengths,      observation_nodes,
                                 observations, nucleotide_model,
                                 gamma.view()};
  std::vector<Scalar> uniforms(model.sites * (plan.num_nodes() + 1),
                               Scalar{0.5});

  for (const InferenceTask task : {
           InferenceTask::kLikelihood,
           InferenceTask::kMaximum,
           InferenceTask::kSampling,
           InferenceTask::kMarginals,
       }) {
    CpuTaskReference reference;
    reference.Reserve(model, task);
    const std::span<const Scalar> task_uniforms =
        task == InferenceTask::kSampling ? std::span<const Scalar>(uniforms)
                                         : std::span<const Scalar>();
    reference.Evaluate(model, InputUpdate::kAll, task_uniforms);
    const std::vector<Scalar> initial_logs(reference.log_values().begin(),
                                           reference.log_values().end());
    const std::vector<Scalar> initial_nodes(reference.node_values().begin(),
                                            reference.node_values().end());
    const std::vector<Scalar> initial_edges(reference.edge_values().begin(),
                                            reference.edge_values().end());
    const std::vector<std::size_t> initial_states(reference.states().begin(),
                                                  reference.states().end());
    reference.Evaluate(model, InputUpdate::kNone, task_uniforms);
    Check(std::equal(initial_logs.begin(), initial_logs.end(),
                     reference.log_values().begin()));
    Check(std::equal(initial_nodes.begin(), initial_nodes.end(),
                     reference.node_values().begin()));
    Check(std::equal(initial_edges.begin(), initial_edges.end(),
                     reference.edge_values().begin()));
    Check(std::equal(initial_states.begin(), initial_states.end(),
                     reference.states().begin()));
    reference.Evaluate(model, InputUpdate::kFactors, task_uniforms);
  }

  CpuTaskReference likelihood;
  likelihood.Reserve(model, InferenceTask::kLikelihood);
  likelihood.Evaluate(model, InputUpdate::kAll);
  for (std::size_t site = 0; site < model.sites; ++site) {
    const std::vector<Nucleotide> node_observations{Nucleotide::kUnknown,
                                                    observations[site * 2],
                                                    observations[site * 2 + 1]};
    const Scalar expected =
        SiteLogLikelihood({plan, lengths, node_observations,
                           model.nucleotide_model, model.rate_mixture});
    Check(Near(expected, likelihood.log_values()[site]));
  }

  CpuTaskReference maximum;
  maximum.Reserve(model, InferenceTask::kMaximum);
  maximum.Evaluate(model, InputUpdate::kAll);
  CpuTaskReference marginals;
  marginals.Reserve(model, InferenceTask::kMarginals);
  marginals.Evaluate(model, InputUpdate::kAll);
  CpuTaskReference sampling;
  sampling.Reserve(model, InferenceTask::kSampling);
  sampling.Evaluate(model, InputUpdate::kAll, uniforms);
  for (std::size_t site = 0; site < model.sites; ++site) {
    const std::vector<Nucleotide> node_observations{Nucleotide::kUnknown,
                                                    observations[site * 2],
                                                    observations[site * 2 + 1]};
    const SiteModelView site_model{plan, lengths, node_observations,
                                   nucleotide_model, gamma.view()};
    const SitePosterior expected_posterior = AncestralPosterior(site_model);
    Check(Near(std::log(expected_posterior.likelihood),
               marginals.log_values()[site]));
    for (std::size_t index = 0; index < plan.num_nodes() * 4; ++index) {
      Check(Near(expected_posterior.ancestral_states[index],
                 marginals.node_values()[site * plan.num_nodes() * 4 + index]));
    }
    for (std::size_t index = 0; index < plan.num_edges() * 16; ++index) {
      Check(
          Near(expected_posterior.substitutions[index],
               marginals.edge_values()[site * plan.num_edges() * 16 + index]));
    }
    for (std::size_t category = 0; category < gamma.rates().size();
         ++category) {
      Check(Near(
          expected_posterior.rate_categories[category],
          marginals.category_values()[site * gamma.rates().size() + category]));
    }

    Scalar expected_maximum = -std::numeric_limits<Scalar>::infinity();
    std::size_t expected_category = 0;
    std::array<std::size_t, 3> expected_states{};
    for (std::size_t category = 0; category < gamma.rates().size();
         ++category) {
      const SiteModelView category_model =
          SelectRateCategory(site_model, category);
      for (std::size_t code = 0; code < 64; ++code) {
        const std::array<std::size_t, 3> states{code / 16, (code / 4) % 4,
                                                code % 4};
        const Scalar candidate =
            AssignmentLogWeight(category_model, states) +
            static_cast<Scalar>(std::log(gamma.weights()[category]));
        if (candidate > expected_maximum) {
          expected_maximum = candidate;
          expected_category = category;
          expected_states = states;
        }
      }
    }
    Check(Near(expected_maximum, maximum.log_values()[site]));
    Check(expected_category == maximum.selected_categories()[site]);
    Check(std::equal(expected_states.begin(), expected_states.end(),
                     maximum.states().begin() + site * plan.num_nodes()));

    Scalar cumulative = 0.0;
    std::size_t sampled_category = gamma.rates().size() - 1;
    for (std::size_t category = 0; category + 1 < gamma.rates().size();
         ++category) {
      cumulative += expected_posterior.rate_categories[category];
      if (Scalar{0.5} < cumulative) {
        sampled_category = category;
        break;
      }
    }
    Check(sampled_category == sampling.selected_categories()[site]);
  }
}
