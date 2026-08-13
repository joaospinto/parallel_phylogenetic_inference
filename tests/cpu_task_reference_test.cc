#include "benchmarks/cpu_task_reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  const AlignmentModelView model{plan, 3, lengths, observation_nodes,
                                 observations};
  std::vector<Scalar> uniforms(model.sites * plan.num_nodes(), Scalar{0.5});

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
                           model.root_frequencies, model.substitution_rate});
    Check(Near(expected, likelihood.log_values()[site]));
  }
}
