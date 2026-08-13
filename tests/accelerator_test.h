#ifndef PARALLEL_PHYLOGENETICS_ACCELERATOR_TEST_H_
#define PARALLEL_PHYLOGENETICS_ACCELERATOR_TEST_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "parallel_phylogenetics/alignment.h"
#include "tree_hmm/inference.h"

namespace parallel_phylogenetics::test {

inline void Check(bool condition, const char *description) {
  if (!condition)
    throw std::runtime_error(description);
}

template <class Workspace, class Reserve, class Evaluate>
void TestAccelerator(Workspace &workspace, Reserve &&reserve,
                     Evaluate &&evaluate) {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 1, 3, 2});
  const std::vector<Scalar> lengths{0.1, 0.3, 0.2, 0.4, 0.15, 0.5};
  const std::vector<btrc::Index> observation_nodes{4, 5, 6};
  constexpr std::size_t kSites = 5;
  using N = Nucleotide;
  const std::vector<N> observations{
      N::kR, N::kG, N::kT, N::kC, N::kC, N::kA, N::kUnknown, N::kY,
      N::kB, N::kV, N::kW, N::kD, N::kH, N::kM, N::kS,
  };
  const std::array<Scalar, 4> frequencies{0.3, 0.2, 0.2, 0.3};
  const AlignmentModelView model{
      plan, kSites, lengths, observation_nodes, observations, frequencies, 1.0};

  SequentialWorkspace sequential_workspace;
  sequential_workspace.Reserve(plan, kSites);
  const std::span<const Scalar> expected =
      LogLikelihoodsPrepared(model, sequential_workspace);

  reserve(model, 2);
  for (int repeat = 0; repeat < 2; ++repeat) {
    const std::span<const Scalar> actual = evaluate(model, workspace);
    Check(actual.size() == kSites,
          "phylogenetic accelerator returned a wrong site count");
    for (std::size_t site = 0; site < kSites; ++site) {
      const double error =
          std::abs(static_cast<double>(actual[site]) - expected[site]);
      Check(error <= 2e-5 * std::max(1.0, std::abs(static_cast<double>(
                                              expected[site]))),
            "phylogenetic accelerator likelihood is inaccurate");
    }
  }
}

template <class Workspace, class Reserve, class Evaluate>
void TestResidentAccelerator(Workspace &workspace, Reserve &&reserve,
                             Evaluate &&evaluate) {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 1, 2, 2});
  std::vector<Scalar> lengths{0.1, 0.3, 0.2, 0.4, 0.15, 0.5};
  const std::vector<btrc::Index> observation_nodes{3, 4, 5, 6};
  constexpr std::size_t kSites = 5;
  using N = Nucleotide;
  std::vector<N> observations{
      N::kA, N::kC, N::kG, N::kT, N::kC, N::kG, N::kT,
      N::kA, N::kG, N::kT, N::kA, N::kC, N::kT, N::kA,
      N::kC, N::kG, N::kA, N::kG, N::kC, N::kT,
  };
  const AlignmentModelView model{plan, kSites, lengths, observation_nodes,
                                 observations};
  reserve(model, kSites);

  for (const InputUpdate update : {InputUpdate::kFactors,
                                   InputUpdate::kNone}) {
    bool rejected = false;
    try {
      static_cast<void>(evaluate(model, workspace, update));
    } catch (const std::logic_error &) {
      rejected = true;
    }
    Check(rejected, "phylogenetic accelerator accepted unstaged inputs");
  }

  const std::span<const Scalar> initial_view =
      evaluate(model, workspace, InputUpdate::kAll);
  const std::vector<Scalar> initial(initial_view.begin(), initial_view.end());
  observations.front() = N::kT;
  const std::span<const Scalar> observations_reused =
      evaluate(model, workspace, InputUpdate::kFactors);
  Check(std::equal(observations_reused.begin(), observations_reused.end(),
                   initial.begin()),
        "phylogenetic accelerator did not reuse observations");
  const std::span<const Scalar> observations_updated =
      evaluate(model, workspace, InputUpdate::kAll);
  const std::vector<Scalar> after_observation_update(observations_updated.begin(),
                                                     observations_updated.end());
  Check(!std::equal(after_observation_update.begin(),
                    after_observation_update.end(), initial.begin()),
        "phylogenetic accelerator did not update observations");

  lengths.front() *= 1.7f;
  const std::span<const Scalar> factors_reused =
      evaluate(model, workspace, InputUpdate::kNone);
  Check(std::equal(factors_reused.begin(), factors_reused.end(),
                   after_observation_update.begin()),
        "phylogenetic accelerator did not reuse factors");
  const std::span<const Scalar> factors_updated =
      evaluate(model, workspace, InputUpdate::kFactors);
  Check(!std::equal(factors_updated.begin(), factors_updated.end(),
                    after_observation_update.begin()),
        "phylogenetic accelerator did not update factors");

  const PreparedTimings timings = workspace.LastTimings();
  Check(timings.site_batches == 1,
        "resident phylogenetic evaluation used multiple site batches");
  Check(timings.backend.upload_ms >= 0.0 && timings.backend.kernel_ms >= 0.0 &&
            timings.backend.download_ms >= 0.0 &&
            timings.backend.wall_ms >= 0.0 &&
            timings.evaluation_wall_ms >= timings.backend.wall_ms,
        "phylogenetic accelerator timings are invalid");

  reserve(model, 2);
  bool chunked_reuse_rejected = false;
  try {
    static_cast<void>(evaluate(model, workspace, InputUpdate::kFactors));
  } catch (const std::invalid_argument &) {
    chunked_reuse_rejected = true;
  }
  Check(chunked_reuse_rejected,
        "phylogenetic accelerator treated a chunked alignment as resident");
}

template <class Workspace, class ReserveMaximum, class Maximum,
          class ReserveSampling, class Sample, class ReserveMarginals,
          class Marginals>
void TestRecoveryAccelerator(
    Workspace &workspace, ReserveMaximum &&reserve_maximum, Maximum &&maximum,
    ReserveSampling &&reserve_sampling, Sample &&sample,
    ReserveMarginals &&reserve_marginals, Marginals &&marginals) {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 1, 3, 2});
  const std::vector<Scalar> lengths{0.1, 0.3, 0.2, 0.4, 0.15, 0.5};
  const std::vector<btrc::Index> observation_nodes{4, 5, 6};
  constexpr std::size_t kSites = 5;
  using N = Nucleotide;
  const std::vector<N> observations{
      N::kR, N::kG, N::kT, N::kC, N::kC, N::kA, N::kUnknown, N::kY,
      N::kB, N::kV, N::kW, N::kD, N::kH, N::kM, N::kS,
  };
  const std::array<Scalar, 4> frequencies{0.3, 0.2, 0.2, 0.3};
  const AlignmentModelView full{
      plan, kSites, lengths, observation_nodes, observations, frequencies, 1.0};
  const AlignmentModelView model = SelectSites(full, 1, 2);

  AlignmentWorkspace factor_workspace;
  factor_workspace.Reserve(plan, model.sites);
  const tree_hmm::BatchedModelView factors = Prepare(model, factor_workspace);
  tree_hmm::Workspace cpu_workspace;
  cpu_workspace.Reserve(plan, 4);

  reserve_maximum(full, 2);
  const AlignmentMaximumView actual_maximum = maximum(model, workspace);
  Check(actual_maximum.log_weights.size() == model.sites,
        "phylogenetic accelerator returned a wrong MAP weight shape");
  Check(actual_maximum.states.size() == model.sites * plan.num_nodes(),
        "phylogenetic accelerator returned a wrong MAP state shape");
  for (std::size_t site = 0; site < model.sites; ++site) {
    const std::size_t node_values = plan.num_nodes() * 4;
    const std::vector<Scalar> nodes(
        factors.node_potentials.begin() + site * node_values,
        factors.node_potentials.begin() + (site + 1) * node_values);
    const std::vector<Scalar> edges(factors.edge_potentials.begin(),
                                    factors.edge_potentials.end());
    const tree_hmm::MaximumAssignmentView expected =
        tree_hmm::MaximumAPosterioriPrepared({plan, 4, nodes, edges},
                                             cpu_workspace);
    Check(std::abs(actual_maximum.log_weights[site] - expected.log_weight) <
              2e-5,
          "phylogenetic accelerator MAP weight is inaccurate");
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      Check(actual_maximum.states[site * plan.num_nodes() + node] ==
                expected.states[node],
            "phylogenetic accelerator MAP state is inaccurate");
    }
  }

  std::vector<Scalar> uniforms(model.sites * plan.num_nodes());
  for (std::size_t index = 0; index < uniforms.size(); ++index)
    uniforms[index] = 0.07f + 0.11f * static_cast<float>(index % 8);
  reserve_sampling(full, 2);
  const AlignmentPosteriorSampleView actual_sample =
      sample(model, uniforms, workspace);
  Check(actual_sample.states.size() == model.sites * plan.num_nodes(),
        "phylogenetic accelerator returned a wrong sample shape");
  for (std::size_t site = 0; site < model.sites; ++site) {
    const std::size_t node_values = plan.num_nodes() * 4;
    const std::vector<Scalar> nodes(
        factors.node_potentials.begin() + site * node_values,
        factors.node_potentials.begin() + (site + 1) * node_values);
    const std::vector<Scalar> edges(factors.edge_potentials.begin(),
                                    factors.edge_potentials.end());
    const std::vector<Scalar> site_uniforms(
        uniforms.begin() + site * plan.num_nodes(),
        uniforms.begin() + (site + 1) * plan.num_nodes());
    const std::span<const std::size_t> expected =
        tree_hmm::PosteriorSamplePrepared({plan, 4, nodes, edges},
                                          site_uniforms, cpu_workspace);
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      Check(actual_sample.states[site * plan.num_nodes() + node] ==
                expected[node],
            "phylogenetic accelerator posterior sample is inaccurate");
    }
  }

  reserve_marginals(full, 2);
  const AlignmentPosteriorView actual_marginals = marginals(model, workspace);
  const std::size_t nodes_per_site = plan.num_nodes() * 4;
  const std::size_t edges_per_site = plan.num_edges() * 16;
  Check(actual_marginals.log_likelihoods.size() == model.sites,
        "phylogenetic accelerator returned a wrong likelihood shape");
  Check(actual_marginals.ancestral_states.size() ==
            model.sites * nodes_per_site,
        "phylogenetic accelerator returned a wrong node-marginal shape");
  Check(actual_marginals.substitutions.size() == model.sites * edges_per_site,
        "phylogenetic accelerator returned a wrong edge-marginal shape");
  for (std::size_t site = 0; site < model.sites; ++site) {
    const std::vector<Scalar> nodes(
        factors.node_potentials.begin() + site * nodes_per_site,
        factors.node_potentials.begin() + (site + 1) * nodes_per_site);
    const std::vector<Scalar> edges(factors.edge_potentials.begin(),
                                    factors.edge_potentials.end());
    const tree_hmm::MarginalView expected =
        tree_hmm::PosteriorMarginalsPrepared({plan, 4, nodes, edges},
                                             cpu_workspace);
    Check(std::abs(actual_marginals.log_likelihoods[site] -
                   expected.log_partition) < 2e-5,
          "phylogenetic accelerator marginal likelihood is inaccurate");
    for (std::size_t index = 0; index < nodes_per_site; ++index) {
      Check(
          std::abs(
              actual_marginals.ancestral_states[site * nodes_per_site + index] -
              expected.nodes[index]) < 8e-5,
          "phylogenetic accelerator node marginal is inaccurate");
    }
    for (std::size_t index = 0; index < edges_per_site; ++index) {
      Check(std::abs(
                actual_marginals.substitutions[site * edges_per_site + index] -
                expected.edges[index]) < 8e-5,
            "phylogenetic accelerator edge marginal is inaccurate");
    }
  }

  bool oversized_batch_rejected = false;
  try {
    static_cast<void>(marginals(SelectSites(full, 0, 3), workspace));
  } catch (const std::invalid_argument &) {
    oversized_batch_rejected = true;
  }
  Check(oversized_batch_rejected,
        "phylogenetic accelerator accepted a recovery batch over capacity");
}

} // namespace parallel_phylogenetics::test

#endif // PARALLEL_PHYLOGENETICS_ACCELERATOR_TEST_H_
