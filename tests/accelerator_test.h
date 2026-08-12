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
  const std::vector<double> lengths{0.1, 0.3, 0.2, 0.4, 0.15, 0.5};
  const std::vector<btrc::Index> observation_nodes{4, 5, 6};
  constexpr std::size_t kSites = 5;
  using N = Nucleotide;
  const std::vector<N> observations{
      N::kR, N::kG, N::kT, N::kC, N::kC, N::kA, N::kUnknown, N::kY,
      N::kB, N::kV, N::kW, N::kD, N::kH, N::kM, N::kS,
  };
  const std::array<double, 4> frequencies{0.3, 0.2, 0.2, 0.3};
  const AlignmentModelView model{
      plan, kSites, lengths, observation_nodes, observations, frequencies, 1.0};

  SequentialWorkspace sequential_workspace;
  sequential_workspace.Reserve(plan, kSites);
  const std::span<const double> expected =
      LogLikelihoodsPrepared(model, sequential_workspace);

  reserve(model, 2);
  for (int repeat = 0; repeat < 2; ++repeat) {
    const std::span<const float> actual = evaluate(model, workspace);
    Check(actual.size() == kSites,
          "phylogenetic accelerator returned a wrong site count");
    for (std::size_t site = 0; site < kSites; ++site) {
      const double error =
          std::abs(static_cast<double>(actual[site]) - expected[site]);
      Check(error <= 2e-5 * std::max(1.0, std::abs(expected[site])),
            "phylogenetic accelerator likelihood is inaccurate");
    }
  }
}

} // namespace parallel_phylogenetics::test

#endif // PARALLEL_PHYLOGENETICS_ACCELERATOR_TEST_H_
