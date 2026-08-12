#include "parallel_phylogenetics/cuda.h"
#include "tests/accelerator_test.h"

#include <iostream>

int main() {
  if (!parallel_phylogenetics::cuda::Available()) {
    std::cout << "no CUDA device is available; skipping\n";
    return 0;
  }
  parallel_phylogenetics::cuda::Workspace workspace;
  parallel_phylogenetics::test::TestAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.Reserve(model, batch);
      },
      parallel_phylogenetics::cuda::LogLikelihoodsPrepared);
  parallel_phylogenetics::test::TestRecoveryAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveMaximum(model, batch);
      },
      parallel_phylogenetics::cuda::MaximumAPosterioriPrepared,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveSampling(model, batch);
      },
      parallel_phylogenetics::cuda::PosteriorSamplePrepared,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveMarginals(model, batch);
      },
      parallel_phylogenetics::cuda::PosteriorMarginalsPrepared);
  return 0;
}
