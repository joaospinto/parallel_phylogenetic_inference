#include "tests/accelerator_test.h"

#include <iostream>

int main() {
  if (!parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
          Available()) {
    std::cout << "no " PARALLEL_PHYLOGENETICS_GPU_NAME
                 " device is available; skipping\n";
    return 0;
  }
  parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::Workspace
      workspace;
  parallel_phylogenetics::test::TestAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.Reserve(model, batch);
      },
      parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
          LogLikelihoodsPrepared);
  parallel_phylogenetics::test::TestRecoveryAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveMaximum(model, batch);
      },
      parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
          MaximumAPosterioriPrepared,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveSampling(model, batch);
      },
      parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
          PosteriorSamplePrepared,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveMarginals(model, batch);
      },
      parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
          PosteriorMarginalsPrepared);
  return 0;
}
