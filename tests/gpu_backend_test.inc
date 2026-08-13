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
      [](parallel_phylogenetics::AlignmentModelView model, auto &workspace) {
        return parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
            LogLikelihoodsPrepared(model, workspace);
      });
  parallel_phylogenetics::test::TestResidentAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.Reserve(model, batch);
      },
      [](parallel_phylogenetics::AlignmentModelView model, auto &workspace,
         parallel_phylogenetics::InputUpdate update) {
        return parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
            LogLikelihoodsPrepared(model, workspace, update);
      });
  parallel_phylogenetics::test::TestRecoveryAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveMaximum(model, batch);
      },
      [](parallel_phylogenetics::AlignmentModelView model, auto &workspace) {
        return parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
            MaximumAPosterioriPrepared(model, workspace);
      },
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveSampling(model, batch);
      },
      [](parallel_phylogenetics::AlignmentModelView model,
         std::span<const parallel_phylogenetics::Scalar> uniforms,
         auto &workspace) {
        return parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
            PosteriorSamplePrepared(model, uniforms, workspace);
      },
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.ReserveMarginals(model, batch);
      },
      [](parallel_phylogenetics::AlignmentModelView model, auto &workspace) {
        return parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE::
            PosteriorMarginalsPrepared(model, workspace);
      });
  return 0;
}
