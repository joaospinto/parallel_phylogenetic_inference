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
  return 0;
}
