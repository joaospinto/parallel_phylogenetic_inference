#include "parallel_phylogenetics/metal.h"
#include "tests/accelerator_test.h"

#include <iostream>

int main() {
  if (!parallel_phylogenetics::metal::Available()) {
    std::cout << "no Metal device is available; skipping\n";
    return 0;
  }
  parallel_phylogenetics::metal::Workspace workspace;
  parallel_phylogenetics::test::TestAccelerator(
      workspace,
      [&](parallel_phylogenetics::AlignmentModelView model, std::size_t batch) {
        workspace.Reserve(model, batch);
      },
      parallel_phylogenetics::metal::LogLikelihoodsPrepared);
  return 0;
}
