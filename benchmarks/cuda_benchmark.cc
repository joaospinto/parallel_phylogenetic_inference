#include "benchmark.h"

#include "tree_hmm/cuda.h"

#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics;
    using namespace parallel_phylogenetics::benchmark;
    if (!tree_hmm::cuda::Available()) {
      std::cerr << "no CUDA device is available\n";
      return 2;
    }
    const Options options = ParseOptions(argc, argv);
    const Problem problem = MakeProblem(options);
    const AlignmentModelView model{problem.plan, problem.sites,
                                   problem.branch_lengths,
                                   problem.observations};
    tree_hmm::cuda::Workspace device_workspace;
    device_workspace.Reserve(problem.plan, 4, problem.sites);
    const BenchmarkResult result =
        RunInterleaved(model, options.repeats, device_workspace.Inputs(),
                       [&](tree_hmm::BatchedModelView factors) {
                         return tree_hmm::cuda::LogPartitionFunctionPrepared(
                             factors, device_workspace);
                       });
    PrintHeader("cuda", tree_hmm::cuda::DeviceDescription(), problem);
    PrintRow("cuda", options, problem, result.cpu_ms, result.prepare_ms,
             result.accelerator, result.total_accelerator_ms,
             MaxAbsoluteError(result.cpu_values, result.accelerator.values),
             MaxRelativeError(result.cpu_values, result.accelerator.values));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
