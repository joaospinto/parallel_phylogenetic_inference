#include "benchmark.h"

#include "tree_hmm/metal.h"

#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics;
    using namespace parallel_phylogenetics::benchmark;
    if (!tree_hmm::metal::Available()) {
      std::cerr << "no Metal device is available\n";
      return 2;
    }
    const Options options = ParseOptions(argc, argv);
    const Problem problem = MakeProblem(options);
    const AlignmentModelView model{
        problem.plan, problem.sites, problem.branch_lengths,
        problem.observation_nodes, problem.observations};
    const std::size_t site_batch =
        options.site_batch == 0 ? problem.sites
                                : std::min(options.site_batch, problem.sites);
    tree_hmm::metal::Workspace full_workspace;
    full_workspace.Reserve(problem.plan, 4, site_batch);
    const BenchmarkResult result =
        site_batch == problem.sites
            ? RunInterleaved(
                  model, options.repeats, full_workspace.Inputs(),
                  [&](tree_hmm::BatchedModelView factors) {
                    return tree_hmm::metal::LogPartitionFunctionPrepared(
                        factors, full_workspace);
                  })
            : RunChunkedInterleaved(
                  model, options.repeats, site_batch, full_workspace.Inputs(),
                  [&](tree_hmm::BatchedModelView factors) {
                    return tree_hmm::metal::LogPartitionFunctionPrepared(
                        factors, full_workspace);
                  });
    PrintHeader("metal", tree_hmm::metal::DeviceDescription(), problem);
    PrintRow("metal", options, problem, result.cpu_ms, result.prepare_ms,
             result.accelerator_timings, result.total_accelerator_ms,
             result.cpu_values, result.accelerator_values,
             MaxAbsoluteError(result.cpu_values, result.accelerator_values),
             MaxRelativeError(result.cpu_values, result.accelerator_values));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
