#include "benchmark.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void RunOne(const parallel_phylogenetics::benchmark::Options &options,
            std::size_t replicate) {
  using namespace parallel_phylogenetics;
  using namespace parallel_phylogenetics::benchmark;
  const Problem problem = MakeProblem(options, replicate);
  const AlignmentModelView model{
      problem.plan, problem.sites, problem.branch_lengths,
      problem.observation_nodes, problem.observations};
  const std::size_t site_batch =
      options.site_batch == 0 ? problem.sites
                              : std::min(options.site_batch, problem.sites);
  tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::Workspace
      full_workspace;
  full_workspace.ReserveCategorical(problem.plan, 4, site_batch, 16,
                                    problem.observation_nodes);
  const BenchmarkResult result =
      site_batch == problem.sites
          ? RunInterleaved(
                model, options.repeats, options.conditioning_ms,
                full_workspace.CategoricalInputs(),
                [&](tree_hmm::BatchedCategoricalModelView factors) {
                  return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
                      LogPartitionFunctionPrepared(factors, full_workspace);
                })
          : RunChunkedInterleaved(
                model, options.repeats, options.conditioning_ms, site_batch,
                full_workspace.CategoricalInputs(),
                [&](tree_hmm::BatchedCategoricalModelView factors) {
                  return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
                      LogPartitionFunctionPrepared(factors, full_workspace);
                });
  PrintHeader(PARALLEL_PHYLOGENETICS_BACKEND_NAME,
              tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
                  DeviceDescription(),
              options, problem);
  PrintRow(PARALLEL_PHYLOGENETICS_BACKEND_NAME, options, problem, result,
           MaxAbsoluteError(result.cpu_values, result.accelerator_values),
           MaxRelativeError(result.cpu_values, result.accelerator_values));
}

} // namespace

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics::benchmark;
    if (!tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::Available()) {
      std::cerr << "no " PARALLEL_PHYLOGENETICS_GPU_NAME
                   " device is available\n";
      return 2;
    }
    const Options options = ParseOptions(argc, argv);
    for (std::size_t replicate = options.replicate_start;
         replicate < options.replicate_start + options.replicates; ++replicate)
      RunOne(options, replicate);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
