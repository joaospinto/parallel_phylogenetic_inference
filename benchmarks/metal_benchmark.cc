#include "benchmark.h"

#include "parallel_phylogenetics/metal.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void RunOne(const parallel_phylogenetics::benchmark::Options &options,
            std::size_t replicate) {
  using namespace parallel_phylogenetics;
  using namespace parallel_phylogenetics::benchmark;
  const Problem problem = MakeProblem(options, replicate);
  const BenchmarkEvolutionModel evolution =
      MakeBenchmarkEvolutionModel(options);
  const AlignmentModelView model{
      problem.plan,           problem.sites,
      problem.branch_lengths, problem.observation_nodes,
      problem.observations,   evolution.nucleotide,
      evolution.rate_view()};
  const std::size_t site_batch =
      options.site_batch == 0 ? problem.sites
                              : std::min(options.site_batch, problem.sites);
  metal::Workspace full_workspace;
  metal::Workspace tail_workspace;
  const BenchmarkResult result = RunInterleaved(
      model, options.repeats, options.conditioning_ms, site_batch,
      BenchmarkInputUpdate(options.benchmark_mode), full_workspace,
      tail_workspace,
      [](auto &workspace, AlignmentModelView view, std::size_t capacity) {
        workspace.Reserve(view, capacity);
      },
      [](AlignmentModelView chunk, auto &workspace, InputUpdate update) {
        return metal::LogLikelihoodsPrepared(chunk, workspace, update);
      });
  const double absolute_error =
      MaxAbsoluteError(result.cpu_values, result.accelerator_values);
  const double relative_error =
      MaxRelativeError(result.cpu_values, result.accelerator_values);
  RequireBenchmarkCorrectness(absolute_error, relative_error);
  PrintHeader("metal", metal::DeviceDescription(), options, problem);
  PrintRow("metal", options, problem, result, absolute_error, relative_error);
}

} // namespace

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics::benchmark;
    if (!parallel_phylogenetics::metal::Available()) {
      std::cerr << "no Metal device is available\n";
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
