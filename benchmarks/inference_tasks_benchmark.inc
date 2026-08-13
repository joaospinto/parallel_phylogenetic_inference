#include "benchmark.h"

#include <cstddef>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace parallel_phylogenetics::benchmark {
namespace {

template <class Reserve, class Evaluate>
void TimeTask(const char *backend, const char *task, const Options &options,
              const Problem &problem, Reserve &&reserve, Evaluate &&evaluate) {
  typename TASK_NAMESPACE::Workspace workspace;
  const AlignmentModelView model{
      problem.plan, problem.sites, problem.branch_lengths,
      problem.observation_nodes, problem.observations};
  const Clock::time_point planning_begin = Clock::now();
  reserve(workspace, model);
  const double workspace_setup_ms = Milliseconds(planning_begin, Clock::now());
  evaluate(workspace, model);
  std::vector<double> samples;
  samples.reserve(options.repeats);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    const Clock::time_point begin = Clock::now();
    evaluate(workspace, model);
    samples.push_back(Milliseconds(begin, Clock::now()));
  }
  const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
  std::cout << "# task_timing=host wall time of a preallocated public prepared "
               "call; input preparation and device transfers are included\n"
            << "backend,precision,task,dataset,topology,seed_base,seed,replicate,"
               "leaves,nodes,unique_patterns,tree_height,normalized_colless,"
               "primitive_levels,planning_ms,workspace_setup_ms,repeats,"
               "median_ms,p25_ms,p75_ms,samples_ms\n"
            << std::setprecision(10) << backend << ','
            << tree_hmm::kPrecisionName << ',' << task << ','
            << problem.dataset << ',' << problem.topology << ','
            << problem.base_seed << ',' << problem.seed << ','
            << problem.replicate << ',' << problem.leaves << ','
            << problem.plan.num_nodes() << ',' << problem.sites << ','
            << problem.shape.height << ','
            << problem.shape.normalized_colless << ','
            << statistics.primitive_levels << ',' << problem.planning_ms << ','
            << workspace_setup_ms << ',' << options.repeats << ','
            << Median(samples) << ',' << Quantile(samples, 0.25) << ','
            << Quantile(samples, 0.75) << ',' << JoinSamples(samples) << '\n';
}

void RunOne(const char *backend, const Options &options,
            std::size_t replicate) {
  const Problem problem = MakeProblem(options, replicate);
  TimeTask(
      backend, "likelihood", options, problem,
      [](auto &workspace, AlignmentModelView model) { workspace.Reserve(model); },
      [](auto &workspace, AlignmentModelView model) {
        static_cast<void>(TASK_NAMESPACE::LogLikelihoodsPrepared(model,
                                                                 workspace));
      });
  TimeTask(
      backend, "joint-map", options, problem,
      [](auto &workspace, AlignmentModelView model) {
        workspace.ReserveMaximum(model);
      },
      [](auto &workspace, AlignmentModelView model) {
        static_cast<void>(
            TASK_NAMESPACE::MaximumAPosterioriPrepared(model, workspace));
      });

  std::vector<Scalar> uniforms(problem.sites * problem.plan.num_nodes(),
                               Scalar{0.5});
  TimeTask(
      backend, "posterior-sample", options, problem,
      [](auto &workspace, AlignmentModelView model) {
        workspace.ReserveSampling(model);
      },
      [&](auto &workspace, AlignmentModelView model) {
        static_cast<void>(TASK_NAMESPACE::PosteriorSamplePrepared(
            model, std::span<const Scalar>(uniforms), workspace));
      });
  TimeTask(
      backend, "all-marginals", options, problem,
      [](auto &workspace, AlignmentModelView model) {
        workspace.ReserveMarginals(model);
      },
      [](auto &workspace, AlignmentModelView model) {
        static_cast<void>(
            TASK_NAMESPACE::PosteriorMarginalsPrepared(model, workspace));
      });
}

} // namespace
} // namespace parallel_phylogenetics::benchmark

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics::benchmark;
    const Options options = ParseOptions(argc, argv);
    const std::size_t stop = options.replicate_start + options.replicates;
    if (stop < options.replicate_start)
      throw std::overflow_error("synthetic replicate range overflows");
    for (std::size_t replicate = options.replicate_start; replicate < stop;
         ++replicate) {
      RunOne(TASK_BACKEND_NAME, options, replicate);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
