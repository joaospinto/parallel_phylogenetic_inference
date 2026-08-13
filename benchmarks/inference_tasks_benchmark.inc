#include "benchmark.h"
#include "cpu_task_reference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace parallel_phylogenetics::benchmark {
namespace {

struct TaskErrors {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  std::size_t state_mismatches = 0;
};

TaskErrors ScalarErrors(std::span<const Scalar> expected,
                        std::span<const Scalar> actual) {
  return {MaxAbsoluteError(expected, actual),
          MaxRelativeError(expected, actual), 0};
}

void IncludeScalarErrors(TaskErrors &errors, std::span<const Scalar> expected,
                         std::span<const Scalar> actual) {
  const TaskErrors next = ScalarErrors(expected, actual);
  errors.maximum_absolute =
      std::max(errors.maximum_absolute, next.maximum_absolute);
  errors.maximum_relative =
      std::max(errors.maximum_relative, next.maximum_relative);
}

template <class Actual>
std::size_t CountStateMismatches(std::span<const std::size_t> expected,
                                 std::span<const Actual> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("task assignment shapes do not match");
  std::size_t result = 0;
  for (std::size_t index = 0; index < expected.size(); ++index)
    result += expected[index] != static_cast<std::size_t>(actual[index]);
  return result;
}

struct TaskTimings {
  std::vector<double> cpu;
  std::vector<double> accelerator;
  std::vector<double> upload;
  std::vector<double> kernel;
  std::vector<double> download;
};

template <class Reserve, class Evaluate, class Compare>
void TimeTask(const char *backend, const char *task, InferenceTask operation,
              const Options &options, const Problem &problem,
              std::span<const Scalar> uniforms, Reserve &&reserve,
              Evaluate &&evaluate, Compare &&compare) {
  const AlignmentModelView model{
      problem.plan, problem.sites, problem.branch_lengths,
      problem.observation_nodes, problem.observations};
  if (options.site_batch != 0 && options.site_batch < model.sites) {
    throw std::invalid_argument(
        "recovery-task benchmarks require one unchunked site batch");
  }

  CpuTaskReference cpu;
  const Clock::time_point cpu_setup_begin = Clock::now();
  cpu.Reserve(model, operation);
  const double cpu_workspace_setup_ms =
      Milliseconds(cpu_setup_begin, Clock::now());

  typename TASK_NAMESPACE::Workspace accelerator;
  const Clock::time_point accelerator_setup_begin = Clock::now();
  reserve(accelerator, model);
  const double accelerator_workspace_setup_ms =
      Milliseconds(accelerator_setup_begin, Clock::now());

  cpu.Evaluate(model, InputUpdate::kAll, uniforms);
  static_cast<void>(evaluate(accelerator, model, InputUpdate::kAll));
  const InputUpdate update = BenchmarkInputUpdate(options.benchmark_mode);

  TaskTimings timings;
  timings.cpu.reserve(options.repeats);
  timings.accelerator.reserve(options.repeats);
  timings.upload.reserve(options.repeats);
  timings.kernel.reserve(options.repeats);
  timings.download.reserve(options.repeats);
  const auto run_cpu = [&] {
    const Clock::time_point begin = Clock::now();
    cpu.Evaluate(model, update, uniforms);
    timings.cpu.push_back(Milliseconds(begin, Clock::now()));
  };
  const auto run_accelerator = [&] {
    const Clock::time_point begin = Clock::now();
    static_cast<void>(evaluate(accelerator, model, update));
    timings.accelerator.push_back(Milliseconds(begin, Clock::now()));
    const PreparedTimings result = accelerator.LastTimings();
    timings.upload.push_back(result.backend.upload_ms);
    timings.kernel.push_back(result.backend.kernel_ms);
    timings.download.push_back(result.backend.download_ms);
  };

  ConditionInterleaved(options.conditioning_ms, run_cpu, run_accelerator);
  timings = {};
  timings.cpu.reserve(options.repeats);
  timings.accelerator.reserve(options.repeats);
  timings.upload.reserve(options.repeats);
  timings.kernel.reserve(options.repeats);
  timings.download.reserve(options.repeats);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    if (repeat % 2 == 0) {
      run_cpu();
      run_accelerator();
    } else {
      run_accelerator();
      run_cpu();
    }
  }

  // One untimed matched call leaves both output views in a known state for the
  // correctness check without retaining views across later workspace calls.
  cpu.Evaluate(model, update, uniforms);
  const auto accelerator_result = evaluate(accelerator, model, update);
  const TaskErrors errors = compare(cpu, accelerator_result);

  const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
  const std::size_t primitive_operations =
      statistics.rakes + problem.plan.num_branch_combinations() +
      problem.plan.num_branch_absorptions() + statistics.compressions;
  const double cpu_ms = Median(timings.cpu);
  const double accelerator_ms = Median(timings.accelerator);
  std::cout
      << "# task_timing=host wall time of one preallocated public call; "
         "workspace allocation and warmup are excluded\n"
      << "# task_cpu_reference=the reusable scalar generic tree-HMM CPU "
         "algebra, evaluated site by site from preallocated dense factors\n"
      << "# task_input_update=full-input refreshes observations and factors; "
         "factor-update refreshes numerical factors; fixed-model retains "
         "both. The generic CPU prepared call still copies each dense site "
         "into its scalar workspace.\n"
      << "backend,precision,benchmark_mode,task,dataset,topology,seed_base,"
         "seed,replicate,leaves,nodes,sites,unique_patterns,site_batch,"
         "binary_tree,tree_height,sackin_index,colless_index,"
         "normalized_colless,structural_rounds,primitive_levels,"
         "primitive_operations,planning_ms,cpu_workspace_setup_ms,"
         "accelerator_workspace_setup_ms,repeats,conditioning_ms,cpu_ms,"
         "cpu_p25_ms,cpu_p75_ms,accelerator_ms,accelerator_p25_ms,"
         "accelerator_p75_ms,upload_ms,kernel_ms,download_ms,"
         "cpu_over_accelerator,max_abs_error,max_relative_error,"
         "state_mismatches,cpu_samples_ms,accelerator_samples_ms\n"
      << std::setprecision(10) << backend << ',' << tree_hmm::kPrecisionName
      << ',' << options.benchmark_mode << ',' << task << ',' << problem.dataset
      << ',' << problem.topology << ',' << problem.base_seed << ','
      << problem.seed << ',' << problem.replicate << ',' << problem.leaves
      << ',' << problem.plan.num_nodes() << ',' << problem.raw_sites << ','
      << problem.sites << ',' << problem.sites << ','
      << (problem.shape.binary ? 1 : 0) << ',' << problem.shape.height << ','
      << problem.shape.sackin << ',' << problem.shape.colless << ','
      << problem.shape.normalized_colless << ',' << statistics.rounds << ','
      << statistics.primitive_levels << ',' << primitive_operations << ','
      << problem.planning_ms << ',' << cpu_workspace_setup_ms << ','
      << accelerator_workspace_setup_ms << ',' << options.repeats << ','
      << options.conditioning_ms << ',' << cpu_ms << ','
      << Quantile(timings.cpu, 0.25) << ',' << Quantile(timings.cpu, 0.75)
      << ',' << accelerator_ms << ',' << Quantile(timings.accelerator, 0.25)
      << ',' << Quantile(timings.accelerator, 0.75) << ','
      << Median(timings.upload) << ',' << Median(timings.kernel) << ','
      << Median(timings.download) << ',' << cpu_ms / accelerator_ms << ','
      << errors.maximum_absolute << ',' << errors.maximum_relative << ','
      << errors.state_mismatches << ',' << JoinSamples(timings.cpu) << ','
      << JoinSamples(timings.accelerator) << '\n';
}

void RunOne(const char *backend, const Options &options,
            std::size_t replicate) {
  const Problem problem = MakeProblem(options, replicate);
  TimeTask(
      backend, "likelihood", InferenceTask::kLikelihood, options, problem, {},
      [](auto &workspace, AlignmentModelView model) {
        workspace.Reserve(model);
      },
      [](auto &workspace, AlignmentModelView model, InputUpdate update) {
        return TASK_NAMESPACE::LogLikelihoodsPrepared(model, workspace, update);
      },
      [](const CpuTaskReference &cpu, std::span<const Scalar> actual) {
        return ScalarErrors(cpu.log_values(), actual);
      });

  TimeTask(
      backend, "joint-map", InferenceTask::kMaximum, options, problem, {},
      [](auto &workspace, AlignmentModelView model) {
        workspace.ReserveMaximum(model);
      },
      [](auto &workspace, AlignmentModelView model, InputUpdate update) {
        return TASK_NAMESPACE::MaximumAPosterioriPrepared(model, workspace,
                                                          update);
      },
      [](const CpuTaskReference &cpu, AlignmentMaximumView actual) {
        TaskErrors errors = ScalarErrors(cpu.log_values(), actual.log_weights);
        errors.state_mismatches =
            CountStateMismatches(cpu.states(), actual.states);
        return errors;
      });

  std::vector<Scalar> uniforms(problem.sites * problem.plan.num_nodes(),
                               Scalar{0.5});
  TimeTask(
      backend, "posterior-sample", InferenceTask::kSampling, options, problem,
      uniforms,
      [](auto &workspace, AlignmentModelView model) {
        workspace.ReserveSampling(model);
      },
      [&](auto &workspace, AlignmentModelView model, InputUpdate update) {
        return TASK_NAMESPACE::PosteriorSamplePrepared(
            model, std::span<const Scalar>(uniforms), workspace, update);
      },
      [](const CpuTaskReference &cpu, AlignmentPosteriorSampleView actual) {
        TaskErrors errors;
        errors.state_mismatches =
            CountStateMismatches(cpu.states(), actual.states);
        return errors;
      });

  TimeTask(
      backend, "all-marginals", InferenceTask::kMarginals, options, problem, {},
      [](auto &workspace, AlignmentModelView model) {
        workspace.ReserveMarginals(model);
      },
      [](auto &workspace, AlignmentModelView model, InputUpdate update) {
        return TASK_NAMESPACE::PosteriorMarginalsPrepared(model, workspace,
                                                          update);
      },
      [](const CpuTaskReference &cpu, AlignmentPosteriorView actual) {
        TaskErrors errors;
        IncludeScalarErrors(errors, cpu.log_values(), actual.log_likelihoods);
        IncludeScalarErrors(errors, cpu.node_values(), actual.ancestral_states);
        IncludeScalarErrors(errors, cpu.edge_values(), actual.substitutions);
        return errors;
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
