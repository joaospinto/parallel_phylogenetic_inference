#ifndef PARALLEL_PHYLOGENETICS_BENCHMARK_H_
#define PARALLEL_PHYLOGENETICS_BENCHMARK_H_

#include "parallel_phylogenetics/alignment.h"
#include "parallel_phylogenetics/io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_phylogenetics::benchmark {

using Clock = std::chrono::steady_clock;

struct Options {
  std::size_t leaves = 1024;
  std::size_t sites = 256;
  std::size_t site_batch = 0;
  int repeats = 5;
  std::size_t conditioning_ms = 0;
  std::string topology = "balanced";
  std::optional<std::filesystem::path> newick;
  std::optional<std::filesystem::path> fasta;
  std::optional<std::filesystem::path> phylip;
};

struct Problem {
  btrc::Plan plan;
  std::vector<double> branch_lengths;
  std::vector<btrc::Index> observation_nodes;
  std::vector<Nucleotide> observations;
  std::string dataset;
  std::string topology;
  std::size_t leaves = 0;
  std::size_t sites = 0;
};

inline std::size_t ParseSize(const char *text, const char *description) {
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text == end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string("invalid ") + description);
  }
  return static_cast<std::size_t>(value);
}

inline std::size_t ParseNonnegativeSize(const char *text,
                                        const char *description) {
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (text == end || *end != '\0' ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string("invalid ") + description);
  }
  return static_cast<std::size_t>(value);
}

inline Options ParseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc)
      throw std::invalid_argument(std::string("missing value for ") +
                                  argv[index]);
    const std::string_view option = argv[index++];
    if (option == "--leaves") {
      options.leaves = ParseSize(argv[index], "leaf count");
    } else if (option == "--sites") {
      options.sites = ParseSize(argv[index], "site count");
    } else if (option == "--site-batch") {
      options.site_batch = ParseSize(argv[index], "site batch size");
    } else if (option == "--repeats") {
      const std::size_t repeats = ParseSize(argv[index], "repeat count");
      if (repeats > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("repeat count is too large");
      options.repeats = static_cast<int>(repeats);
    } else if (option == "--conditioning-ms") {
      options.conditioning_ms =
          ParseNonnegativeSize(argv[index], "conditioning duration");
    } else if (option == "--topology") {
      options.topology = argv[index];
      if (options.topology != "balanced" && options.topology != "caterpillar") {
        throw std::invalid_argument(
            "--topology must be balanced or caterpillar");
      }
    } else if (option == "--newick") {
      options.newick = argv[index];
    } else if (option == "--fasta") {
      options.fasta = argv[index];
    } else if (option == "--phylip") {
      options.phylip = argv[index];
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  if (options.fasta.has_value() && options.phylip.has_value())
    throw std::invalid_argument("--fasta and --phylip are alternatives");
  const bool has_alignment =
      options.fasta.has_value() || options.phylip.has_value();
  if (options.newick.has_value() != has_alignment)
    throw std::invalid_argument(
        "--newick and exactly one of --fasta or --phylip must be supplied "
        "together");
  return options;
}

inline Problem MakeProblem(const Options &options) {
  if (options.newick.has_value()) {
    Phylogeny phylogeny = LoadNewick(*options.newick);
    const SequenceAlignment alignment = options.fasta.has_value()
                                            ? LoadFasta(*options.fasta)
                                            : LoadPhylip(*options.phylip);
    EncodedAlignment encoded = EncodeAlignment(phylogeny, alignment);
    std::vector<std::size_t> out_degree(phylogeny.plan.num_nodes(), 0);
    for (const btrc::Index parent : phylogeny.plan.edge_parents())
      ++out_degree[parent];
    const std::size_t leaves = static_cast<std::size_t>(
        std::count(out_degree.begin(), out_degree.end(), std::size_t{0}));
    return {std::move(phylogeny.plan),
            std::move(phylogeny.branch_lengths),
            std::move(encoded.observation_nodes),
            std::move(encoded.observations),
            options.newick->stem().string(),
            "empirical",
            leaves,
            encoded.sites};
  }

  const std::size_t leaves = options.leaves;
  const std::size_t sites = options.sites;
  if (leaves == 0 || (leaves & (leaves - 1)) != 0)
    throw std::invalid_argument("--leaves must be a power of two");
  if (leaves > std::numeric_limits<std::size_t>::max() / 2 + 1)
    throw std::length_error("the requested tree is too large");
  const std::size_t nodes = 2 * leaves - 1;
  if (nodes - 1 > std::numeric_limits<btrc::Index>::max())
    throw std::length_error("the requested tree exceeds the planner limit");
  std::vector<std::int64_t> parents(nodes, -1);
  if (options.topology == "balanced") {
    for (std::size_t node = 1; node < nodes; ++node)
      parents[node] = static_cast<std::int64_t>((node - 1) / 2);
  } else {
    if (leaves == 1) {
      parents[0] = -1;
    } else {
      const std::size_t internal_nodes = leaves - 1;
      for (std::size_t node = 1; node < internal_nodes; ++node)
        parents[node] = static_cast<std::int64_t>(node - 1);
      for (std::size_t leaf = 0; leaf + 2 < leaves; ++leaf) {
        parents[internal_nodes + leaf] = static_cast<std::int64_t>(leaf);
      }
      parents[nodes - 2] = static_cast<std::int64_t>(internal_nodes - 1);
      parents[nodes - 1] = static_cast<std::int64_t>(internal_nodes - 1);
    }
  }
  btrc::Plan plan = btrc::MakePlan(parents);
  std::vector<double> lengths(plan.num_edges());
  for (std::size_t edge = 0; edge < lengths.size(); ++edge)
    lengths[edge] = 0.02 + 0.001 * static_cast<double>(edge % 29);

  const std::size_t first_leaf = leaves - 1;
  std::vector<btrc::Index> observation_nodes(leaves);
  for (std::size_t leaf = 0; leaf < leaves; ++leaf)
    observation_nodes[leaf] = static_cast<btrc::Index>(first_leaf + leaf);
  std::vector<Nucleotide> observations(sites * leaves);
  constexpr std::array<Nucleotide, 4> kStates{Nucleotide::kA, Nucleotide::kC,
                                              Nucleotide::kG, Nucleotide::kT};
  for (std::size_t site = 0; site < sites; ++site) {
    for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
      const std::size_t state = (site * 17 + leaf * 13 + (site ^ leaf)) % 4;
      observations[site * leaves + leaf] = kStates[state];
    }
  }
  return {std::move(plan),
          std::move(lengths),
          std::move(observation_nodes),
          std::move(observations),
          "synthetic",
          options.topology,
          leaves,
          sites};
}

inline double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0)
    return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

inline double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct BenchmarkResult {
  std::vector<double> cpu_values;
  std::vector<float> accelerator_values;
  tree_hmm::AcceleratorTimings accelerator_timings;
  double cpu_ms = 0.0;
  double prepare_ms = 0.0;
  double total_accelerator_ms = 0.0;
};

template <typename Cpu, typename Accelerator>
void ConditionInterleaved(std::size_t milliseconds, Cpu &&cpu,
                          Accelerator &&accelerator) {
  if (milliseconds == 0)
    return;
  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(milliseconds);
  bool cpu_first = true;
  do {
    if (cpu_first) {
      cpu();
      accelerator();
    } else {
      accelerator();
      cpu();
    }
    cpu_first = !cpu_first;
  } while (Clock::now() < deadline);
}

// The callable must synchronously evaluate the factors and return a view whose
// storage remains valid until the next call. Alternating which implementation
// runs first reduces order and thermal bias on passively cooled devices.
template <typename Destination, typename Accelerator>
BenchmarkResult RunInterleaved(AlignmentModelView model, int repeats,
                               std::size_t conditioning_ms,
                               Destination destination,
                               Accelerator &&accelerator) {
  SequentialWorkspace cpu_workspace;
  cpu_workspace.Reserve(model.plan, model.sites);

  static_cast<void>(LogLikelihoodsPrepared(model, cpu_workspace));
  const auto warm_factors = Prepare(model, destination);
  static_cast<void>(accelerator(warm_factors));

  std::vector<double> cpu_times;
  std::vector<double> prepare_times;
  std::vector<double> wall_times;
  std::vector<double> upload_times;
  std::vector<double> kernel_times;
  std::vector<double> download_times;
  std::vector<double> total_times;
  cpu_times.reserve(repeats);
  prepare_times.reserve(repeats);
  wall_times.reserve(repeats);
  upload_times.reserve(repeats);
  kernel_times.reserve(repeats);
  download_times.reserve(repeats);
  total_times.reserve(repeats);

  std::span<const double> cpu_values;
  tree_hmm::PartitionView accelerator_result;
  const auto run_cpu = [&] {
    const Clock::time_point begin = Clock::now();
    cpu_values = LogLikelihoodsPrepared(model, cpu_workspace);
    cpu_times.push_back(Milliseconds(begin, Clock::now()));
  };
  const auto run_accelerator = [&] {
    const Clock::time_point total_begin = Clock::now();
    const Clock::time_point prepare_begin = total_begin;
    const auto factors = Prepare(model, destination);
    const Clock::time_point prepare_end = Clock::now();
    accelerator_result = accelerator(factors);
    const Clock::time_point total_end = Clock::now();
    prepare_times.push_back(Milliseconds(prepare_begin, prepare_end));
    wall_times.push_back(accelerator_result.timings.wall_ms);
    upload_times.push_back(accelerator_result.timings.upload_ms);
    kernel_times.push_back(accelerator_result.timings.kernel_ms);
    download_times.push_back(accelerator_result.timings.download_ms);
    total_times.push_back(Milliseconds(total_begin, total_end));
  };

  ConditionInterleaved(conditioning_ms, run_cpu, run_accelerator);
  cpu_times.clear();
  prepare_times.clear();
  wall_times.clear();
  upload_times.clear();
  kernel_times.clear();
  download_times.clear();
  total_times.clear();

  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (repeat % 2 == 0) {
      run_cpu();
      run_accelerator();
    } else {
      run_accelerator();
      run_cpu();
    }
  }

  accelerator_result.timings.wall_ms = Median(wall_times);
  accelerator_result.timings.upload_ms = Median(upload_times);
  accelerator_result.timings.kernel_ms = Median(kernel_times);
  accelerator_result.timings.download_ms = Median(download_times);
  return {std::vector<double>(cpu_values.begin(), cpu_values.end()),
          std::vector<float>(accelerator_result.values.begin(),
                             accelerator_result.values.end()),
          accelerator_result.timings,
          Median(cpu_times),
          Median(prepare_times),
          Median(total_times)};
}

inline AlignmentModelView SiteBatch(AlignmentModelView model,
                                    std::size_t first_site,
                                    std::size_t site_count) {
  return {
      model.plan,
      site_count,
      model.branch_lengths,
      model.observation_nodes,
      model.observations.subspan(first_site * model.observation_nodes.size(),
                                 site_count * model.observation_nodes.size()),
      model.root_frequencies,
      model.substitution_rate};
}

inline tree_hmm::MutableBatchedModelView
BatchPrefix(tree_hmm::MutableBatchedModelView destination, std::size_t batch) {
  if (batch == 0 || batch > destination.batch)
    throw std::invalid_argument(
        "requested batch exceeds the accelerator input capacity");
  if (destination.node_potentials.size() % destination.batch != 0)
    throw std::invalid_argument("accelerator input capacity has a wrong shape");
  const std::size_t node_values =
      batch * (destination.node_potentials.size() / destination.batch);
  return {destination.plan, destination.states, batch,
          destination.node_potentials.first(node_values),
          destination.edge_potentials};
}

inline tree_hmm::MutableBatchedCategoricalModelView
BatchPrefix(tree_hmm::MutableBatchedCategoricalModelView destination,
            std::size_t batch) {
  if (batch == 0 || batch > destination.batch)
    throw std::invalid_argument(
        "requested batch exceeds the accelerator input capacity");
  if (destination.observations.size() % destination.batch != 0) {
    throw std::invalid_argument(
        "categorical accelerator input capacity has a wrong shape");
  }
  const std::size_t observation_values =
      batch * (destination.observations.size() / destination.batch);
  return {destination.plan,
          destination.states,
          batch,
          destination.categories,
          destination.observation_nodes,
          destination.observations.first(observation_values),
          destination.root_potential,
          destination.emission_potentials,
          destination.edge_potentials};
}

// Evaluates a complete alignment in fixed-size batches. The caller reserves
// one accelerator workspace at the full batch capacity; its prefix is reused
// for the tail, so neither execution path allocates while timing. Result
// concatenation is deliberately outside the measured intervals because both
// backends have already materialized those values.
template <typename Destination, typename Accelerator>
BenchmarkResult RunChunkedInterleaved(AlignmentModelView model, int repeats,
                                      std::size_t conditioning_ms,
                                      std::size_t site_batch,
                                      Destination full_destination,
                                      Accelerator &&accelerator) {
  if (site_batch == 0 || site_batch >= model.sites)
    throw std::invalid_argument(
        "chunked inference requires a batch smaller than the alignment");
  const std::size_t remainder = model.sites % site_batch;
  SequentialWorkspace full_cpu;
  full_cpu.Reserve(model.plan, site_batch);
  SequentialWorkspace tail_cpu;
  if (remainder != 0)
    tail_cpu.Reserve(model.plan, remainder);

  const AlignmentModelView first = SiteBatch(model, 0, site_batch);
  static_cast<void>(LogLikelihoodsPrepared(first, full_cpu));
  static_cast<void>(accelerator(Prepare(first, full_destination)));
  if (remainder != 0) {
    const AlignmentModelView tail =
        SiteBatch(model, model.sites - remainder, remainder);
    static_cast<void>(LogLikelihoodsPrepared(tail, tail_cpu));
    static_cast<void>(
        accelerator(Prepare(tail, BatchPrefix(full_destination, remainder))));
  }

  std::vector<double> cpu_values(model.sites);
  std::vector<float> accelerator_values(model.sites);
  std::vector<double> cpu_times;
  std::vector<double> prepare_times;
  std::vector<double> wall_times;
  std::vector<double> upload_times;
  std::vector<double> kernel_times;
  std::vector<double> download_times;
  std::vector<double> total_times;
  cpu_times.reserve(repeats);
  prepare_times.reserve(repeats);
  wall_times.reserve(repeats);
  upload_times.reserve(repeats);
  kernel_times.reserve(repeats);
  download_times.reserve(repeats);
  total_times.reserve(repeats);

  const auto run_cpu = [&] {
    double elapsed = 0.0;
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += site_batch) {
      const std::size_t count = std::min(site_batch, model.sites - first_site);
      const AlignmentModelView chunk = SiteBatch(model, first_site, count);
      SequentialWorkspace &workspace =
          count == site_batch ? full_cpu : tail_cpu;
      const Clock::time_point begin = Clock::now();
      const std::span<const double> values =
          LogLikelihoodsPrepared(chunk, workspace);
      elapsed += Milliseconds(begin, Clock::now());
      std::copy(values.begin(), values.end(), cpu_values.begin() + first_site);
    }
    cpu_times.push_back(elapsed);
  };

  const auto run_accelerator = [&] {
    double prepare_elapsed = 0.0;
    double wall_elapsed = 0.0;
    double upload_elapsed = 0.0;
    double kernel_elapsed = 0.0;
    double download_elapsed = 0.0;
    double total_elapsed = 0.0;
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += site_batch) {
      const std::size_t count = std::min(site_batch, model.sites - first_site);
      const AlignmentModelView chunk = SiteBatch(model, first_site, count);
      const auto destination = BatchPrefix(full_destination, count);
      const Clock::time_point total_begin = Clock::now();
      const Clock::time_point prepare_begin = total_begin;
      const auto factors = Prepare(chunk, destination);
      prepare_elapsed += Milliseconds(prepare_begin, Clock::now());
      const tree_hmm::PartitionView result = accelerator(factors);
      total_elapsed += Milliseconds(total_begin, Clock::now());
      wall_elapsed += result.timings.wall_ms;
      upload_elapsed += result.timings.upload_ms;
      kernel_elapsed += result.timings.kernel_ms;
      download_elapsed += result.timings.download_ms;
      std::copy(result.values.begin(), result.values.end(),
                accelerator_values.begin() + first_site);
    }
    prepare_times.push_back(prepare_elapsed);
    wall_times.push_back(wall_elapsed);
    upload_times.push_back(upload_elapsed);
    kernel_times.push_back(kernel_elapsed);
    download_times.push_back(download_elapsed);
    total_times.push_back(total_elapsed);
  };

  ConditionInterleaved(conditioning_ms, run_cpu, run_accelerator);
  cpu_times.clear();
  prepare_times.clear();
  wall_times.clear();
  upload_times.clear();
  kernel_times.clear();
  download_times.clear();
  total_times.clear();

  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (repeat % 2 == 0) {
      run_cpu();
      run_accelerator();
    } else {
      run_accelerator();
      run_cpu();
    }
  }
  return {std::move(cpu_values),
          std::move(accelerator_values),
          {Median(upload_times), Median(kernel_times), Median(download_times),
           Median(wall_times)},
          Median(cpu_times),
          Median(prepare_times),
          Median(total_times)};
}

inline double MaxAbsoluteError(std::span<const double> expected,
                               std::span<const float> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    result = std::max(
        result, std::abs(expected[index] - static_cast<double>(actual[index])));
  }
  return result;
}

inline double MaxRelativeError(std::span<const double> expected,
                               std::span<const float> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double error =
        std::abs(expected[index] - static_cast<double>(actual[index]));
    result = std::max(result, error / std::max(1.0, std::abs(expected[index])));
  }
  return result;
}

inline void PrintHeader(const char *backend, const std::string &device,
                        const Options &options, const Problem &problem) {
  const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
  std::cout << "# backend=" << backend << '\n'
            << "# device=" << device << '\n'
            << "# dataset=" << problem.dataset << '\n'
            << "# topology=" << problem.topology << "-bifurcating-jc69\n"
            << "# prepared calls exclude workspace allocation and warmup\n"
            << "# conditioning_ms=" << options.conditioning_ms << '\n'
            << "# CPU and accelerator execution order alternates by repeat\n"
            << "backend,dataset,topology,leaves,nodes,sites,site_batch,"
               "primitive_levels,repeats,conditioning_ms,"
               "cpu_ms,prepare_ms,accelerator_wall_ms,upload_ms,kernel_ms,"
               "download_ms,total_accelerator_ms,wall_speedup,"
               "cpu_log_likelihood,accelerator_log_likelihood,max_abs_error,"
               "max_relative_error\n";
  static_cast<void>(statistics);
}

inline void PrintRow(const char *backend, const Options &options,
                     const Problem &problem, double cpu_ms, double prepare_ms,
                     const tree_hmm::AcceleratorTimings &accelerator,
                     double total_accelerator_ms,
                     std::span<const double> cpu_values,
                     std::span<const float> accelerator_values,
                     double absolute_error, double relative_error) {
  const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
  std::cout << std::setprecision(10) << backend << ',' << problem.dataset << ','
            << problem.topology << ',' << problem.leaves << ','
            << problem.plan.num_nodes() << ',' << problem.sites << ','
            << (options.site_batch == 0
                    ? problem.sites
                    : std::min(options.site_batch, problem.sites))
            << ',' << statistics.primitive_levels << ',' << options.repeats
            << ',' << options.conditioning_ms << ',' << cpu_ms << ','
            << prepare_ms << ',' << accelerator.wall_ms
            << ',' << accelerator.upload_ms << ',' << accelerator.kernel_ms
            << ',' << accelerator.download_ms << ',' << total_accelerator_ms
            << ',' << cpu_ms / total_accelerator_ms << ','
            << std::accumulate(cpu_values.begin(), cpu_values.end(), 0.0) << ','
            << std::accumulate(accelerator_values.begin(),
                               accelerator_values.end(), 0.0)
            << ',' << absolute_error << ',' << relative_error << '\n';
}

} // namespace parallel_phylogenetics::benchmark

#endif // PARALLEL_PHYLOGENETICS_BENCHMARK_H_
