#ifndef PARALLEL_PHYLOGENETICS_BENCHMARK_H_
#define PARALLEL_PHYLOGENETICS_BENCHMARK_H_

#include "parallel_phylogenetics/alignment.h"
#include "parallel_phylogenetics/io.h"
#include "src/alignment_internal.h"

#include "synthetic.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace parallel_phylogenetics::benchmark {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Options {
  std::size_t leaves = 1024;
  std::size_t sites = 256;
  std::size_t site_batch = 0;
  int repeats = 5;
  std::size_t conditioning_ms = 0;
  std::string topology = "balanced";
  std::uint64_t seed = 1;
  std::size_t replicate_start = 0;
  std::size_t replicates = 1;
  bool compress_patterns = false;
  std::optional<std::filesystem::path> newick;
  std::optional<std::filesystem::path> fasta;
  std::optional<std::filesystem::path> phylip;
  std::optional<std::filesystem::path> pattern_weights;
  std::optional<std::string> dataset_label;
};

struct Problem {
  btrc::Plan plan;
  std::vector<Scalar> branch_lengths;
  std::vector<btrc::Index> observation_nodes;
  std::vector<Nucleotide> observations;
  std::string dataset;
  std::string topology;
  std::size_t leaves = 0;
  std::size_t sites = 0;
  std::size_t raw_sites = 0;
  std::vector<std::uint64_t> pattern_weights;
  std::uint64_t base_seed = 0;
  std::uint64_t seed = 0;
  std::size_t replicate = 0;
  double planning_ms = 0.0;
  TreeShapeStatistics shape;
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
      if (options.topology != "balanced" &&
          options.topology != "caterpillar" && options.topology != "yule" &&
          options.topology != "beta-critical" &&
          options.topology != "uniform") {
        throw std::invalid_argument(
            "--topology must be balanced, caterpillar, yule, beta-critical, "
            "or uniform");
      }
    } else if (option == "--seed") {
      options.seed = ParseNonnegativeSize(argv[index], "random seed");
    } else if (option == "--replicates") {
      options.replicates = ParseSize(argv[index], "replicate count");
    } else if (option == "--replicate-start") {
      options.replicate_start =
          ParseNonnegativeSize(argv[index], "first replicate");
    } else if (option == "--compress-patterns") {
      const std::string_view value = argv[index];
      if (value != "true" && value != "false")
        throw std::invalid_argument("--compress-patterns must be true or false");
      options.compress_patterns = value == "true";
    } else if (option == "--newick") {
      options.newick = argv[index];
    } else if (option == "--fasta") {
      options.fasta = argv[index];
    } else if (option == "--phylip") {
      options.phylip = argv[index];
    } else if (option == "--pattern-weights") {
      options.pattern_weights = argv[index];
    } else if (option == "--dataset-label") {
      options.dataset_label = argv[index];
      if (options.dataset_label->empty() ||
          options.dataset_label->find(',') != std::string::npos) {
        throw std::invalid_argument(
            "--dataset-label must be nonempty and contain no comma");
      }
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
  if (options.pattern_weights.has_value() && !has_alignment) {
    throw std::invalid_argument(
        "--pattern-weights applies only to an empirical alignment");
  }
  if (options.dataset_label.has_value() && !has_alignment) {
    throw std::invalid_argument(
        "--dataset-label applies only to an empirical alignment");
  }
  if (options.pattern_weights.has_value() && options.compress_patterns) {
    throw std::invalid_argument(
        "--pattern-weights and --compress-patterns are alternatives");
  }
  if (options.newick.has_value() &&
      (options.replicates != 1 || options.replicate_start != 0)) {
    throw std::invalid_argument(
        "--replicates and --replicate-start apply only to synthetic trees");
  }
  return options;
}

inline std::vector<std::uint64_t>
LoadPatternWeights(const std::filesystem::path &path,
                   std::size_t expected_patterns) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("failed to open " + path.string());
  std::vector<std::uint64_t> weights;
  std::uint64_t weight = 0;
  while (stream >> weight) {
    if (weight == 0)
      throw std::invalid_argument("pattern weights must be positive");
    weights.push_back(weight);
  }
  if (!stream.eof())
    throw std::invalid_argument("pattern-weight file contains invalid text");
  if (weights.size() != expected_patterns) {
    throw std::invalid_argument(
        "pattern-weight count does not match the alignment");
  }
  return weights;
}

inline void CompressPatterns(Problem &problem) {
  if (problem.sites == 0 || problem.observation_nodes.empty())
    throw std::invalid_argument("cannot compress an empty alignment");
  const std::size_t width = problem.observation_nodes.size();
  std::unordered_map<std::string_view, std::size_t> unique;
  unique.reserve(problem.sites);
  std::vector<Nucleotide> observations;
  observations.reserve(problem.observations.size());
  std::vector<std::uint64_t> weights;
  weights.reserve(problem.sites);
  for (std::size_t site = 0; site < problem.sites; ++site) {
    const auto *data = reinterpret_cast<const char *>(
        problem.observations.data() + site * width);
    const std::string_view pattern(data, width * sizeof(Nucleotide));
    const auto [iterator, inserted] = unique.emplace(pattern, weights.size());
    if (inserted) {
      observations.insert(observations.end(),
                          problem.observations.begin() + site * width,
                          problem.observations.begin() + (site + 1) * width);
      weights.push_back(1);
    } else {
      ++weights[iterator->second];
    }
  }
  problem.sites = weights.size();
  problem.observations = std::move(observations);
  problem.pattern_weights = std::move(weights);
}

inline Problem MakeProblem(const Options &options, std::size_t replicate = 0) {
  if (replicate < options.replicate_start ||
      replicate - options.replicate_start >= options.replicates) {
    throw std::invalid_argument("synthetic replicate is outside the requested range");
  }
  if (options.newick.has_value()) {
    const Clock::time_point planning_begin = Clock::now();
    Phylogeny phylogeny = LoadNewick(*options.newick);
    const double planning_ms = Milliseconds(planning_begin, Clock::now());
    const SequenceAlignment alignment = options.fasta.has_value()
                                            ? LoadFasta(*options.fasta)
                                            : LoadPhylip(*options.phylip);
    EncodedAlignment encoded = EncodeAlignment(phylogeny, alignment);
    std::vector<std::size_t> out_degree(phylogeny.plan.num_nodes(), 0);
    for (const btrc::Index parent : phylogeny.plan.edge_parents())
      ++out_degree[parent];
    const std::size_t leaves = static_cast<std::size_t>(
        std::count(out_degree.begin(), out_degree.end(), std::size_t{0}));
    std::vector<std::uint64_t> weights =
        options.pattern_weights.has_value()
            ? LoadPatternWeights(*options.pattern_weights, encoded.sites)
            : std::vector<std::uint64_t>(encoded.sites, 1);
    std::uint64_t raw_sites = 0;
    for (const std::uint64_t weight : weights) {
      if (weight > std::numeric_limits<std::uint64_t>::max() - raw_sites)
        throw std::overflow_error("total pattern weight overflows");
      raw_sites += weight;
    }
    if (raw_sites > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("total pattern weight exceeds size_t");
    Problem result{std::move(phylogeny.plan),
                   std::move(phylogeny.branch_lengths),
                   std::move(encoded.observation_nodes),
                   std::move(encoded.observations),
                   options.dataset_label.value_or(
                       options.newick->stem().string()),
                   "empirical",
                   leaves,
                   encoded.sites,
                   static_cast<std::size_t>(raw_sites),
                   std::move(weights),
                   options.seed,
                   options.seed,
                   0,
                   planning_ms,
                   {}};
    if (options.compress_patterns)
      CompressPatterns(result);
    result.shape = ShapeStatistics(result.plan);
    return result;
  }

  const std::size_t leaves = options.leaves;
  const std::size_t sites = options.sites;
  const std::uint64_t seed =
      SyntheticSeed(options.seed, leaves, sites, replicate, options.topology);
  SyntheticTopology topology =
      MakeSyntheticTopology(options.topology, leaves, seed);
  const Clock::time_point planning_begin = Clock::now();
  btrc::Plan plan = btrc::MakePlan(topology.parents);
  const double planning_ms = Milliseconds(planning_begin, Clock::now());
  std::vector<Scalar> lengths(plan.num_edges());
  for (std::size_t edge = 0; edge < lengths.size(); ++edge)
    lengths[edge] = Scalar{0.02} + Scalar{0.18} * static_cast<Scalar>(
                                                DeterministicRandom(seed + edge)
                                                    .Unit());

  Problem result{std::move(plan),
                 std::move(lengths),
                 std::move(topology.leaves),
                 MakeUniquePatterns(sites, leaves, seed),
                 "synthetic",
                 options.topology,
                 leaves,
                 sites,
                 sites,
                 std::vector<std::uint64_t>(sites, 1),
                 options.seed,
                 seed,
                 replicate,
                 planning_ms,
                 {}};
  result.shape = ShapeStatistics(result.plan);
  return result;
}

inline double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0)
    return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

inline double Quantile(std::vector<double> values, double probability) {
  if (values.empty() || probability < 0.0 || probability > 1.0)
    throw std::invalid_argument("invalid benchmark quantile");
  std::sort(values.begin(), values.end());
  const double position = probability * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

inline std::string JoinSamples(const std::vector<double> &values) {
  std::ostringstream stream;
  stream << std::setprecision(17);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      stream << '|';
    stream << values[index];
  }
  return stream.str();
}

struct BenchmarkResult {
  std::vector<Scalar> cpu_values;
  std::vector<Scalar> accelerator_values;
  tree_hmm::AcceleratorTimings accelerator_timings;
  double cpu_ms = 0.0;
  double prepare_ms = 0.0;
  double total_accelerator_ms = 0.0;
  double cpu_p25_ms = 0.0;
  double cpu_p75_ms = 0.0;
  double prepared_p25_ms = 0.0;
  double prepared_p75_ms = 0.0;
  double total_p25_ms = 0.0;
  double total_p75_ms = 0.0;
  std::vector<double> cpu_samples_ms;
  std::vector<double> prepare_samples_ms;
  std::vector<double> prepared_samples_ms;
  std::vector<double> upload_samples_ms;
  std::vector<double> kernel_samples_ms;
  std::vector<double> download_samples_ms;
  std::vector<double> total_samples_ms;
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

  std::span<const Scalar> cpu_values;
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
  return {std::vector<Scalar>(cpu_values.begin(), cpu_values.end()),
          std::vector<Scalar>(accelerator_result.values.begin(),
                              accelerator_result.values.end()),
          accelerator_result.timings,
          Median(cpu_times),
          Median(prepare_times),
          Median(total_times),
          Quantile(cpu_times, 0.25),
          Quantile(cpu_times, 0.75),
          Quantile(wall_times, 0.25),
          Quantile(wall_times, 0.75),
          Quantile(total_times, 0.25),
          Quantile(total_times, 0.75),
          cpu_times,
          prepare_times,
          wall_times,
          upload_times,
          kernel_times,
          download_times,
          total_times};
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
BenchmarkResult
RunChunkedInterleaved(AlignmentModelView model, int repeats,
                      std::size_t conditioning_ms, std::size_t site_batch,
                      Destination full_destination, Accelerator &&accelerator) {
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

  std::vector<Scalar> cpu_values(model.sites);
  std::vector<Scalar> accelerator_values(model.sites);
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
      const std::span<const Scalar> values =
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
    const Clock::time_point total_begin = Clock::now();
    const Clock::time_point shared_begin = Clock::now();
    internal::PrepareCategoricalShared(model, full_destination);
    prepare_elapsed += Milliseconds(shared_begin, Clock::now());
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += site_batch) {
      const std::size_t count = std::min(site_batch, model.sites - first_site);
      const AlignmentModelView chunk = SiteBatch(model, first_site, count);
      const auto destination = BatchPrefix(full_destination, count);
      const Clock::time_point prepare_begin = Clock::now();
      const auto factors =
          internal::PrepareCategoricalObservations(chunk, destination);
      prepare_elapsed += Milliseconds(prepare_begin, Clock::now());
      const tree_hmm::PartitionView result = accelerator(factors);
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
    total_times.push_back(Milliseconds(total_begin, Clock::now()));
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
          Median(total_times),
          Quantile(cpu_times, 0.25),
          Quantile(cpu_times, 0.75),
          Quantile(wall_times, 0.25),
          Quantile(wall_times, 0.75),
          Quantile(total_times, 0.25),
          Quantile(total_times, 0.75),
          cpu_times,
          prepare_times,
          wall_times,
          upload_times,
          kernel_times,
          download_times,
          total_times};
}

inline double MaxAbsoluteError(std::span<const Scalar> expected,
                               std::span<const Scalar> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    result = std::max(
        result, std::abs(expected[index] - static_cast<double>(actual[index])));
  }
  return result;
}

inline double MaxRelativeError(std::span<const Scalar> expected,
                               std::span<const Scalar> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double error =
        std::abs(expected[index] - static_cast<double>(actual[index]));
    result = std::max(
        result,
        error / std::max(1.0, std::abs(static_cast<double>(expected[index]))));
  }
  return result;
}

inline void PrintHeader(const char *backend, const std::string &device,
                        const Options &options, const Problem &problem) {
  const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
  std::cout
      << "# backend=" << backend << '\n'
      << "# precision=" << tree_hmm::kPrecisionName << '\n'
      << "# device=" << device << '\n'
      << "# dataset=" << problem.dataset << '\n'
      << "# topology=" << problem.topology << "-bifurcating-jc69\n"
      << "# timing_prepared=host wall time of one preallocated backend call; "
         "topology planning, workspace allocation, host factor construction, "
         "and warmup excluded\n"
      << "# timing_resident=not measured: every public prepared call uploads "
         "its inputs\n"
      << "# timing_device_compute_download=kernel+result-download device events; "
         "input upload excluded\n"
      << "# timing_end_to_end=host factor construction+prepared call\n"
      << "# conditioning_ms=" << options.conditioning_ms << '\n'
      << "# CPU and accelerator execution order alternates by repeat\n"
      << "backend,precision,dataset,topology,seed_base,seed,replicate,leaves,nodes,"
         "sites,unique_patterns,site_batch,binary_tree,tree_height,sackin_index,"
         "colless_index,normalized_colless,structural_rounds,"
         "primitive_levels,primitive_operations,planning_ms,staged_input_bytes,"
         "repeats,conditioning_ms,cpu_ms,cpu_p25_ms,cpu_p75_ms,prepare_ms,"
         "prepared_ms,prepared_p25_ms,prepared_p75_ms,resident_ms,"
         "device_compute_download_ms,upload_ms,kernel_ms,download_ms,"
         "end_to_end_ms,end_to_end_p25_ms,"
         "end_to_end_p75_ms,end_to_end_speedup,prepared_speedup,"
         "cpu_log_likelihood,accelerator_log_likelihood,max_abs_error,"
         "max_relative_error,cpu_samples_ms,prepare_samples_ms,"
         "prepared_samples_ms,upload_samples_ms,kernel_samples_ms,"
         "download_samples_ms,end_to_end_samples_ms\n";
  static_cast<void>(statistics);
}

inline void PrintRow(const char *backend, const Options &options,
                     const Problem &problem, const BenchmarkResult &result,
                     double absolute_error, double relative_error) {
  const double cpu_ms = result.cpu_ms;
  const double prepare_ms = result.prepare_ms;
  const tree_hmm::AcceleratorTimings &accelerator = result.accelerator_timings;
  const double total_accelerator_ms = result.total_accelerator_ms;
  const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
  const std::size_t site_batch =
      options.site_batch == 0 ? problem.sites
                              : std::min(options.site_batch, problem.sites);
  const std::size_t input_bytes =
      site_batch * problem.observation_nodes.size() * sizeof(Nucleotide) +
      problem.plan.num_edges() * 16 * sizeof(Scalar) +
      problem.observation_nodes.size() * sizeof(btrc::Index) +
      (4 + 16 * 4) * sizeof(Scalar);
  const std::size_t primitive_operations =
      statistics.rakes + problem.plan.num_branch_combinations() +
      problem.plan.num_branch_absorptions() + statistics.compressions;
  const auto weighted_sum = [&](std::span<const Scalar> values) {
    if (values.size() != problem.pattern_weights.size())
      throw std::invalid_argument("benchmark likelihood weights have a wrong shape");
    double result = 0.0;
    for (std::size_t pattern = 0; pattern < values.size(); ++pattern) {
      result += static_cast<double>(problem.pattern_weights[pattern]) *
                static_cast<double>(values[pattern]);
    }
    return result;
  };
  std::cout << std::setprecision(10) << backend << ','
            << tree_hmm::kPrecisionName << ',' << problem.dataset << ','
            << problem.topology << ',' << problem.base_seed << ','
            << problem.seed << ','
            << problem.replicate << ',' << problem.leaves << ','
            << problem.plan.num_nodes() << ',' << problem.raw_sites << ','
            << problem.sites << ',' << site_batch << ','
            << (problem.shape.binary ? 1 : 0) << ',' << problem.shape.height
            << ',' << problem.shape.sackin << ','
            << problem.shape.colless << ','
            << problem.shape.normalized_colless << ',' << statistics.rounds
            << ',' << statistics.primitive_levels << ','
            << primitive_operations << ',' << problem.planning_ms << ','
            << input_bytes << ',' << options.repeats << ','
            << options.conditioning_ms << ',' << cpu_ms << ','
            << result.cpu_p25_ms << ',' << result.cpu_p75_ms << ','
            << prepare_ms << ',' << accelerator.wall_ms << ','
            << result.prepared_p25_ms << ',' << result.prepared_p75_ms << ','
            << ',' << accelerator.kernel_ms + accelerator.download_ms << ','
            << accelerator.upload_ms << ',' << accelerator.kernel_ms << ','
            << accelerator.download_ms << ',' << total_accelerator_ms << ','
            << result.total_p25_ms << ',' << result.total_p75_ms << ','
            << cpu_ms / total_accelerator_ms << ','
            << cpu_ms / accelerator.wall_ms << ','
            << weighted_sum(result.cpu_values) << ','
            << weighted_sum(result.accelerator_values)
            << ',' << absolute_error << ',' << relative_error << ','
            << JoinSamples(result.cpu_samples_ms) << ','
            << JoinSamples(result.prepare_samples_ms) << ','
            << JoinSamples(result.prepared_samples_ms) << ','
            << JoinSamples(result.upload_samples_ms) << ','
            << JoinSamples(result.kernel_samples_ms) << ','
            << JoinSamples(result.download_samples_ms) << ','
            << JoinSamples(result.total_samples_ms) << '\n';
}

} // namespace parallel_phylogenetics::benchmark

#endif // PARALLEL_PHYLOGENETICS_BENCHMARK_H_
