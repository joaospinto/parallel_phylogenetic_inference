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
#include <utility>
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
  std::string benchmark_mode = "full-input-update";
  std::string synthetic_sequence_model = "independent-patterns";
  std::optional<double> evolutionary_root_to_tip_distance;
  double minimum_branch_length = 0.0;
  std::string study = "standard";
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
  std::string sequence_generation = "empirical";
  std::optional<double> evolutionary_root_to_tip_distance;
  double minimum_branch_length = 0.0;
  std::size_t floored_branch_count = 0;
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

inline double ParsePositiveDouble(const char *text, const char *description) {
  char *end = nullptr;
  const double value = std::strtod(text, &end);
  if (text == end || *end != '\0' || !(value > 0.0) || !std::isfinite(value))
    throw std::invalid_argument(std::string("invalid ") + description);
  return value;
}

inline double ParseNonnegativeDouble(const char *text,
                                     const char *description) {
  char *end = nullptr;
  const double value = std::strtod(text, &end);
  if (text == end || *end != '\0' || value < 0.0 || !std::isfinite(value))
    throw std::invalid_argument(std::string("invalid ") + description);
  return value;
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
    } else if (option == "--benchmark-mode") {
      options.benchmark_mode = argv[index];
      if (options.benchmark_mode != "fixed-model" &&
          options.benchmark_mode != "factor-update" &&
          options.benchmark_mode != "full-input-update") {
        throw std::invalid_argument(
            "--benchmark-mode must be fixed-model, factor-update, or "
            "full-input-update");
      }
    } else if (option == "--synthetic-sequence-model") {
      options.synthetic_sequence_model = argv[index];
      if (options.synthetic_sequence_model != "independent-patterns" &&
          options.synthetic_sequence_model != "jc69") {
        throw std::invalid_argument(
            "--synthetic-sequence-model must be independent-patterns or jc69");
      }
    } else if (option == "--evolutionary-root-to-tip-distance") {
      options.evolutionary_root_to_tip_distance = ParsePositiveDouble(
          argv[index], "evolutionary root-to-tip distance");
    } else if (option == "--minimum-branch-length") {
      options.minimum_branch_length =
          ParseNonnegativeDouble(argv[index], "minimum branch length");
    } else if (option == "--study-label") {
      options.study = argv[index];
      if (options.study.empty() || options.study.find(',') != std::string::npos ||
          options.study.find('\n') != std::string::npos) {
        throw std::invalid_argument(
            "--study-label must be a nonempty CSV-safe value");
      }
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
  if (has_alignment &&
      (options.synthetic_sequence_model != "independent-patterns" ||
       options.evolutionary_root_to_tip_distance.has_value())) {
    throw std::invalid_argument(
        "synthetic sequence options do not apply to empirical alignments");
  }
  if (options.synthetic_sequence_model == "jc69" &&
      !options.evolutionary_root_to_tip_distance.has_value()) {
    throw std::invalid_argument(
        "JC69 simulation requires --evolutionary-root-to-tip-distance");
  }
  if (options.synthetic_sequence_model != "jc69" &&
      options.evolutionary_root_to_tip_distance.has_value()) {
    throw std::invalid_argument(
        "--evolutionary-root-to-tip-distance applies only to JC69 simulation");
  }
  if (!has_alignment && options.minimum_branch_length != 0.0) {
    throw std::invalid_argument(
        "--minimum-branch-length applies only to empirical trees");
  }
  return options;
}

inline std::size_t FloorBranchLengths(std::vector<Scalar> &lengths,
                                      double minimum) {
  std::size_t count = 0;
  for (Scalar &length : lengths) {
    if (static_cast<double>(length) < minimum) {
      length = static_cast<Scalar>(minimum);
      ++count;
    }
  }
  return count;
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
    const std::size_t floored_branch_count =
        FloorBranchLengths(phylogeny.branch_lengths,
                           options.minimum_branch_length);
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
                   {},
                   "empirical",
                   std::nullopt,
                   options.minimum_branch_length,
                   floored_branch_count};
    if (options.compress_patterns)
      CompressPatterns(result);
    result.shape = ShapeStatistics(result.plan);
    return result;
  }

  const std::size_t leaves = options.leaves;
  const std::size_t sites = options.sites;
  const bool simulate_jc69 = options.synthetic_sequence_model == "jc69";
  const std::uint64_t topology_seed =
      simulate_jc69
          ? SyntheticTopologySeed(options.seed, leaves, replicate,
                                  options.topology)
          : SyntheticSeed(options.seed, leaves, sites, replicate,
                          options.topology);
  const std::uint64_t seed =
      simulate_jc69
          ? SyntheticSequenceSeed(topology_seed, sites,
                                  *options.evolutionary_root_to_tip_distance)
          : topology_seed;
  SyntheticTopology topology =
      MakeSyntheticTopology(options.topology, leaves, topology_seed);
  const Clock::time_point planning_begin = Clock::now();
  btrc::Plan plan = btrc::MakePlan(topology.parents);
  const double planning_ms = Milliseconds(planning_begin, Clock::now());
  std::vector<Scalar> lengths;
  std::vector<Nucleotide> observations;
  if (simulate_jc69) {
    const std::vector<double> simulation_lengths = MakeClockLikeBranchLengths(
        plan, *options.evolutionary_root_to_tip_distance);
    observations = SimulateJukesCantorAlignment(
        plan, simulation_lengths, topology.leaves, sites, seed);
    lengths.reserve(simulation_lengths.size());
    std::transform(simulation_lengths.begin(), simulation_lengths.end(),
                   std::back_inserter(lengths),
                   [](double length) { return static_cast<Scalar>(length); });
  } else {
    lengths.resize(plan.num_edges());
    for (std::size_t edge = 0; edge < lengths.size(); ++edge)
      lengths[edge] = Scalar{0.02} + Scalar{0.18} * static_cast<Scalar>(
                                                  DeterministicRandom(seed + edge)
                                                      .Unit());
    observations = MakeUniquePatterns(sites, leaves, seed);
  }

  Problem result{std::move(plan),
                 std::move(lengths),
                 std::move(topology.leaves),
                 std::move(observations),
                 simulate_jc69 ? "synthetic-jc69" : "synthetic",
                 options.topology,
                 leaves,
                 sites,
                 sites,
                 std::vector<std::uint64_t>(sites, 1),
                 options.seed,
                 seed,
                 replicate,
                 planning_ms,
                 {},
                 options.synthetic_sequence_model,
                 options.evolutionary_root_to_tip_distance,
                 0.0,
                 0};
  if (simulate_jc69 && options.compress_patterns)
    CompressPatterns(result);
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
  double initial_staging_ms = 0.0;
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

inline InputUpdate BenchmarkInputUpdate(std::string_view mode) {
  if (mode == "full-input-update")
    return InputUpdate::kAll;
  if (mode == "factor-update")
    return InputUpdate::kFactors;
  if (mode == "fixed-model")
    return InputUpdate::kNone;
  throw std::invalid_argument("unknown benchmark input-update mode");
}

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

inline std::size_t CpuReferenceSiteBatch(AlignmentModelView model) {
  // Keep the conventional correctness baseline independently bounded even
  // when an accelerator can accommodate a much larger site batch.
  constexpr std::size_t kTargetNodeSites = std::size_t{1} << 20;
  return std::min(model.sites,
                  std::max(std::size_t{1},
                           kTargetNodeSites / model.plan.num_nodes()));
}

template <typename AcceleratorWorkspace, typename Reserve, typename Evaluate>
BenchmarkResult RunCompleteAlignmentInterleaved(
    AlignmentModelView model, int repeats, std::size_t conditioning_ms,
    std::size_t site_batch, AcceleratorWorkspace &accelerator_workspace,
    Reserve &&reserve, Evaluate &&evaluate) {
  const std::size_t cpu_site_batch = CpuReferenceSiteBatch(model);
  const std::size_t remainder = model.sites % cpu_site_batch;
  SequentialWorkspace full_cpu;
  full_cpu.Reserve(model.plan, cpu_site_batch);
  SequentialWorkspace tail_cpu;
  if (remainder != 0)
    tail_cpu.Reserve(model.plan, remainder);
  reserve(accelerator_workspace, model, site_batch);

  std::vector<Scalar> cpu_values(model.sites);
  std::vector<Scalar> accelerator_values(model.sites);
  std::vector<double> cpu_times;
  std::vector<double> prepare_times;
  std::vector<double> wall_times;
  std::vector<double> upload_times;
  std::vector<double> kernel_times;
  std::vector<double> download_times;
  std::vector<double> total_times;
  const auto run_cpu = [&] {
    const Clock::time_point begin = Clock::now();
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += cpu_site_batch) {
      const std::size_t count =
          std::min(cpu_site_batch, model.sites - first_site);
      const AlignmentModelView chunk = SiteBatch(model, first_site, count);
      SequentialWorkspace &workspace =
          count == cpu_site_batch ? full_cpu : tail_cpu;
      const auto values = LogLikelihoodsPrepared(chunk, workspace);
      std::copy(values.begin(), values.end(), cpu_values.begin() + first_site);
    }
    cpu_times.push_back(Milliseconds(begin, Clock::now()));
  };
  const auto run_accelerator = [&] {
    const Clock::time_point outer_begin = Clock::now();
    const auto values =
        evaluate(model, accelerator_workspace, InputUpdate::kAll);
    const double outer_wall_ms = Milliseconds(outer_begin, Clock::now());
    const PreparedTimings timing = accelerator_workspace.LastTimings();
    prepare_times.push_back(
        std::max(0.0, outer_wall_ms - timing.backend.wall_ms));
    wall_times.push_back(timing.backend.wall_ms);
    upload_times.push_back(timing.backend.upload_ms);
    kernel_times.push_back(timing.backend.kernel_ms);
    download_times.push_back(timing.backend.download_ms);
    total_times.push_back(outer_wall_ms);
    std::copy(values.begin(), values.end(), accelerator_values.begin());
  };

  run_cpu();
  run_accelerator();
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
          0.0,
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

// Times resident-input modes as a sum of per-chunk prepared calls. Capacity-bounded
// alignments are processed one exact site chunk at a time. Each chunk is first
// staged by one untimed kAll call, after which its timed kFactors or kNone
// repetitions reuse precisely those resident observations. The workspace is
// then reused for the next chunk, so the benchmark neither allocates one
// device workspace per chunk nor implies that the full alignment is resident.
template <typename AcceleratorWorkspace, typename Reserve, typename Evaluate>
BenchmarkResult RunResidentChunkProjection(
    AlignmentModelView model, int repeats, std::size_t conditioning_ms,
    std::size_t site_batch, InputUpdate update,
    AcceleratorWorkspace &full_accelerator,
    AcceleratorWorkspace &tail_accelerator, Reserve &&reserve,
    Evaluate &&evaluate) {
  if (site_batch == 0 || site_batch > model.sites)
    throw std::invalid_argument("invalid accelerator site batch");
  const std::size_t cpu_site_batch = CpuReferenceSiteBatch(model);
  const std::size_t remainder = model.sites % site_batch;
  const std::size_t cpu_remainder = model.sites % cpu_site_batch;
  SequentialWorkspace full_cpu;
  full_cpu.Reserve(model.plan, cpu_site_batch);
  SequentialWorkspace tail_cpu;
  if (cpu_remainder != 0)
    tail_cpu.Reserve(model.plan, cpu_remainder);

  const AlignmentModelView first = SiteBatch(model, 0, site_batch);
  reserve(full_accelerator, first, first.sites);
  if (remainder != 0) {
    const AlignmentModelView tail =
        SiteBatch(model, model.sites - remainder, remainder);
    reserve(tail_accelerator, tail, tail.sites);
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
  prepare_times.assign(repeats, 0.0);
  wall_times.assign(repeats, 0.0);
  upload_times.assign(repeats, 0.0);
  kernel_times.assign(repeats, 0.0);
  download_times.assign(repeats, 0.0);
  total_times.assign(repeats, 0.0);

  // A full-input conditioning sweep warms the same kernels without relying on
  // resident state that will be replaced by the subsequent chunk.
  const auto run_cpu = [&] {
    const Clock::time_point begin = Clock::now();
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += cpu_site_batch) {
      const std::size_t count =
          std::min(cpu_site_batch, model.sites - first_site);
      const AlignmentModelView chunk = SiteBatch(model, first_site, count);
      SequentialWorkspace &workspace =
          count == cpu_site_batch ? full_cpu : tail_cpu;
      const auto values = LogLikelihoodsPrepared(chunk, workspace);
      std::copy(values.begin(), values.end(), cpu_values.begin() + first_site);
    }
    cpu_times.push_back(Milliseconds(begin, Clock::now()));
  };
  std::size_t conditioning_chunk = 0;
  const auto condition_accelerator = [&] {
    const std::size_t first_site = conditioning_chunk % model.sites;
    const std::size_t count = std::min(site_batch, model.sites - first_site);
    const AlignmentModelView chunk = SiteBatch(model, first_site, count);
    AcceleratorWorkspace &workspace =
        count == site_batch ? full_accelerator : tail_accelerator;
    static_cast<void>(evaluate(chunk, workspace, InputUpdate::kAll));
    conditioning_chunk = (first_site + count) % model.sites;
  };
  ConditionInterleaved(conditioning_ms, run_cpu,
                       condition_accelerator);
  run_cpu();
  cpu_times.clear();
  for (int repeat = 0; repeat < repeats; ++repeat)
    run_cpu();

  double initial_staging_ms = 0.0;
  for (std::size_t first_site = 0; first_site < model.sites;
       first_site += site_batch) {
    const std::size_t count = std::min(site_batch, model.sites - first_site);
    const AlignmentModelView chunk = SiteBatch(model, first_site, count);
    AcceleratorWorkspace &accelerator_workspace =
        count == site_batch ? full_accelerator : tail_accelerator;

    const Clock::time_point staging_begin = Clock::now();
    static_cast<void>(
        evaluate(chunk, accelerator_workspace, InputUpdate::kAll));
    initial_staging_ms += Milliseconds(staging_begin, Clock::now());

    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto run_accelerator = [&] {
        const Clock::time_point outer_begin = Clock::now();
        const std::span<const Scalar> values =
            evaluate(chunk, accelerator_workspace, update);
        const double outer_wall_ms = Milliseconds(outer_begin, Clock::now());
        const PreparedTimings timing = accelerator_workspace.LastTimings();
        prepare_times[repeat] +=
            std::max(0.0, outer_wall_ms - timing.backend.wall_ms);
        wall_times[repeat] += timing.backend.wall_ms;
        upload_times[repeat] += timing.backend.upload_ms;
        kernel_times[repeat] += timing.backend.kernel_ms;
        download_times[repeat] += timing.backend.download_ms;
        total_times[repeat] += outer_wall_ms;
        std::copy(values.begin(), values.end(),
                  accelerator_values.begin() + first_site);
      };
      run_accelerator();
    }
  }
  return {std::move(cpu_values),
          std::move(accelerator_values),
          {Median(upload_times), Median(kernel_times), Median(download_times),
           Median(wall_times)},
          Median(cpu_times),
          Median(prepare_times),
          Median(total_times),
          initial_staging_ms,
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

template <typename AcceleratorWorkspace, typename Reserve, typename Evaluate>
BenchmarkResult RunInterleaved(
    AlignmentModelView model, int repeats, std::size_t conditioning_ms,
    std::size_t site_batch, InputUpdate update,
    AcceleratorWorkspace &full_accelerator,
    AcceleratorWorkspace &tail_accelerator, Reserve &&reserve,
    Evaluate &&evaluate) {
  if (site_batch == 0 || site_batch > model.sites)
    throw std::invalid_argument("invalid accelerator site batch");
  if (update == InputUpdate::kAll) {
    return RunCompleteAlignmentInterleaved(
        model, repeats, conditioning_ms, site_batch, full_accelerator,
        std::forward<Reserve>(reserve), std::forward<Evaluate>(evaluate));
  }
  return RunResidentChunkProjection(
      model, repeats, conditioning_ms, site_batch, update, full_accelerator,
      tail_accelerator, std::forward<Reserve>(reserve),
      std::forward<Evaluate>(evaluate));
}

inline double MaxAbsoluteError(std::span<const Scalar> expected,
                               std::span<const Scalar> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double reference = static_cast<double>(expected[index]);
    const double value = static_cast<double>(actual[index]);
    if (!std::isfinite(reference) || !std::isfinite(value))
      return std::numeric_limits<double>::infinity();
    result = std::max(result, std::abs(reference - value));
  }
  return result;
}

inline double MaxRelativeError(std::span<const Scalar> expected,
                               std::span<const Scalar> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double reference = static_cast<double>(expected[index]);
    const double value = static_cast<double>(actual[index]);
    if (!std::isfinite(reference) || !std::isfinite(value))
      return std::numeric_limits<double>::infinity();
    const double error = std::abs(reference - value);
    result = std::max(
        result,
        error / std::max(1.0, std::abs(reference)));
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
      << "# benchmark_mode=" << options.benchmark_mode << '\n'
      << "# study=" << options.study << '\n'
      << "# sequence_generation=" << problem.sequence_generation << '\n'
      << "# topological_height_edges=" << problem.shape.height << '\n';
  if (problem.evolutionary_root_to_tip_distance.has_value()) {
    std::cout << "# evolutionary_root_to_tip_distance="
              << *problem.evolutionary_root_to_tip_distance << '\n';
  }
  std::cout
      << "# minimum_branch_length=" << problem.minimum_branch_length << '\n'
      << "# floored_branch_count=" << problem.floored_branch_count << '\n'
      << "# timing_prepared=host wall time inside the preallocated tree-HMM "
         "backend; it includes only the input transfers selected by "
         "benchmark_mode, computation, and result transfer\n"
      << "# timing_device_compute_download=kernel+result-download device events; "
         "input upload excluded\n"
      << "# timing_measured_total=complete-alignment public-call wall time "
         "for full-input-update; sum of exact per-chunk resident-call wall "
         "times for factor-update and fixed-model\n"
      << "# timing_initial_staging=one untimed full-input kAll evaluation per "
         "exact site chunk for resident modes; reported separately and "
         "excluded from timed calls\n"
      << "# warmup=all modes execute one untimed evaluation before "
         "measurement; this is not included in initial_staging_ms\n"
      << "# measurement_scope="
      << (options.benchmark_mode == "full-input-update" ||
                  (options.site_batch == 0 || options.site_batch >= problem.sites)
              ? "complete-alignment-wall-time"
              : "sum-of-per-chunk-resident-calls")
      << '\n'
      << "# resident_chunking=factor-update and fixed-model stage and measure "
         "each exact chunk before reusing the workspace; their reported "
         "totals are projections, not complete-alignment wall times, unless "
         "site_batch equals the number of unique patterns\n"
      << "# factor_projection=for multiple chunks, factor-update sums one "
         "factor refresh per resident chunk; it does not estimate a storage "
         "scheme that shares one refreshed factor copy across all chunks\n"
      << "# conditioning_ms=" << options.conditioning_ms << '\n'
      << "# conventional_cpu_reference=full evaluation with an independently "
         "memory-bounded workspace; it is a correctness/reference timing, not a "
         "mode-matched resident implementation\n"
      << "# execution_order=CPU and accelerator alternate for full-input; "
         "resident projections time the conventional CPU reference "
         "separately\n"
      << "backend,precision,benchmark_mode,study,dataset,topology,sequence_generation,"
         "evolutionary_root_to_tip_distance,minimum_branch_length,floored_branch_count,seed_base,seed,replicate,leaves,nodes,"
         "sites,unique_patterns,site_batch,cpu_reference_site_batch,binary_tree,tree_height,sackin_index,"
         "colless_index,normalized_colless,structural_rounds,"
         "primitive_levels,primitive_operations,planning_ms,"
         "repeats,conditioning_ms,cpu_ms,cpu_p25_ms,cpu_p75_ms,prepare_ms,"
         "prepared_ms,prepared_p25_ms,prepared_p75_ms,initial_staging_ms,"
         "device_compute_download_ms,upload_ms,kernel_ms,download_ms,"
         "measured_total_ms,measured_total_p25_ms,"
         "measured_total_p75_ms,conventional_full_cpu_over_measured_total,"
         "conventional_full_cpu_over_prepared,"
         "cpu_log_likelihood,accelerator_log_likelihood,max_abs_error,"
         "max_relative_error,cpu_samples_ms,prepare_samples_ms,"
         "prepared_samples_ms,upload_samples_ms,kernel_samples_ms,"
         "download_samples_ms,measured_total_samples_ms\n";
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
            << tree_hmm::kPrecisionName << ',' << options.benchmark_mode << ','
            << options.study << ','
            << problem.dataset << ','
            << problem.topology << ',' << problem.sequence_generation << ',';
  if (problem.evolutionary_root_to_tip_distance.has_value())
    std::cout << *problem.evolutionary_root_to_tip_distance;
  std::cout << ',' << problem.minimum_branch_length << ','
            << problem.floored_branch_count << ',' << problem.base_seed << ','
            << problem.seed << ','
            << problem.replicate << ',' << problem.leaves << ','
            << problem.plan.num_nodes() << ',' << problem.raw_sites << ','
            << problem.sites << ',' << site_batch << ','
            << CpuReferenceSiteBatch(
                   AlignmentModelView{problem.plan, problem.sites,
                                      problem.branch_lengths,
                                      problem.observation_nodes,
                                      problem.observations})
            << ','
            << (problem.shape.binary ? 1 : 0) << ',' << problem.shape.height
            << ',' << problem.shape.sackin << ','
            << problem.shape.colless << ','
            << problem.shape.normalized_colless << ',' << statistics.rounds
            << ',' << statistics.primitive_levels << ','
            << primitive_operations << ',' << problem.planning_ms << ','
            << options.repeats << ','
            << options.conditioning_ms << ',' << cpu_ms << ','
            << result.cpu_p25_ms << ',' << result.cpu_p75_ms << ','
            << prepare_ms << ',' << accelerator.wall_ms << ','
            << result.prepared_p25_ms << ',' << result.prepared_p75_ms << ','
            << result.initial_staging_ms << ','
            << accelerator.kernel_ms + accelerator.download_ms << ','
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
