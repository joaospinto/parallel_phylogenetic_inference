#include "benchmark.h"

#include <libhmsbeagle/beagle.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace parallel_phylogenetics::benchmark {
namespace {

struct BeagleOptions {
  Options problem;
  std::string resource = "cpu";
  int threads = 1;
};

int CheckedInt(std::size_t value, const char *description) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error(std::string(description) + " exceeds BEAGLE's limit");
  return static_cast<int>(value);
}

BeagleOptions ParseBeagleOptions(int argc, char **argv) {
  BeagleOptions result;
  std::vector<char *> problem_arguments;
  problem_arguments.reserve(static_cast<std::size_t>(argc));
  problem_arguments.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--beagle-resource" || option == "--beagle-threads") {
      if (++index >= argc)
        throw std::invalid_argument("missing value for " + std::string(option));
      if (option == "--beagle-resource") {
        result.resource = argv[index];
        if (result.resource != "cpu" && result.resource != "cuda") {
          throw std::invalid_argument(
              "--beagle-resource must be cpu or cuda");
        }
      } else {
        const std::size_t threads = ParseSize(argv[index], "BEAGLE thread count");
        if (threads >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          throw std::invalid_argument("BEAGLE thread count is too large");
        }
        result.threads = static_cast<int>(threads);
      }
      continue;
    }
    problem_arguments.push_back(argv[index]);
    if (index + 1 >= argc)
      throw std::invalid_argument("missing value for " + std::string(option));
    problem_arguments.push_back(argv[++index]);
  }
  result.problem = ParseOptions(CheckedInt(problem_arguments.size(),
                                           "argument count"),
                                problem_arguments.data());
  return result;
}

std::string BeagleError(int code) {
  std::string result = "BEAGLE error " + std::to_string(code);
  if (code == BEAGLE_ERROR_OUT_OF_MEMORY)
    result += " (out of memory)";
  return result;
}

void CheckBeagle(int code, const char *operation) {
  if (code != BEAGLE_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with " +
                             BeagleError(code));
}

class BeagleInstance {
public:
  explicit BeagleInstance(int instance) : instance_(instance) {
    if (instance_ < 0)
      throw std::runtime_error("beagleCreateInstance failed with " +
                               BeagleError(instance_));
  }

  ~BeagleInstance() {
    if (instance_ >= 0)
      static_cast<void>(beagleFinalizeInstance(instance_));
  }

  BeagleInstance(const BeagleInstance &) = delete;
  BeagleInstance &operator=(const BeagleInstance &) = delete;

  int get() const { return instance_; }

private:
  int instance_;
};

struct BeagleTree {
  std::vector<int> node_buffers;
  std::vector<int> matrix_indices;
  std::vector<int> scale_indices;
  std::vector<BeagleOperation> operations;
  std::vector<std::vector<double>> tip_partials;
  std::vector<std::vector<int>> tip_states;
  std::vector<std::size_t> tip_observation_indices;
  std::vector<bool> compact_tip;
  int tip_count = 0;
  int partials_buffer_count = 0;
  int constant_buffer = BEAGLE_OP_NONE;
  int root_buffer = BEAGLE_OP_NONE;
  int cumulative_scale = BEAGLE_OP_NONE;
};

BeagleTree MakeBeagleTree(AlignmentModelView model,
                          std::span<const std::uint8_t> compact_observations) {
  const std::size_t nodes = model.plan.num_nodes();
  const std::size_t edges = model.plan.num_edges();
  if (edges + 1 != nodes)
    throw std::invalid_argument("BEAGLE benchmark requires a tree");

  std::vector<std::vector<btrc::Index>> child_edges(nodes);
  for (std::size_t edge = 0; edge < edges; ++edge) {
    const btrc::Index parent = model.plan.edge_parents()[edge];
    if (parent >= nodes)
      throw std::invalid_argument("tree contains an invalid parent index");
    child_edges[parent].push_back(static_cast<btrc::Index>(edge));
  }

  std::vector<std::size_t> observation_index(nodes, nodes);
  for (std::size_t index = 0; index < model.observation_nodes.size(); ++index) {
    const btrc::Index node = model.observation_nodes[index];
    if (node >= nodes || observation_index[node] != nodes) {
      throw std::invalid_argument(
          "BEAGLE benchmark requires distinct valid observation nodes");
    }
    observation_index[node] = index;
  }

  std::vector<btrc::Index> postorder;
  postorder.reserve(nodes);
  std::vector<btrc::Index> stack{model.plan.root()};
  while (!stack.empty()) {
    const btrc::Index node = stack.back();
    stack.pop_back();
    postorder.push_back(node);
    for (const btrc::Index edge : child_edges[node])
      stack.push_back(model.plan.edge_children()[edge]);
  }
  std::reverse(postorder.begin(), postorder.end());
  if (postorder.size() != nodes)
    throw std::invalid_argument("tree is not connected");

  BeagleTree result;
  result.node_buffers.assign(nodes, BEAGLE_OP_NONE);
  std::size_t temporary_count = 0;
  std::size_t operation_count = 0;
  bool needs_constant = false;
  for (std::size_t node = 0; node < nodes; ++node) {
    if (child_edges[node].empty()) {
      if (observation_index[node] == nodes) {
        throw std::invalid_argument(
            "BEAGLE benchmark requires one observation at every tip");
      }
      result.node_buffers[node] = result.tip_count++;
    } else if (observation_index[node] != nodes) {
      throw std::invalid_argument(
          "BEAGLE benchmark does not support observations at internal nodes");
    } else {
      needs_constant = needs_constant || child_edges[node].size() == 1;
      operation_count += std::max(std::size_t{1}, child_edges[node].size() - 1);
      if (child_edges[node].size() > 2)
        temporary_count += child_edges[node].size() - 2;
    }
  }
  if (static_cast<std::size_t>(result.tip_count) !=
      model.observation_nodes.size()) {
    throw std::invalid_argument(
        "BEAGLE benchmark requires observations only at every tip");
  }

  int next_buffer = result.tip_count;
  for (std::size_t node = 0; node < nodes; ++node) {
    if (!child_edges[node].empty())
      result.node_buffers[node] = next_buffer++;
  }
  int next_temporary = next_buffer;
  next_buffer += CheckedInt(temporary_count, "temporary partial count");
  if (needs_constant)
    result.constant_buffer = next_buffer++;
  result.partials_buffer_count = next_buffer;

  result.operations.reserve(operation_count);
  result.scale_indices.reserve(operation_count);
  int next_scale = 0;
  for (const btrc::Index node : postorder) {
    if (child_edges[node].empty())
      continue;
    const std::vector<btrc::Index> &children = child_edges[node];
    if (children.size() == 1) {
      const btrc::Index edge = children.front();
      const btrc::Index child = model.plan.edge_children()[edge];
      result.scale_indices.push_back(next_scale);
      result.operations.push_back(
          {result.node_buffers[node], next_scale++, BEAGLE_OP_NONE,
           result.node_buffers[child], static_cast<int>(edge),
           result.constant_buffer, CheckedInt(edges, "edge count")});
      continue;
    }

    const btrc::Index first_edge = children[0];
    const btrc::Index second_edge = children[1];
    const btrc::Index first = model.plan.edge_children()[first_edge];
    const btrc::Index second = model.plan.edge_children()[second_edge];
    int accumulator = children.size() == 2 ? result.node_buffers[node]
                                           : next_temporary++;
    result.scale_indices.push_back(next_scale);
    result.operations.push_back(
        {accumulator, next_scale++, BEAGLE_OP_NONE,
         result.node_buffers[first], static_cast<int>(first_edge),
         result.node_buffers[second], static_cast<int>(second_edge)});
    for (std::size_t index = 2; index < children.size(); ++index) {
      const btrc::Index edge = children[index];
      const btrc::Index child = model.plan.edge_children()[edge];
      const int destination = index + 1 == children.size()
                                  ? result.node_buffers[node]
                                  : next_temporary++;
      result.scale_indices.push_back(next_scale);
      result.operations.push_back(
          {destination, next_scale++, BEAGLE_OP_NONE, accumulator,
           CheckedInt(edges, "edge count"), result.node_buffers[child],
           static_cast<int>(edge)});
      accumulator = destination;
    }
  }
  if (next_temporary !=
          CheckedInt(nodes + temporary_count, "partials buffer count") ||
      next_scale != CheckedInt(operation_count, "operation count")) {
    throw std::logic_error("failed to construct the BEAGLE postorder");
  }
  result.root_buffer = result.node_buffers[model.plan.root()];
  result.cumulative_scale = result.operations.empty() ? BEAGLE_OP_NONE
                                                       : next_scale;

  const bool needs_identity = needs_constant || temporary_count != 0;
  result.matrix_indices.resize(edges + (needs_identity ? 1 : 0));
  std::iota(result.matrix_indices.begin(), result.matrix_indices.end(), 0);
  result.tip_partials.resize(static_cast<std::size_t>(result.tip_count));
  result.tip_states.resize(static_cast<std::size_t>(result.tip_count));
  result.tip_observation_indices.resize(
      static_cast<std::size_t>(result.tip_count));
  result.compact_tip.resize(static_cast<std::size_t>(result.tip_count));
  for (std::size_t node = 0; node < nodes; ++node) {
    if (!child_edges[node].empty())
      continue;
    const std::size_t tip =
        static_cast<std::size_t>(result.node_buffers[node]);
    const std::size_t observed = observation_index[node];
    result.tip_observation_indices[tip] = observed;
    const bool compact = compact_observations[observed] != 0;
    result.compact_tip[tip] = compact;
    if (compact)
      result.tip_states[tip].resize(model.sites);
    else
      result.tip_partials[tip].resize(model.sites * 4);
  }
  return result;
}

long PrecisionFlag() {
  if constexpr (std::is_same_v<Scalar, float>)
    return BEAGLE_FLAG_PRECISION_SINGLE;
  return BEAGLE_FLAG_PRECISION_DOUBLE;
}

struct BeagleEvaluation {
  double tip_ms = 0.0;
  double pruning_ms = 0.0;
  double total_ms = 0.0;
  double log_likelihood = 0.0;
};

class BeagleWorkspace {
public:
  BeagleWorkspace(AlignmentModelView model, const std::string &resource,
                  int threads,
                  std::span<const std::uint8_t> compact_observations)
      : tree_(MakeBeagleTree(model, compact_observations)),
        site_values_(model.sites),
        transition_matrices_(tree_.matrix_indices.size() * 16),
        transition_padded_values_(tree_.matrix_indices.size(), 1.0) {
    const bool threaded = resource == "cpu" && threads > 1;
    const long processor = resource == "cpu" ? BEAGLE_FLAG_PROCESSOR_CPU
                                               : BEAGLE_FLAG_PROCESSOR_GPU;
    const long framework = resource == "cpu" ? BEAGLE_FLAG_FRAMEWORK_CPU
                                               : BEAGLE_FLAG_FRAMEWORK_CUDA;
    const long threading = threaded ? BEAGLE_FLAG_THREADING_CPP
                                    : BEAGLE_FLAG_THREADING_NONE;
    const long required =
        PrecisionFlag() | BEAGLE_FLAG_COMPUTATION_SYNCH |
        BEAGLE_FLAG_EIGEN_REAL | BEAGLE_FLAG_SCALING_MANUAL |
        BEAGLE_FLAG_SCALERS_LOG | processor | framework | threading;
    BeagleInstanceDetails details{};
    const int compact_buffer_count = CheckedInt(
        static_cast<std::size_t>(std::count(tree_.compact_tip.begin(),
                                           tree_.compact_tip.end(), true)),
        "compact tip count");
    const int partials_buffer_count =
        tree_.partials_buffer_count - compact_buffer_count;
    instance_ = std::make_unique<BeagleInstance>(beagleCreateInstance(
        tree_.tip_count, partials_buffer_count, compact_buffer_count, 4,
        CheckedInt(model.sites, "site count"), 1,
        CheckedInt(tree_.matrix_indices.size(), "matrix count"), 1,
        tree_.operations.empty()
            ? 0
            : CheckedInt(tree_.operations.size() + 1, "scale count"),
        nullptr, 0,
        required, required, &details));
    resource_name_ = details.resourceName == nullptr ? "unknown"
                                                     : details.resourceName;
    implementation_name_ =
        details.implName == nullptr ? "unknown" : details.implName;
    implementation_flags_ = details.flags;
    if ((details.flags & PrecisionFlag()) == 0) {
      throw std::runtime_error(
          "BEAGLE selected an implementation with the wrong precision");
    }
    if (threaded)
      CheckBeagle(beagleSetCPUThreadCount(instance_->get(), threads),
                  "beagleSetCPUThreadCount");

    if (tree_.constant_buffer != BEAGLE_OP_NONE) {
      const std::vector<double> constant(model.sites * 4, 1.0);
      CheckBeagle(beagleSetPartials(instance_->get(), tree_.constant_buffer,
                                    constant.data()),
                  "beagleSetPartials");
    }

    const std::vector<double> unit_pattern_weights(model.sites, 1.0);
    CheckBeagle(
        beagleSetPatternWeights(instance_->get(), unit_pattern_weights.data()),
        "beagleSetPatternWeights");
    static_cast<void>(UpdateFactors(model));
  }

  double UpdateFactors(AlignmentModelView model) {
    const Clock::time_point begin = Clock::now();
    if (tree_.matrix_indices.size() != model.branch_lengths.size() &&
        tree_.matrix_indices.size() != model.branch_lengths.size() + 1) {
      throw std::invalid_argument("BEAGLE branch-factor shape changed");
    }
    constexpr std::array<double, 1> kCategoryRates{1.0};
    constexpr std::array<double, 1> kCategoryWeights{1.0};
    std::array<double, 4> frequencies{};
    for (std::size_t state = 0; state < frequencies.size(); ++state)
      frequencies[state] = static_cast<double>(model.root_frequencies[state]);
    CheckBeagle(beagleSetStateFrequencies(instance_->get(), 0,
                                          frequencies.data()),
                "beagleSetStateFrequencies");
    CheckBeagle(beagleSetCategoryRates(instance_->get(),
                                       kCategoryRates.data()),
                "beagleSetCategoryRates");
    CheckBeagle(beagleSetCategoryWeights(instance_->get(), 0,
                                         kCategoryWeights.data()),
                "beagleSetCategoryWeights");
    const auto set_transition = [&](std::size_t matrix, Scalar branch_length) {
      const std::array<Scalar, 16> transition =
          JukesCantorTransition(branch_length, model.substitution_rate);
      std::transform(transition.begin(), transition.end(),
                     transition_matrices_.begin() + matrix * 16,
                     [](Scalar value) { return static_cast<double>(value); });
    };
    for (std::size_t edge = 0; edge < model.branch_lengths.size(); ++edge)
      set_transition(edge, model.branch_lengths[edge]);
    if (tree_.matrix_indices.size() > model.branch_lengths.size())
      set_transition(model.branch_lengths.size(), Scalar{0});
    CheckBeagle(
        beagleSetTransitionMatrices(
            instance_->get(), tree_.matrix_indices.data(),
            transition_matrices_.data(), transition_padded_values_.data(),
            CheckedInt(tree_.matrix_indices.size(), "matrix count")),
        "beagleSetTransitionMatrices");
    return Milliseconds(begin, Clock::now());
  }

  BeagleEvaluation Evaluate(AlignmentModelView model, bool update_inputs) {
    const Clock::time_point total_begin = Clock::now();
    if (update_inputs) {
      SetTipData(model);
    }
    const Clock::time_point tip_end = Clock::now();
    CheckBeagle(beagleUpdatePartials(
                    instance_->get(), tree_.operations.data(),
                    CheckedInt(tree_.operations.size(), "operation count"),
                    BEAGLE_OP_NONE),
                "beagleUpdatePartials");
    if (!tree_.operations.empty()) {
      CheckBeagle(beagleResetScaleFactors(instance_->get(),
                                          tree_.cumulative_scale),
                  "beagleResetScaleFactors");
      CheckBeagle(beagleAccumulateScaleFactors(
                      instance_->get(), tree_.scale_indices.data(),
                      CheckedInt(tree_.scale_indices.size(), "scale count"),
                      tree_.cumulative_scale),
                  "beagleAccumulateScaleFactors");
    }
    const int category_weights = 0;
    const int state_frequencies = 0;
    CheckBeagle(beagleCalculateRootLogLikelihoods(
                    instance_->get(), &tree_.root_buffer, &category_weights,
                    &state_frequencies, &tree_.cumulative_scale, 1,
                    &log_likelihood_),
                "beagleCalculateRootLogLikelihoods");
    CheckBeagle(beagleGetSiteLogLikelihoods(instance_->get(),
                                            site_values_.data()),
                "beagleGetSiteLogLikelihoods");
    const Clock::time_point total_end = Clock::now();
    return {Milliseconds(total_begin, tip_end),
            Milliseconds(tip_end, total_end),
            Milliseconds(total_begin, total_end), log_likelihood_};
  }

  std::span<const double> site_values() const { return site_values_; }
  const std::string &resource_name() const { return resource_name_; }
  const std::string &implementation_name() const { return implementation_name_; }
  long implementation_flags() const { return implementation_flags_; }
  std::size_t compact_tip_count() const {
    return static_cast<std::size_t>(
        std::count(tree_.compact_tip.begin(), tree_.compact_tip.end(), true));
  }
  std::size_t partial_tip_count() const {
    return tree_.compact_tip.size() - compact_tip_count();
  }

private:
  void SetTipData(AlignmentModelView model) {
    if (model.sites != site_values_.size() ||
        model.observation_nodes.size() != tree_.tip_partials.size()) {
      throw std::invalid_argument(
          "BEAGLE workspace does not match the alignment batch shape");
    }
    for (std::size_t tip = 0; tip < tree_.tip_partials.size(); ++tip) {
      const std::size_t observed = tree_.tip_observation_indices[tip];
      if (tree_.compact_tip[tip]) {
        std::vector<int> &states = tree_.tip_states[tip];
        for (std::size_t site = 0; site < model.sites; ++site) {
          const std::uint8_t mask = static_cast<std::uint8_t>(
              model.observations[site * model.observation_nodes.size() +
                                 observed]);
          states[site] = mask == 0b1111 ? 4
                       : mask == 0b0001 ? 0
                       : mask == 0b0010 ? 1
                       : mask == 0b0100 ? 2
                                        : 3;
        }
        CheckBeagle(beagleSetTipStates(instance_->get(),
                                       static_cast<int>(tip), states.data()),
                    "beagleSetTipStates");
      } else {
        std::vector<double> &partials = tree_.tip_partials[tip];
        for (std::size_t site = 0; site < model.sites; ++site) {
          const Nucleotide nucleotide =
              model.observations[site * model.observation_nodes.size() +
                                 observed];
          for (std::size_t state = 0; state < 4; ++state) {
            partials[site * 4 + state] =
                AllowsState(nucleotide, state) ? 1.0 : 0.0;
          }
        }
        CheckBeagle(beagleSetTipPartials(instance_->get(),
                                         static_cast<int>(tip), partials.data()),
                    "beagleSetTipPartials");
      }
    }
  }

  BeagleTree tree_;
  std::vector<double> site_values_;
  std::vector<double> transition_matrices_;
  std::vector<double> transition_padded_values_;
  std::unique_ptr<BeagleInstance> instance_;
  std::string resource_name_;
  std::string implementation_name_;
  long implementation_flags_ = 0;
  double log_likelihood_ = 0.0;
};

double BeagleMaxAbsoluteError(std::span<const Scalar> expected,
                              std::span<const double> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double reference = static_cast<double>(expected[index]);
    if (!std::isfinite(reference) || !std::isfinite(actual[index]))
      return std::numeric_limits<double>::infinity();
    result = std::max(result, std::abs(reference - actual[index]));
  }
  return result;
}

double BeagleMaxRelativeError(std::span<const Scalar> expected,
                              std::span<const double> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double reference = static_cast<double>(expected[index]);
    if (!std::isfinite(reference) || !std::isfinite(actual[index]))
      return std::numeric_limits<double>::infinity();
    const double error = std::abs(reference - actual[index]);
    result = std::max(
        result,
        error / std::max(1.0, std::abs(reference)));
  }
  return result;
}

} // namespace
} // namespace parallel_phylogenetics::benchmark

namespace parallel_phylogenetics::benchmark {
namespace {

void RunOne(const BeagleOptions &options, std::size_t replicate) {
    const Problem problem = MakeProblem(options.problem, replicate);
    const AlignmentModelView model{
        problem.plan, problem.sites, problem.branch_lengths,
        problem.observation_nodes, problem.observations};
    const std::size_t site_batch =
        options.problem.site_batch == 0
            ? problem.sites
            : std::min(options.problem.site_batch, problem.sites);
    const std::size_t remainder = problem.sites % site_batch;
    const std::size_t cpu_site_batch = CpuReferenceSiteBatch(model);
    const std::size_t cpu_remainder = problem.sites % cpu_site_batch;
    const bool update_factors =
        options.problem.benchmark_mode != "fixed-model";
    const bool update_inputs =
        options.problem.benchmark_mode == "full-input-update";
    const bool allow_compact_tips =
        options.resource != "cpu" ||
        (!update_inputs && site_batch == model.sites &&
         options.problem.conditioning_ms == 0);
    std::vector<std::uint8_t> compact_observations(
        model.observation_nodes.size(), allow_compact_tips);
    for (std::size_t observed = 0; observed < model.observation_nodes.size();
         ++observed) {
      for (std::size_t site = 0;
           compact_observations[observed] != 0 && site < model.sites; ++site) {
        const std::uint8_t mask = static_cast<std::uint8_t>(
            model.observations[site * model.observation_nodes.size() +
                               observed]);
        compact_observations[observed] =
            mask == 0b0001 || mask == 0b0010 || mask == 0b0100 ||
            mask == 0b1000 || mask == 0b1111;
      }
    }
    const AlignmentModelView first = SiteBatch(model, 0, site_batch);
    SequentialWorkspace full_sequential;
    full_sequential.Reserve(problem.plan, cpu_site_batch);
    BeagleWorkspace full_beagle(first, options.resource, options.threads,
                                compact_observations);
    std::unique_ptr<SequentialWorkspace> tail_sequential;
    std::unique_ptr<BeagleWorkspace> tail_beagle;
    if (cpu_remainder != 0) {
      tail_sequential = std::make_unique<SequentialWorkspace>();
      tail_sequential->Reserve(problem.plan, cpu_remainder);
    }
    if (remainder != 0) {
      const AlignmentModelView tail =
          SiteBatch(model, problem.sites - remainder, remainder);
      tail_beagle = std::make_unique<BeagleWorkspace>(
          tail, options.resource, options.threads, compact_observations);
    }

    std::vector<double> sequential_times(options.problem.repeats, 0.0);
    std::vector<double> tip_times(options.problem.repeats, 0.0);
    std::vector<double> factor_times(options.problem.repeats, 0.0);
    std::vector<double> pruning_times(options.problem.repeats, 0.0);
    std::vector<double> total_times(options.problem.repeats, 0.0);
    std::vector<Scalar> sequential_values(problem.sites);
    std::vector<double> beagle_values(problem.sites);
    const auto run_sequential_complete = [&] {
      const Clock::time_point begin = Clock::now();
      for (std::size_t first_site = 0; first_site < model.sites;
           first_site += cpu_site_batch) {
        const std::size_t count =
            std::min(cpu_site_batch, model.sites - first_site);
        const AlignmentModelView batch = SiteBatch(model, first_site, count);
        SequentialWorkspace &workspace =
            count == cpu_site_batch ? full_sequential : *tail_sequential;
        const auto values = LogLikelihoodsPrepared(batch, workspace);
        std::copy(values.begin(), values.end(),
                  sequential_values.begin() + first_site);
      }
      return Milliseconds(begin, Clock::now());
    };
    const auto condition_sequential = [&] {
      static_cast<void>(run_sequential_complete());
    };
    const auto condition_beagle = [&] {
      static_cast<void>(full_beagle.UpdateFactors(first));
      if (tail_beagle != nullptr) {
        static_cast<void>(tail_beagle->UpdateFactors(
            SiteBatch(model, model.sites - remainder, remainder)));
      }
      for (std::size_t first_site = 0; first_site < model.sites;
           first_site += site_batch) {
        const std::size_t count =
            std::min(site_batch, model.sites - first_site);
        const AlignmentModelView batch = SiteBatch(model, first_site, count);
        BeagleWorkspace &workspace =
            count == site_batch ? full_beagle : *tail_beagle;
        static_cast<void>(workspace.Evaluate(batch, true));
      }
    };
    ConditionInterleaved(options.problem.conditioning_ms,
                         condition_sequential, condition_beagle);

    double initial_staging_ms = 0.0;
    if (update_inputs) {
      // Match the native full-input benchmark's unconditional first-use
      // warmup, including the complete alignment when it spans chunks.
      static_cast<void>(run_sequential_complete());
      static_cast<void>(full_beagle.UpdateFactors(first));
      if (tail_beagle != nullptr) {
        static_cast<void>(tail_beagle->UpdateFactors(
            SiteBatch(model, model.sites - remainder, remainder)));
      }
      for (std::size_t first_site = 0; first_site < model.sites;
           first_site += site_batch) {
        const std::size_t count =
            std::min(site_batch, model.sites - first_site);
        const AlignmentModelView batch = SiteBatch(model, first_site, count);
        BeagleWorkspace &workspace =
            count == site_batch ? full_beagle : *tail_beagle;
        static_cast<void>(workspace.Evaluate(batch, true));
      }
      for (int repeat = 0; repeat < options.problem.repeats; ++repeat) {
        const auto run_sequential = [&] {
          sequential_times[repeat] += run_sequential_complete();
        };
        const auto run_beagle = [&] {
          const Clock::time_point total_begin = Clock::now();
          factor_times[repeat] += full_beagle.UpdateFactors(first);
          if (tail_beagle != nullptr) {
            factor_times[repeat] += tail_beagle->UpdateFactors(
                SiteBatch(model, model.sites - remainder, remainder));
          }
          for (std::size_t first_site = 0; first_site < model.sites;
               first_site += site_batch) {
            const std::size_t count =
                std::min(site_batch, model.sites - first_site);
            const AlignmentModelView batch =
                SiteBatch(model, first_site, count);
            BeagleWorkspace &workspace =
                count == site_batch ? full_beagle : *tail_beagle;
            const BeagleEvaluation result =
                workspace.Evaluate(batch, true);
            tip_times[repeat] += result.tip_ms;
            pruning_times[repeat] += result.pruning_ms;
            const std::span<const double> values = workspace.site_values();
            std::copy(values.begin(), values.end(),
                      beagle_values.begin() + first_site);
          }
          total_times[repeat] += Milliseconds(total_begin, Clock::now());
        };
        if (repeat % 2 == 0) {
          run_sequential();
          run_beagle();
        } else {
          run_beagle();
          run_sequential();
        }
      }
    } else {
    static_cast<void>(run_sequential_complete());
    for (int repeat = 0; repeat < options.problem.repeats; ++repeat)
      sequential_times[repeat] = run_sequential_complete();
    std::size_t chunk_index = 0;
    for (std::size_t first_site = 0; first_site < model.sites;
         first_site += site_batch, ++chunk_index) {
      const std::size_t count =
          std::min(site_batch, model.sites - first_site);
      const AlignmentModelView batch = SiteBatch(model, first_site, count);
      BeagleWorkspace &beagle_workspace =
          count == site_batch ? full_beagle : *tail_beagle;

      const Clock::time_point staging_begin = Clock::now();
      static_cast<void>(beagle_workspace.UpdateFactors(batch));
      static_cast<void>(beagle_workspace.Evaluate(batch, true));
      initial_staging_ms += Milliseconds(staging_begin, Clock::now());

      for (int repeat = 0; repeat < options.problem.repeats; ++repeat) {
        const auto run_beagle = [&] {
          const Clock::time_point total_begin = Clock::now();
          if (update_factors)
            factor_times[repeat] += beagle_workspace.UpdateFactors(batch);
          const BeagleEvaluation result =
              beagle_workspace.Evaluate(batch, update_inputs);
          tip_times[repeat] += result.tip_ms;
          pruning_times[repeat] += result.pruning_ms;
          total_times[repeat] += Milliseconds(total_begin, Clock::now());
          const std::span<const double> values = beagle_workspace.site_values();
          std::copy(values.begin(), values.end(),
                    beagle_values.begin() + first_site);
        };
        run_beagle();
      }
    }
    }

    const double sequential_ms = Median(sequential_times);
    const double beagle_tip_ms = Median(tip_times);
    const double beagle_factor_ms = Median(factor_times);
    const double beagle_pruning_ms = Median(pruning_times);
    const double beagle_total_ms = Median(total_times);
    const double absolute_error = BeagleMaxAbsoluteError(
        std::span<const Scalar>(sequential_values),
        std::span<const double>(beagle_values));
    const double relative_error = BeagleMaxRelativeError(
        std::span<const Scalar>(sequential_values),
        std::span<const double>(beagle_values));
    RequireBenchmarkCorrectness(absolute_error, relative_error);
    double sequential_log_likelihood = 0.0;
    double beagle_log_likelihood = 0.0;
    for (std::size_t pattern = 0; pattern < problem.sites; ++pattern) {
      sequential_log_likelihood +=
          static_cast<double>(problem.pattern_weights[pattern]) *
          static_cast<double>(sequential_values[pattern]);
      beagle_log_likelihood +=
          static_cast<double>(problem.pattern_weights[pattern]) *
          beagle_values[pattern];
    }
    const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
    std::cout << "# baseline=BEAGLE " << beagleGetVersion() << '\n'
              << "# precision=" << tree_hmm::kPrecisionName << '\n'
              << "# resource=" << full_beagle.resource_name() << '\n'
              << "# implementation=" << full_beagle.implementation_name()
              << '\n'
              << "# implementation_flags="
              << full_beagle.implementation_flags()
              << '\n'
              << "# compact_tip_count=" << full_beagle.compact_tip_count()
              << '\n'
              << "# partial_tip_count=" << full_beagle.partial_tip_count()
              << '\n'
              << "# repeated fixed-topology evaluations exclude instance, "
                 "workspace, and topology setup; benchmark_mode determines "
                 "which numerical inputs are refreshed\n"
              << "# benchmark_mode=" << options.problem.benchmark_mode << '\n'
              << "# study=" << options.problem.study << '\n'
              << "# sequence_generation=" << problem.sequence_generation
              << '\n'
              << "# topological_height_edges=" << problem.shape.height
              << '\n';
    if (problem.evolutionary_root_to_tip_distance.has_value()) {
      std::cout << "# evolutionary_root_to_tip_distance="
                << *problem.evolutionary_root_to_tip_distance << '\n';
    }
    std::cout
              << "# minimum_branch_length=" << problem.minimum_branch_length
              << '\n'
              << "# floored_branch_count=" << problem.floored_branch_count
              << '\n'
              << "# timing_beagle_total=host wall time for the input updates "
                 "selected by benchmark_mode, pruning, scaling, root "
                 "integration, and per-pattern likelihood retrieval\n"
              << "# timed_output=per-unique-pattern log likelihoods; pattern "
                 "multiplicities are fixed to one inside BEAGLE and the "
                 "empirical multiplicities are applied outside timing for "
                 "both implementations\n"
              << "# transition_matrices=the native stable closed-form JC69 "
                 "routine constructs matrices in the configured precision; "
                 "they are copied to BEAGLE's double-valued public input and "
                 "uploaded through "
                 "beagleSetTransitionMatrices; construction and upload are "
                 "included whenever factors are timed\n"
              << "# warmup=all modes execute one untimed evaluation before "
                 "measurement; this is not included in initial_staging_ms\n"
              << "# timing_initial_staging=one untimed full-input evaluation "
                 "per exact site chunk for resident modes; reported "
                 "separately and excluded from beagle_total_ms\n"
              << "# measurement_scope="
              << (update_inputs || site_batch == problem.sites
                      ? "complete-alignment-wall-time"
                      : "sum-of-per-chunk-resident-calls")
              << '\n'
              << "# resident_chunking=factor-update and fixed-model stage "
                 "and measure each exact chunk before reusing its workspace; "
                 "their totals are projections unless site_batch equals the "
                 "number of unique patterns\n"
              << "# factor_projection=for multiple chunks, factor-update "
                 "sums one factor refresh per resident chunk; it does not "
                 "estimate a scheme sharing one refreshed factor copy across "
                 "all chunks\n"
              << "# conventional_cpu_reference=full evaluation with an "
                 "independently memory-bounded workspace; it is not a mode-matched "
                 "resident implementation\n"
              << "baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,"
                 "sequence_generation,evolutionary_root_to_tip_distance,minimum_branch_length,floored_branch_count,"
                 "seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,"
                 "site_batch,cpu_reference_site_batch,binary_tree,tree_height,"
                 "sackin_index,colless_index,"
                 "normalized_colless,structural_rounds,primitive_levels,"
                 "primitive_operations,planning_ms,repeats,"
                 "conditioning_ms,threads,beagle_resource_name,"
                 "beagle_implementation,beagle_implementation_flags,"
                 "compact_tip_count,partial_tip_count,"
                 "sequential_ms,beagle_input_ms,beagle_factor_ms,"
                 "beagle_pruning_ms,beagle_total_ms,initial_staging_ms,"
                 "conventional_full_cpu_over_beagle,"
                 "sequential_log_likelihood,beagle_log_likelihood,"
                 "max_abs_error,max_relative_error,sequential_samples_ms,"
                 "beagle_input_samples_ms,beagle_factor_samples_ms,"
                 "beagle_pruning_samples_ms,beagle_total_samples_ms\n"
              << std::setprecision(10) << "beagle," << options.resource << ','
              << tree_hmm::kPrecisionName << ','
              << options.problem.benchmark_mode << ',' << options.problem.study
              << ','
              << problem.dataset << ','
              << problem.topology << ',' << problem.sequence_generation << ',';
    if (problem.evolutionary_root_to_tip_distance.has_value())
      std::cout << *problem.evolutionary_root_to_tip_distance;
    std::cout << ',' << problem.minimum_branch_length << ','
              << problem.floored_branch_count << ',' << problem.base_seed << ','
              << problem.seed << ',' << problem.replicate << ','
              << problem.leaves << ',' << problem.plan.num_nodes() << ','
              << problem.raw_sites << ',' << problem.sites << ',' << site_batch
              << ',' << cpu_site_batch << ','
              << (problem.shape.binary ? 1 : 0) << ','
              << problem.shape.height << ',' << problem.shape.sackin << ','
              << problem.shape.colless << ','
              << problem.shape.normalized_colless << ',' << statistics.rounds
              << ',' << statistics.primitive_levels << ','
              << statistics.rakes + problem.plan.num_branch_combinations() +
                     problem.plan.num_branch_absorptions() +
                     statistics.compressions
              << ',' << problem.planning_ms << ','
              << options.problem.repeats << ','
              << options.problem.conditioning_ms << ',' << options.threads
              << ',' << full_beagle.resource_name() << ','
              << full_beagle.implementation_name() << ','
              << full_beagle.implementation_flags() << ','
              << full_beagle.compact_tip_count() << ','
              << full_beagle.partial_tip_count() << ',' << sequential_ms
              << ',' << beagle_tip_ms << ',' << beagle_factor_ms << ','
              << beagle_pruning_ms << ','
              << beagle_total_ms << ',' << initial_staging_ms << ','
              << sequential_ms / beagle_total_ms << ','
              << sequential_log_likelihood << ','
              << beagle_log_likelihood << ',' << absolute_error << ','
              << relative_error << ',' << JoinSamples(sequential_times) << ','
              << JoinSamples(tip_times) << ','
              << JoinSamples(factor_times) << ','
              << JoinSamples(pruning_times) << ','
              << JoinSamples(total_times) << '\n';
}

} // namespace
} // namespace parallel_phylogenetics::benchmark

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics::benchmark;
    const BeagleOptions options = ParseBeagleOptions(argc, argv);
    const std::size_t stop =
        options.problem.replicate_start + options.problem.replicates;
    if (stop < options.problem.replicate_start)
      throw std::overflow_error("synthetic replicate range overflows");
    for (std::size_t replicate = options.problem.replicate_start;
         replicate < stop; ++replicate) {
      RunOne(options, replicate);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
