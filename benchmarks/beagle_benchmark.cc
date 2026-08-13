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

void CheckBeagle(int code, const char *operation) {
  if (code != BEAGLE_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with BEAGLE error " +
                             std::to_string(code));
  }
}

class BeagleInstance {
public:
  explicit BeagleInstance(int instance) : instance_(instance) {
    if (instance_ < 0) {
      throw std::runtime_error("beagleCreateInstance failed with error " +
                               std::to_string(instance_));
    }
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
  std::vector<double> edge_lengths;
  std::vector<BeagleOperation> operations;
  std::vector<std::vector<double>> tip_partials;
  std::vector<std::size_t> tip_observation_indices;
  int tip_count = 0;
  int partials_buffer_count = 0;
  int constant_buffer = BEAGLE_OP_NONE;
  int root_buffer = BEAGLE_OP_NONE;
  int cumulative_scale = BEAGLE_OP_NONE;
};

BeagleTree MakeBeagleTree(AlignmentModelView model) {
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
  result.edge_lengths.reserve(result.matrix_indices.size());
  for (const Scalar length : model.branch_lengths) {
    result.edge_lengths.push_back(static_cast<double>(
        length * model.substitution_rate));
  }
  if (needs_identity)
    result.edge_lengths.push_back(0.0);

  result.tip_partials.resize(static_cast<std::size_t>(result.tip_count));
  result.tip_observation_indices.resize(
      static_cast<std::size_t>(result.tip_count));
  for (std::size_t node = 0; node < nodes; ++node) {
    if (!child_edges[node].empty())
      continue;
    const std::size_t tip =
        static_cast<std::size_t>(result.node_buffers[node]);
    std::vector<double> &partials = result.tip_partials[tip];
    partials.resize(model.sites * 4);
    result.tip_observation_indices[tip] = observation_index[node];
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
  double transition_ms = 0.0;
  double pruning_ms = 0.0;
  double total_ms = 0.0;
  double log_likelihood = 0.0;
};

class BeagleWorkspace {
public:
  BeagleWorkspace(AlignmentModelView model, const std::string &resource,
                  int threads)
      : tree_(MakeBeagleTree(model)), site_values_(model.sites) {
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
    instance_ = std::make_unique<BeagleInstance>(beagleCreateInstance(
        tree_.tip_count, tree_.partials_buffer_count, 0, 4,
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

    SetTipPartials(model);
    if (tree_.constant_buffer != BEAGLE_OP_NONE) {
      const std::vector<double> constant(model.sites * 4, 1.0);
      CheckBeagle(beagleSetPartials(instance_->get(), tree_.constant_buffer,
                                    constant.data()),
                  "beagleSetPartials");
    }

    constexpr std::array<double, 16> kEigenvectors{
        1.0,  2.0, 0.0,  0.5, 1.0, -2.0, 0.5, 0.0,
        1.0,  2.0, 0.0, -0.5, 1.0, -2.0, -0.5, 0.0};
    constexpr std::array<double, 16> kInverseEigenvectors{
        0.25, 0.25,   0.25, 0.25, 0.125, -0.125, 0.125, -0.125,
        0.0,  1.0,    0.0,  -1.0, 1.0,   0.0,    -1.0,  0.0};
    constexpr std::array<double, 4> kEigenvalues{
        0.0, -4.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0};
    std::array<double, 4> frequencies{};
    for (std::size_t state = 0; state < frequencies.size(); ++state)
      frequencies[state] = static_cast<double>(model.root_frequencies[state]);
    constexpr std::array<double, 1> kCategoryRates{1.0};
    constexpr std::array<double, 1> kCategoryWeights{1.0};
    const std::vector<double> pattern_weights(model.sites, 1.0);
    CheckBeagle(beagleSetEigenDecomposition(
                    instance_->get(), 0, kEigenvectors.data(),
                    kInverseEigenvectors.data(), kEigenvalues.data()),
                "beagleSetEigenDecomposition");
    CheckBeagle(beagleSetStateFrequencies(instance_->get(), 0,
                                          frequencies.data()),
                "beagleSetStateFrequencies");
    CheckBeagle(beagleSetCategoryRates(instance_->get(),
                                       kCategoryRates.data()),
                "beagleSetCategoryRates");
    CheckBeagle(beagleSetCategoryWeights(instance_->get(), 0,
                                         kCategoryWeights.data()),
                "beagleSetCategoryWeights");
    CheckBeagle(beagleSetPatternWeights(instance_->get(),
                                        pattern_weights.data()),
                "beagleSetPatternWeights");
  }

  double UpdateTransitionMatrices() {
    const Clock::time_point begin = Clock::now();
    CheckBeagle(beagleUpdateTransitionMatrices(
                    instance_->get(), 0, tree_.matrix_indices.data(), nullptr,
                    nullptr, tree_.edge_lengths.data(),
                    CheckedInt(tree_.matrix_indices.size(), "matrix count")),
                "beagleUpdateTransitionMatrices");
    return Milliseconds(begin, Clock::now());
  }

  BeagleEvaluation Evaluate(AlignmentModelView model, bool update_tips) {
    const Clock::time_point total_begin = Clock::now();
    if (update_tips)
      SetTipPartials(model);
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
            0.0,
            Milliseconds(tip_end, total_end),
            Milliseconds(total_begin, total_end), log_likelihood_};
  }

  std::span<const double> site_values() const { return site_values_; }
  const std::string &resource_name() const { return resource_name_; }
  const std::string &implementation_name() const { return implementation_name_; }
  long implementation_flags() const { return implementation_flags_; }

private:
  void SetTipPartials(AlignmentModelView model) {
    if (model.sites != site_values_.size() ||
        model.observation_nodes.size() != tree_.tip_partials.size()) {
      throw std::invalid_argument(
          "BEAGLE workspace does not match the alignment batch shape");
    }
    for (std::size_t tip = 0; tip < tree_.tip_partials.size(); ++tip) {
      std::vector<double> &partials = tree_.tip_partials[tip];
      const std::size_t observed = tree_.tip_observation_indices[tip];
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

  BeagleTree tree_;
  std::vector<double> site_values_;
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
  for (std::size_t index = 0; index < expected.size(); ++index)
    result = std::max(result, std::abs(static_cast<double>(expected[index]) -
                                      actual[index]));
  return result;
}

double BeagleMaxRelativeError(std::span<const Scalar> expected,
                              std::span<const double> actual) {
  if (expected.size() != actual.size())
    throw std::invalid_argument("benchmark output shapes do not match");
  double result = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double error =
        std::abs(static_cast<double>(expected[index]) - actual[index]);
    result = std::max(
        result,
        error / std::max(1.0, std::abs(static_cast<double>(expected[index]))));
  }
  return result;
}

} // namespace
} // namespace parallel_phylogenetics::benchmark

int main(int argc, char **argv) {
  try {
    using namespace parallel_phylogenetics;
    using namespace parallel_phylogenetics::benchmark;
    const BeagleOptions options = ParseBeagleOptions(argc, argv);
    const Problem problem = MakeProblem(options.problem);
    const AlignmentModelView model{
        problem.plan, problem.sites, problem.branch_lengths,
        problem.observation_nodes, problem.observations};
    const std::size_t site_batch =
        options.problem.site_batch == 0
            ? problem.sites
            : std::min(options.problem.site_batch, problem.sites);
    const std::size_t remainder = problem.sites % site_batch;
    const bool chunked = site_batch != problem.sites;
    const AlignmentModelView first = SiteBatch(model, 0, site_batch);
    SequentialWorkspace full_sequential;
    full_sequential.Reserve(problem.plan, site_batch);
    BeagleWorkspace full_beagle(first, options.resource, options.threads);
    std::unique_ptr<SequentialWorkspace> tail_sequential;
    std::unique_ptr<BeagleWorkspace> tail_beagle;
    if (remainder != 0) {
      const AlignmentModelView tail =
          SiteBatch(model, problem.sites - remainder, remainder);
      tail_sequential = std::make_unique<SequentialWorkspace>();
      tail_sequential->Reserve(problem.plan, remainder);
      tail_beagle = std::make_unique<BeagleWorkspace>(
          tail, options.resource, options.threads);
    }

    std::vector<double> sequential_times;
    std::vector<double> tip_times;
    std::vector<double> transition_times;
    std::vector<double> pruning_times;
    std::vector<double> total_times;
    sequential_times.reserve(options.problem.repeats);
    tip_times.reserve(options.problem.repeats);
    transition_times.reserve(options.problem.repeats);
    pruning_times.reserve(options.problem.repeats);
    total_times.reserve(options.problem.repeats);
    std::vector<Scalar> sequential_values(problem.sites);
    std::vector<double> beagle_values(problem.sites);
    double beagle_log_likelihood = 0.0;
    const auto run_sequential = [&] {
      double elapsed = 0.0;
      for (std::size_t first_site = 0; first_site < model.sites;
           first_site += site_batch) {
        const std::size_t count =
            std::min(site_batch, model.sites - first_site);
        const AlignmentModelView batch = SiteBatch(model, first_site, count);
        SequentialWorkspace &workspace =
            count == site_batch ? full_sequential : *tail_sequential;
        const Clock::time_point begin = Clock::now();
        const std::span<const Scalar> values =
            LogLikelihoodsPrepared(batch, workspace);
        elapsed += Milliseconds(begin, Clock::now());
        std::copy(values.begin(), values.end(),
                  sequential_values.begin() + first_site);
      }
      sequential_times.push_back(elapsed);
    };
    const auto run_beagle = [&] {
      BeagleEvaluation aggregate;
      const Clock::time_point total_begin = Clock::now();
      aggregate.transition_ms += full_beagle.UpdateTransitionMatrices();
      if (tail_beagle != nullptr)
        aggregate.transition_ms += tail_beagle->UpdateTransitionMatrices();
      for (std::size_t first_site = 0; first_site < model.sites;
           first_site += site_batch) {
        const std::size_t count =
            std::min(site_batch, model.sites - first_site);
        const AlignmentModelView batch = SiteBatch(model, first_site, count);
        BeagleWorkspace &workspace =
            count == site_batch ? full_beagle : *tail_beagle;
        const BeagleEvaluation result = workspace.Evaluate(batch, chunked);
        aggregate.tip_ms += result.tip_ms;
        aggregate.transition_ms += result.transition_ms;
        aggregate.pruning_ms += result.pruning_ms;
        aggregate.total_ms += result.total_ms;
        aggregate.log_likelihood += result.log_likelihood;
        const std::span<const double> values = workspace.site_values();
        std::copy(values.begin(), values.end(),
                  beagle_values.begin() + first_site);
      }
      aggregate.total_ms = Milliseconds(total_begin, Clock::now());
      beagle_log_likelihood = aggregate.log_likelihood;
      tip_times.push_back(aggregate.tip_ms);
      transition_times.push_back(aggregate.transition_ms);
      pruning_times.push_back(aggregate.pruning_ms);
      total_times.push_back(aggregate.total_ms);
    };

    run_sequential();
    run_beagle();
    ConditionInterleaved(options.problem.conditioning_ms, run_sequential,
                         run_beagle);
    sequential_times.clear();
    tip_times.clear();
    transition_times.clear();
    pruning_times.clear();
    total_times.clear();
    for (int repeat = 0; repeat < options.problem.repeats; ++repeat) {
      if (repeat % 2 == 0) {
        run_sequential();
        run_beagle();
      } else {
        run_beagle();
        run_sequential();
      }
    }

    const double sequential_ms = Median(sequential_times);
    const double beagle_tip_ms = Median(tip_times);
    const double beagle_transition_ms = Median(transition_times);
    const double beagle_pruning_ms = Median(pruning_times);
    const double beagle_total_ms = Median(total_times);
    const double absolute_error = BeagleMaxAbsoluteError(
        std::span<const Scalar>(sequential_values),
        std::span<const double>(beagle_values));
    const double relative_error = BeagleMaxRelativeError(
        std::span<const Scalar>(sequential_values),
        std::span<const double>(beagle_values));
    const double sequential_log_likelihood =
        std::accumulate(sequential_values.begin(), sequential_values.end(),
                        0.0);
    const btrc::PlanStatistics statistics = btrc::Statistics(problem.plan);
    std::cout << "# baseline=BEAGLE " << beagleGetVersion() << '\n'
              << "# precision=" << tree_hmm::kPrecisionName << '\n'
              << "# resource=" << full_beagle.resource_name() << '\n'
              << "# implementation=" << full_beagle.implementation_name()
              << '\n'
              << "# implementation_flags="
              << full_beagle.implementation_flags()
              << '\n'
              << "# repeated fixed-topology evaluations exclude instance, "
                 "workspace, topology, and substitution-model setup\n"
              << "# tip data remain resident for unchunked calls; chunked "
                 "totals include tip updates\n"
              << "# each BEAGLE instance updates JC69 transition matrices "
                 "once per full alignment evaluation; total times also "
                 "include pruning, scaling, root integration, and site "
                 "likelihoods\n"
              << "baseline,precision,dataset,topology,leaves,nodes,sites,"
                 "site_batch,"
                 "primitive_levels,repeats,conditioning_ms,threads,"
                 "sequential_ms,beagle_tip_ms,beagle_transition_ms,"
                 "beagle_pruning_ms,beagle_total_ms,sequential_over_beagle,"
                 "sequential_log_likelihood,beagle_log_likelihood,"
                 "max_abs_error,max_relative_error\n"
              << std::setprecision(10) << "beagle," << tree_hmm::kPrecisionName
              << ',' << problem.dataset << ',' << problem.topology << ','
              << problem.leaves << ',' << problem.plan.num_nodes() << ','
              << problem.sites << ',' << site_batch << ','
              << statistics.primitive_levels << ','
              << options.problem.repeats << ','
              << options.problem.conditioning_ms << ',' << options.threads
              << ',' << sequential_ms << ',' << beagle_tip_ms << ','
              << beagle_transition_ms << ',' << beagle_pruning_ms << ','
              << beagle_total_ms << ','
              << sequential_ms / beagle_total_ms << ','
              << sequential_log_likelihood << ','
              << beagle_log_likelihood << ',' << absolute_error << ','
              << relative_error << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
