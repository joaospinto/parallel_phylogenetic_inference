#include "parallel_phylogenetics/alignment.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace parallel_phylogenetics {

struct AlignmentWorkspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t sites = 0;
  std::vector<Scalar> nodes;
  std::vector<Scalar> edges;
};

struct SequentialWorkspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t sites = 0;
  std::vector<btrc::Index> postorder;
  std::vector<std::size_t> child_offsets;
  std::vector<btrc::Index> child_edges;
  std::vector<Scalar> transitions;
  std::vector<Scalar> partials;
  std::vector<Scalar> log_scales;
  std::vector<Scalar> output;
};

namespace {

std::size_t CheckedProduct(std::initializer_list<std::size_t> values,
                           const char *description) {
  std::size_t result = 1;
  for (const std::size_t value : values) {
    if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value)
      throw std::length_error(std::string(description) + " overflows size_t");
    result *= value;
  }
  return result;
}

void ValidateProbabilities(const std::array<Scalar, 4> &frequencies) {
  const Scalar sum =
      std::accumulate(frequencies.begin(), frequencies.end(), Scalar{0});
  const Scalar probability_tolerance =
      Scalar{16} * std::numeric_limits<Scalar>::epsilon();
  if (!std::isfinite(sum) ||
      std::abs(sum - Scalar{1}) > probability_tolerance ||
      std::any_of(frequencies.begin(), frequencies.end(), [](Scalar value) {
        return !std::isfinite(value) || value < 0.0;
      })) {
    throw std::invalid_argument(
        "root frequencies must be finite, nonnegative, and sum to one");
  }
}

void ValidateModel(AlignmentModelView model) {
  if (model.branch_lengths.size() != model.plan.num_edges())
    throw std::invalid_argument("one branch length is required per plan edge");
  const std::size_t expected_observations = CheckedProduct(
      {model.sites, model.observation_nodes.size()}, "observations");
  if (model.observations.size() != expected_observations)
    throw std::invalid_argument("alignment observations have the wrong shape");
  if (model.observation_nodes.empty())
    throw std::invalid_argument("an alignment must observe at least one node");
  btrc::Index previous = 0;
  bool first = true;
  for (const btrc::Index node : model.observation_nodes) {
    if (node >= model.plan.num_nodes() || (!first && node <= previous)) {
      throw std::invalid_argument(
          "alignment observation nodes must be valid and strictly "
          "increasing");
    }
    previous = node;
    first = false;
  }
  if (!(model.substitution_rate >= 0.0) ||
      !std::isfinite(model.substitution_rate)) {
    throw std::invalid_argument(
        "the substitution rate must be finite and nonnegative");
  }
  ValidateProbabilities(model.root_frequencies);
}

void ValidateWorkspace(AlignmentModelView model,
                       const btrc::Plan *reserved_plan,
                       std::size_t reserved_sites, const char *workspace_name) {
  if (reserved_plan != &model.plan || reserved_sites != model.sites) {
    throw std::invalid_argument(
        std::string("prepared alignment inference requires ") + workspace_name +
        "::Reserve for this plan and site count");
  }
  ValidateModel(model);
}

tree_hmm::BatchedModelView
FillFactors(AlignmentModelView model,
            tree_hmm::MutableBatchedModelView destination) {
  std::fill(destination.node_potentials.begin(),
            destination.node_potentials.end(), 1.0f);
  for (std::size_t site = 0; site < model.sites; ++site) {
    Scalar *root = destination.node_potentials.data() +
                   (site * model.plan.num_nodes() + model.plan.root()) * 4;
    std::transform(model.root_frequencies.begin(), model.root_frequencies.end(),
                   root,
                   [](Scalar value) { return static_cast<Scalar>(value); });
    for (std::size_t index = 0; index < model.observation_nodes.size();
         ++index) {
      const btrc::Index node = model.observation_nodes[index];
      const Nucleotide observation =
          model.observations[site * model.observation_nodes.size() + index];
      if (observation == Nucleotide::kUnknown)
        continue;
      const std::uint8_t mask = static_cast<std::uint8_t>(observation);
      if (mask == 0 || mask > static_cast<std::uint8_t>(Nucleotide::kUnknown))
        throw std::invalid_argument("invalid nucleotide observation");
      Scalar *factor = destination.node_potentials.data() +
                       (site * model.plan.num_nodes() + node) * 4;
      for (int state = 0; state < 4; ++state) {
        if (!AllowsState(observation, state))
          factor[state] = 0.0f;
      }
    }
  }

  for (std::size_t edge = 0; edge < model.plan.num_edges(); ++edge) {
    const std::array<Scalar, 16> transition = JukesCantorTransition(
        model.branch_lengths[edge], model.substitution_rate);
    std::transform(transition.begin(), transition.end(),
                   destination.edge_potentials.begin() + edge * 16,
                   [](Scalar value) { return static_cast<Scalar>(value); });
  }
  return destination;
}

} // namespace

AlignmentModelView SelectSites(AlignmentModelView model, std::size_t first_site,
                               std::size_t site_count) {
  ValidateModel(model);
  if (site_count == 0 || first_site > model.sites ||
      site_count > model.sites - first_site)
    throw std::invalid_argument("invalid phylogenetic alignment site range");
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

AlignmentWorkspace::AlignmentWorkspace() : impl_(std::make_unique<Impl>()) {}
AlignmentWorkspace::~AlignmentWorkspace() = default;
AlignmentWorkspace::AlignmentWorkspace(AlignmentWorkspace &&) noexcept =
    default;
AlignmentWorkspace &
AlignmentWorkspace::operator=(AlignmentWorkspace &&) noexcept = default;

void AlignmentWorkspace::Reserve(const btrc::Plan &plan, std::size_t sites) {
  if (sites == 0)
    throw std::invalid_argument("an alignment must contain at least one site");
  Impl &storage = *impl_;
  storage.plan = &plan;
  storage.sites = sites;
  storage.nodes.resize(CheckedProduct({sites, plan.num_nodes(), std::size_t{4}},
                                      "alignment node factors"));
  storage.edges.resize(CheckedProduct({plan.num_edges(), std::size_t{16}},
                                      "alignment edge factors"));
}

tree_hmm::BatchedModelView Prepare(AlignmentModelView model,
                                   AlignmentWorkspace &workspace) {
  AlignmentWorkspace::Impl &storage = *workspace.impl_;
  ValidateWorkspace(model, storage.plan, storage.sites, "AlignmentWorkspace");
  return FillFactors(
      model, {model.plan, 4, model.sites, storage.nodes, storage.edges});
}

tree_hmm::BatchedModelView
Prepare(AlignmentModelView model,
        tree_hmm::MutableBatchedModelView destination) {
  ValidateModel(model);
  const std::size_t expected_nodes =
      CheckedProduct({model.sites, model.plan.num_nodes(), std::size_t{4}},
                     "alignment node factors");
  const std::size_t expected_edges = CheckedProduct(
      {model.plan.num_edges(), std::size_t{16}}, "alignment edge factors");
  if (&destination.plan != &model.plan || destination.states != 4 ||
      destination.batch != model.sites ||
      destination.node_potentials.size() != expected_nodes ||
      destination.edge_potentials.size() != expected_edges) {
    throw std::invalid_argument(
        "alignment factors do not match the destination model view");
  }
  return FillFactors(model, destination);
}

tree_hmm::BatchedCategoricalModelView
Prepare(AlignmentModelView model,
        tree_hmm::MutableBatchedCategoricalModelView destination) {
  ValidateModel(model);
  constexpr std::size_t kCategories = 16;
  const std::size_t expected_observations = CheckedProduct(
      {model.sites, model.observation_nodes.size()}, "observations");
  const std::size_t expected_edges = CheckedProduct(
      {model.plan.num_edges(), std::size_t{16}}, "alignment edge factors");
  if (&destination.plan != &model.plan || destination.states != 4 ||
      destination.batch != model.sites ||
      destination.categories != kCategories ||
      !std::equal(destination.observation_nodes.begin(),
                  destination.observation_nodes.end(),
                  model.observation_nodes.begin(),
                  model.observation_nodes.end()) ||
      destination.observations.size() != expected_observations ||
      destination.root_potential.size() != 4 ||
      destination.emission_potentials.size() != kCategories * 4 ||
      destination.edge_potentials.size() != expected_edges) {
    throw std::invalid_argument(
        "categorical alignment factors do not match the destination model "
        "view");
  }
  static_assert(sizeof(Nucleotide) == sizeof(std::uint8_t));
  std::memcpy(destination.observations.data(), model.observations.data(),
              model.observations.size_bytes());
  std::transform(model.root_frequencies.begin(), model.root_frequencies.end(),
                 destination.root_potential.begin(),
                 [](Scalar value) { return static_cast<Scalar>(value); });
  for (std::size_t category = 0; category < kCategories; ++category) {
    for (std::size_t state = 0; state < 4; ++state) {
      destination.emission_potentials[category * 4 + state] =
          (category & (std::size_t{1} << state)) != 0 ? 1.0f : 0.0f;
    }
  }
  for (std::size_t edge = 0; edge < model.plan.num_edges(); ++edge) {
    const std::array<Scalar, 16> transition = JukesCantorTransition(
        model.branch_lengths[edge], model.substitution_rate);
    std::transform(transition.begin(), transition.end(),
                   destination.edge_potentials.begin() + edge * 16,
                   [](Scalar value) { return static_cast<Scalar>(value); });
  }
  return destination;
}

SequentialWorkspace::SequentialWorkspace() : impl_(std::make_unique<Impl>()) {}
SequentialWorkspace::~SequentialWorkspace() = default;
SequentialWorkspace::SequentialWorkspace(SequentialWorkspace &&) noexcept =
    default;
SequentialWorkspace &
SequentialWorkspace::operator=(SequentialWorkspace &&) noexcept = default;

void SequentialWorkspace::Reserve(const btrc::Plan &plan, std::size_t sites) {
  if (sites == 0)
    throw std::invalid_argument("an alignment must contain at least one site");
  Impl &storage = *impl_;
  storage.plan = &plan;
  storage.sites = sites;
  storage.child_offsets.assign(plan.num_nodes() + 1, 0);
  for (const btrc::Index parent : plan.edge_parents())
    ++storage.child_offsets[parent + 1];
  std::partial_sum(storage.child_offsets.begin(), storage.child_offsets.end(),
                   storage.child_offsets.begin());
  storage.child_edges.resize(plan.num_edges());
  std::vector<std::size_t> cursor(storage.child_offsets.begin(),
                                  storage.child_offsets.end() - 1);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    const btrc::Index parent = plan.edge_parents()[edge];
    storage.child_edges[cursor[parent]++] = static_cast<btrc::Index>(edge);
  }

  storage.postorder.clear();
  storage.postorder.reserve(plan.num_nodes());
  std::vector<btrc::Index> stack{plan.root()};
  while (!stack.empty()) {
    const btrc::Index node = stack.back();
    stack.pop_back();
    storage.postorder.push_back(node);
    for (std::size_t index = storage.child_offsets[node];
         index < storage.child_offsets[node + 1]; ++index) {
      stack.push_back(plan.edge_children()[storage.child_edges[index]]);
    }
  }
  std::reverse(storage.postorder.begin(), storage.postorder.end());
  storage.transitions.resize(plan.num_edges() * 16);
  storage.partials.resize(CheckedProduct(
      {sites, plan.num_nodes(), std::size_t{4}}, "sequential partials"));
  storage.log_scales.resize(
      CheckedProduct({sites, plan.num_nodes()}, "sequential scales"));
  storage.output.resize(sites);
}

std::span<const Scalar> LogLikelihoodsPrepared(AlignmentModelView model,
                                               SequentialWorkspace &workspace) {
  SequentialWorkspace::Impl &storage = *workspace.impl_;
  ValidateWorkspace(model, storage.plan, storage.sites, "SequentialWorkspace");
  for (std::size_t edge = 0; edge < model.plan.num_edges(); ++edge) {
    const std::array<Scalar, 16> transition = JukesCantorTransition(
        model.branch_lengths[edge], model.substitution_rate);
    std::copy(transition.begin(), transition.end(),
              storage.transitions.begin() + edge * 16);
  }
  std::fill(storage.partials.begin(), storage.partials.end(), 1.0);
  std::fill(storage.log_scales.begin(), storage.log_scales.end(), 0.0);

  for (std::size_t site = 0; site < model.sites; ++site) {
    Scalar *site_partials =
        storage.partials.data() + site * model.plan.num_nodes() * 4;
    Scalar *root = site_partials + model.plan.root() * 4;
    std::copy(model.root_frequencies.begin(), model.root_frequencies.end(),
              root);
    for (std::size_t index = 0; index < model.observation_nodes.size();
         ++index) {
      const btrc::Index node = model.observation_nodes[index];
      const Nucleotide observation =
          model.observations[site * model.observation_nodes.size() + index];
      if (observation == Nucleotide::kUnknown)
        continue;
      const std::uint8_t mask = static_cast<std::uint8_t>(observation);
      if (mask == 0 || mask > static_cast<std::uint8_t>(Nucleotide::kUnknown))
        throw std::invalid_argument("invalid nucleotide observation");
      Scalar *factor = site_partials + node * 4;
      for (int state = 0; state < 4; ++state) {
        if (!AllowsState(observation, state))
          factor[state] = 0.0;
      }
    }

    Scalar *site_scales =
        storage.log_scales.data() + site * model.plan.num_nodes();
    for (const btrc::Index node : storage.postorder) {
      Scalar input_scale = 0.0;
      Scalar *partial = site_partials + node * 4;
      for (std::size_t child_index = storage.child_offsets[node];
           child_index < storage.child_offsets[node + 1]; ++child_index) {
        const btrc::Index edge = storage.child_edges[child_index];
        const btrc::Index child = model.plan.edge_children()[edge];
        const Scalar *child_partial = site_partials + child * 4;
        const Scalar *transition = storage.transitions.data() + edge * 16;
        for (std::size_t parent_state = 0; parent_state < 4; ++parent_state) {
          Scalar message = 0.0;
          for (std::size_t child_state = 0; child_state < 4; ++child_state) {
            message += transition[parent_state * 4 + child_state] *
                       child_partial[child_state];
          }
          partial[parent_state] *= message;
        }
        input_scale += site_scales[child];
      }
      const Scalar maximum = *std::max_element(partial, partial + 4);
      if (maximum > 0.0) {
        for (std::size_t state = 0; state < 4; ++state)
          partial[state] /= maximum;
        site_scales[node] = input_scale + std::log(maximum);
      } else {
        site_scales[node] = -std::numeric_limits<Scalar>::infinity();
      }
    }
    const Scalar *root_partial = site_partials + model.plan.root() * 4;
    const Scalar root_sum =
        std::accumulate(root_partial, root_partial + 4, Scalar{0});
    storage.output[site] =
        root_sum > 0.0 ? site_scales[model.plan.root()] + std::log(root_sum)
                       : -std::numeric_limits<Scalar>::infinity();
  }
  return storage.output;
}

} // namespace parallel_phylogenetics
