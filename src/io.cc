#include "parallel_phylogenetics/io.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace parallel_phylogenetics {
namespace {

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("failed to open " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

class NewickParser {
public:
  explicit NewickParser(std::string_view text) : text_(text) {}

  Phylogeny Parse() {
    const btrc::Index root = ParseSubtree(std::nullopt);
    SkipIgnored();
    Require(';');
    SkipIgnored();
    if (position_ != text_.size())
      Fail("unexpected text after the Newick terminator");
    if (root != 0)
      Fail("internal parser error: root is not the first node");

    btrc::Plan plan = btrc::MakePlan(parents_, root);
    std::vector<double> edge_lengths(plan.num_edges());
    for (std::size_t edge = 0; edge < plan.num_edges(); ++edge)
      edge_lengths[edge] = lengths_[plan.edge_children()[edge]];
    return {std::move(plan), std::move(edge_lengths), std::move(labels_)};
  }

private:
  btrc::Index ParseSubtree(std::optional<btrc::Index> parent) {
    SkipIgnored();
    if (parents_.size() > std::numeric_limits<btrc::Index>::max())
      Fail("the Newick tree has too many nodes");
    const btrc::Index node = static_cast<btrc::Index>(parents_.size());
    parents_.push_back(parent.has_value() ? static_cast<std::int64_t>(*parent)
                                          : std::int64_t{-1});
    labels_.emplace_back();
    lengths_.push_back(0.0);

    if (Peek('(')) {
      Require('(');
      ParseSubtree(node);
      SkipIgnored();
      while (Peek(',')) {
        Require(',');
        ParseSubtree(node);
        SkipIgnored();
      }
      Require(')');
      labels_[node] = ParseOptionalLabel();
    } else {
      labels_[node] = ParseOptionalLabel();
      if (labels_[node].empty())
        Fail("a leaf must have a label");
    }

    SkipIgnored();
    if (Peek(':')) {
      Require(':');
      lengths_[node] = ParseLength();
    }
    return node;
  }

  std::string ParseOptionalLabel() {
    SkipIgnored();
    if (position_ == text_.size())
      return {};
    if (text_[position_] == '\'') {
      ++position_;
      std::string result;
      while (position_ < text_.size()) {
        const char character = text_[position_++];
        if (character != '\'') {
          result.push_back(character);
          continue;
        }
        if (position_ < text_.size() && text_[position_] == '\'') {
          result.push_back('\'');
          ++position_;
          continue;
        }
        return result;
      }
      Fail("unterminated quoted Newick label");
    }

    const std::size_t begin = position_;
    while (position_ < text_.size()) {
      const char character = text_[position_];
      if (character == '(' || character == ')' || character == ',' ||
          character == ':' || character == ';' || character == '[' ||
          std::isspace(static_cast<unsigned char>(character))) {
        break;
      }
      ++position_;
    }
    return std::string(text_.substr(begin, position_ - begin));
  }

  double ParseLength() {
    SkipIgnored();
    const std::size_t begin = position_;
    while (position_ < text_.size()) {
      const char character = text_[position_];
      if (character == ',' || character == ')' || character == ';' ||
          character == '[' ||
          std::isspace(static_cast<unsigned char>(character))) {
        break;
      }
      ++position_;
    }
    const std::string token(text_.substr(begin, position_ - begin));
    char *parsed_end = nullptr;
    errno = 0;
    const double value = std::strtod(token.c_str(), &parsed_end);
    if (token.empty() || parsed_end != token.c_str() + token.size() ||
        errno == ERANGE || !std::isfinite(value) || value < 0.0) {
      Fail("branch lengths must be finite and nonnegative");
    }
    return value;
  }

  void SkipIgnored() {
    while (position_ < text_.size()) {
      if (std::isspace(static_cast<unsigned char>(text_[position_]))) {
        ++position_;
        continue;
      }
      if (text_[position_] != '[')
        break;
      ++position_;
      int depth = 1;
      while (position_ < text_.size() && depth != 0) {
        if (text_[position_] == '[')
          ++depth;
        else if (text_[position_] == ']')
          --depth;
        ++position_;
      }
      if (depth != 0)
        Fail("unterminated Newick comment");
    }
  }

  bool Peek(char expected) {
    SkipIgnored();
    return position_ < text_.size() && text_[position_] == expected;
  }

  void Require(char expected) {
    SkipIgnored();
    if (position_ == text_.size() || text_[position_] != expected)
      Fail(std::string("expected '") + expected + "'");
    ++position_;
  }

  [[noreturn]] void Fail(const std::string &message) const {
    throw std::invalid_argument("Newick parse error at byte " +
                                std::to_string(position_) + ": " + message);
  }

  std::string_view text_;
  std::size_t position_ = 0;
  std::vector<std::int64_t> parents_;
  std::vector<std::string> labels_;
  std::vector<double> lengths_;
};

std::string RecordName(std::string_view header) {
  const std::size_t first = header.find_first_not_of(" \t\r");
  if (first == std::string_view::npos)
    throw std::invalid_argument("a FASTA header has no record name");
  const std::size_t last = header.find_first_of(" \t\r", first);
  return std::string(header.substr(first, last - first));
}

Nucleotide Decode(char character) {
  switch (
      static_cast<char>(std::toupper(static_cast<unsigned char>(character)))) {
  case 'A':
    return Nucleotide::kA;
  case 'C':
    return Nucleotide::kC;
  case 'G':
    return Nucleotide::kG;
  case 'T':
  case 'U':
    return Nucleotide::kT;
  case 'R':
  case 'Y':
  case 'S':
  case 'W':
  case 'K':
  case 'M':
  case 'B':
  case 'D':
  case 'H':
  case 'V':
  case 'N':
  case 'X':
  case '?':
  case '-':
  case '.':
    return Nucleotide::kUnknown;
  default:
    throw std::invalid_argument(std::string("invalid FASTA symbol '") +
                                character + "'");
  }
}

} // namespace

Phylogeny ParseNewick(std::string_view text) {
  return NewickParser(text).Parse();
}

Phylogeny LoadNewick(const std::filesystem::path &path) {
  return ParseNewick(ReadFile(path));
}

SequenceAlignment ParseFasta(std::string_view text) {
  SequenceAlignment result;
  std::size_t position = 0;
  SequenceRecord *current = nullptr;
  while (position <= text.size()) {
    const std::size_t end = text.find('\n', position);
    const std::size_t line_end =
        end == std::string_view::npos ? text.size() : end;
    std::string_view line = text.substr(position, line_end - position);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (!line.empty() && line.front() == '>') {
      result.records.push_back({RecordName(line.substr(1)), {}});
      current = &result.records.back();
    } else {
      for (const char character : line) {
        if (std::isspace(static_cast<unsigned char>(character)))
          continue;
        if (current == nullptr)
          throw std::invalid_argument("FASTA sequence appears before a header");
        static_cast<void>(Decode(character));
        current->sequence.push_back(static_cast<char>(
            std::toupper(static_cast<unsigned char>(character))));
      }
    }
    if (end == std::string_view::npos)
      break;
    position = end + 1;
  }
  if (result.records.empty())
    throw std::invalid_argument("the FASTA input contains no records");
  result.sites = result.records.front().sequence.size();
  if (result.sites == 0)
    throw std::invalid_argument("FASTA sequences must not be empty");
  std::map<std::string, bool> names;
  for (const SequenceRecord &record : result.records) {
    if (record.sequence.size() != result.sites)
      throw std::invalid_argument("all FASTA sequences must have equal length");
    if (!names.emplace(record.name, true).second)
      throw std::invalid_argument("duplicate FASTA record name " + record.name);
  }
  return result;
}

SequenceAlignment LoadFasta(const std::filesystem::path &path) {
  return ParseFasta(ReadFile(path));
}

EncodedAlignment EncodeAlignment(const Phylogeny &phylogeny,
                                 const SequenceAlignment &alignment) {
  std::map<std::string, const std::string *> sequences;
  for (const SequenceRecord &record : alignment.records)
    sequences.emplace(record.name, &record.sequence);

  std::vector<std::size_t> out_degree(phylogeny.plan.num_nodes(), 0);
  for (const btrc::Index parent : phylogeny.plan.edge_parents())
    ++out_degree[parent];
  std::vector<bool> used(alignment.records.size(), false);
  std::map<std::string, std::size_t> record_indices;
  for (std::size_t index = 0; index < alignment.records.size(); ++index)
    record_indices.emplace(alignment.records[index].name, index);

  EncodedAlignment result;
  result.sites = alignment.sites;
  result.observations.assign(alignment.sites * phylogeny.plan.num_nodes(),
                             Nucleotide::kUnknown);
  std::map<std::string, bool> leaf_labels;
  for (std::size_t node = 0; node < phylogeny.plan.num_nodes(); ++node) {
    if (out_degree[node] != 0)
      continue;
    const std::string &label = phylogeny.labels[node];
    if (label.empty())
      throw std::invalid_argument("every phylogenetic leaf must have a label");
    if (!leaf_labels.emplace(label, true).second)
      throw std::invalid_argument("duplicate phylogenetic leaf label " + label);
    const auto sequence = sequences.find(label);
    if (sequence == sequences.end())
      throw std::invalid_argument("no FASTA record matches leaf " + label);
    used[record_indices.at(label)] = true;
    for (std::size_t site = 0; site < alignment.sites; ++site) {
      result.observations[site * phylogeny.plan.num_nodes() + node] =
          Decode((*sequence->second)[site]);
    }
  }
  if (std::find(used.begin(), used.end(), false) != used.end()) {
    throw std::invalid_argument(
        "the FASTA input contains a record absent from the phylogeny");
  }
  return result;
}

} // namespace parallel_phylogenetics
