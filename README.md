# Parallel Phylogenetic Inference

This repository implements phylogenetic likelihood and ancestral-state
inference on top of two independent reusable packages:

- [`parallel_tree_hmm`](../parallel_tree_hmm) supplies prepared hidden Markov
  tree inference on CPU, Metal, and CUDA;
- [`bidirectional_tree_rake_compress`](../bidirectional_tree_rake_compress)
  supplies the topology planner and bidirectional rake–compress executor.

Phylogenetic models, alignments, and biological file formats live here. The
tree-HMM algebra and scheduler are not duplicated.

## Implemented vertical slice

The current implementation provides:

- JC69 likelihoods and log likelihoods on arbitrary rooted trees;
- node and edge posterior probabilities for individual sites;
- batched alignment conversion for the generic tree-HMM accelerator API;
- conventional scaled Felsenstein pruning as an independent CPU baseline;
- Newick and nucleotide FASTA input;
- preallocated numerical workspaces for repeated likelihood evaluations;
- phylogenetics-facing CUDA and Metal workspaces that select the appropriate
  tree-HMM representation and handle capacity-bounded site batches;
- native Metal and CUDA benchmark executables with numerical cross-checks.

Alignment conversion writes directly into caller-provided accelerator input
storage. CUDA uses pinned host buffers and Metal uses shared buffers, so the
same generic tree-HMM call does not make a second full-batch staging copy.

The accelerator kernels currently use FP32, while the conventional CPU
baseline uses FP64. Every benchmark reports the maximum absolute discrepancy
between their per-site log likelihoods.

Applications do not need to prepare generic tree-HMM factors themselves. For
example, a prepared CUDA evaluation is:

```cpp
#include "parallel_phylogenetics/cuda.h"

parallel_phylogenetics::cuda::Workspace workspace;
workspace.Reserve(model, 4096);
std::span<const float> log_likelihoods =
    parallel_phylogenetics::cuda::LogLikelihoodsPrepared(model, workspace);
```

The corresponding Metal interface differs only in the namespace and has no
device argument. CUDA stages compact categorical observations; Metal writes
dense factors directly into shared host/device storage. Both call the same
backend-neutral tree-HMM contraction executor, and repeated prepared calls do
not resize numerical workspace storage or reconstruct the topology plan.

## Build and test

Keep this repository beside `bidirectional_tree_rake_compress` and
`parallel_tree_hmm`. Bazel resolves both through local module overrides.

```sh
bazel test //...
```

On an Apple system with Metal:

```sh
bazel run //:metal_benchmark -- \
  --topology balanced --leaves 4096 --sites 1024 --repeats 5
```

On a CUDA system, replace `sm_75` with the device's compute capability:

```sh
bazel run --config=cuda --cuda_archs=sm_75 //:cuda_benchmark -- \
  --topology balanced --leaves 4096 --sites 1024 --repeats 5
```

Both benchmark binaries accept empirical FASTA or relaxed sequential PHYLIP
alignments:

```sh
bazel run //:metal_benchmark -- \
  --newick family.nwk --fasta family.fasta --repeats 5
```

Long alignments can be evaluated in a capacity-bounded prepared workspace:

```sh
bazel run //:metal_benchmark -- \
  --newick tree.nwk --phylip alignment.phy --site-batch 1024 --repeats 5
```

Warmup and workspace allocation are excluded. CPU and accelerator execution
order alternates between repetitions to reduce order and thermal bias. The
reported total accelerator time includes conversion from the phylogenetic
model to generic tree-HMM factors, host/device transfer, kernel execution, and
result transfer.

## Public data

PANDIT 17.0 contains 7,738 protein-domain families with nucleotide alignments
and corresponding inferred phylogenies. To download the official archive,
verify its checksum, and extract DNA Newick/FASTA pairs:

```sh
scripts/fetch_pandit.sh
```

By default, columns marked unreliable by PANDIT's HMM mask are removed. The
extractor accepts family, leaf-count, and result-count filters; run
`scripts/extract_pandit.py --help` for details. Downloaded data are ignored by
Git and retain the terms of the [PANDIT data
license](https://www.ebi.ac.uk/goldman-srv/pandit/Pandit/data/COPYRIGHT).

The Fish Tree of Life supplies a much larger empirical nucleotide case with
11,638 taxa and 24,143 sites. `scripts/fetch_fish_tree.sh` downloads and
checksum-verifies its published RAxML tree and full alignment. The CUDA
notebook evaluates the complete alignment at several prepared site-batch
capacities; it does not truncate or replicate the data.

## Reproducible accelerator runs

`scripts/benchmark_metal.sh` runs the standard local scaling sweep. For CUDA,
`notebooks/kaggle_cuda_benchmark.ipynb` records machine information, executes
the host and emulated-kernel tests, validates the native CUDA implementation
with Compute Sanitizer, and then runs the same balanced and caterpillar sweep.

Until these repositories are public, create the notebook input bundle from
clean commits:

```sh
scripts/package_notebook_sources.sh
```

No GitHub remote or push is required for that workflow.

## License

MIT
