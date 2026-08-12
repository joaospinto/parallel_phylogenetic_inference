# Parallel Phylogenetic Inference

This repository implements phylogenetic likelihood and ancestral-state
inference on top of two independent reusable packages:

- [`parallel_tree_hmm`](../parallel_tree_hmm) supplies allocation-free hidden
  Markov tree inference on CPU, Metal, and CUDA;
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
- prepared workspaces that allocate before, rather than during, repeated
  likelihood evaluations;
- native Metal and CUDA benchmark executables with numerical cross-checks.

The accelerator kernels currently use FP32, while the conventional CPU
baseline uses FP64. Every benchmark reports the maximum absolute discrepancy
between their per-site log likelihoods.

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

Both benchmark binaries also accept an empirical nucleotide data set:

```sh
bazel run //:metal_benchmark -- \
  --newick family.nwk --fasta family.fasta --repeats 5
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
