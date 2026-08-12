# Parallel Phylogenetic Inference

Phylogenetic likelihood and ancestral-state inference built on
`parallel_tree_hmm`, which in turn consumes the separate native bidirectional
rake–compress runtime. Biological models and file formats belong here; neither
the tree-HMM algebra nor the scheduler is duplicated.

The initial vertical slice implements a single-site Jukes–Cantor model:

- likelihood evaluation on any rooted phylogeny;
- posterior nucleotide probabilities at all ancestral nodes;
- posterior parent–child nucleotide probabilities on every branch;
- cross-validation against an independent Felsenstein-pruning reference.

Next milestones are stable per-site scaling, alignment batching, standard
nucleotide substitution models, Newick/FASTA input, CUDA and Metal exposure,
and realistic comparisons with established phylogenetics software.

## Build

Keep this repository beside `bidirectional_tree_rake_compress` and
`parallel_tree_hmm`, then run:

```sh
bazel test //...
```

## License

MIT
