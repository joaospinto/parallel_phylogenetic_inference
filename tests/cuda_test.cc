#include "parallel_phylogenetics/cuda.h"

#define PARALLEL_PHYLOGENETICS_GPU_NAMESPACE cuda
#define PARALLEL_PHYLOGENETICS_GPU_NAME "CUDA"
#include "tests/gpu_backend_test.inc"
#undef PARALLEL_PHYLOGENETICS_GPU_NAME
#undef PARALLEL_PHYLOGENETICS_GPU_NAMESPACE
