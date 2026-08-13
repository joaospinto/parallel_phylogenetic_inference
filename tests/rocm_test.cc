#include "parallel_phylogenetics/rocm.h"

#define PARALLEL_PHYLOGENETICS_GPU_NAMESPACE rocm
#define PARALLEL_PHYLOGENETICS_GPU_NAME "ROCm"
#include "tests/gpu_backend_test.inc"
#undef PARALLEL_PHYLOGENETICS_GPU_NAME
#undef PARALLEL_PHYLOGENETICS_GPU_NAMESPACE
