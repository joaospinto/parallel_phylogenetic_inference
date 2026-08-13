#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" ]]; then
  root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
else
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
report="${root}/tests/benchmark_report_fixture.txt"
# shellcheck source=scripts/benchmark_resume.sh
source "${root}/scripts/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${root}/scripts/capacity_bounded.sh"
# shellcheck source=scripts/accelerator_environment.sh
source "${root}/scripts/accelerator_environment.sh"

tree_hmm_version_at_least 580.159.04 525.60.13
tree_hmm_version_at_least 525.60.13 525.60.13
tree_hmm_version_at_least 526.0 525.60.13
! tree_hmm_version_at_least 525.59.99 525.60.13
TREE_HMM_CUDA_DRIVER_VERSION_OVERRIDE=580.159.04 \
  tree_hmm_check_cuda_driver_compatibility >/dev/null
if TREE_HMM_CUDA_DRIVER_VERSION_OVERRIDE=520.0 \
  tree_hmm_check_cuda_driver_compatibility >/dev/null 2>&1; then
  echo "an incompatible CUDA driver was accepted" >&2
  exit 1
fi

benchmark_resume_case_completed "${report}" cuda FP64 synthetic balanced \
  256 1024 1024
! benchmark_resume_case_completed "${report}" cuda FP32 synthetic balanced \
  256 1024 1024
benchmark_resume_case_completed "${report}" beagle-cpu FP64 synthetic \
  caterpillar 64 256 256
benchmark_resume_dataset_batch_completed "${report}" cuda FP64 \
  actinopt_12k_raxml 2048
! benchmark_resume_dataset_batch_completed "${report}" cuda FP64 \
  actinopt_12k_raxml 4096
benchmark_resume_dataset_completed "${report}" metal FP32 PF00004
benchmark_resume_dataset_completed "${report}" rocm FP32 PF00004
benchmark_resume_capacity_reached "${report}" cuda FP64 4096
benchmark_resume_capacity_reached "${report}" cuda FP64 8192
! benchmark_resume_capacity_reached "${report}" cuda FP64 2048
benchmark_resume_validation_completed "${report}" FP32
benchmark_resume_validation_completed "${report}" FP64
! benchmark_resume_validation_completed "${report}" FP16

work_directory="$(mktemp -d "${TMPDIR:-/tmp}/benchmark-shell-test.XXXXXX")"
trap 'rm -rf "${work_directory}"' EXIT
new_report="${work_directory}/new-report.txt"
cat > "${new_report}" <<'EOF'
backend,precision,dataset,topology,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch
cuda,FP32,synthetic,yule,20260813,91,0,128,255,64,64,64
cuda,FP32,synthetic,yule,20260813,92,2,128,255,64,64,64
baseline,beagle_resource,precision,benchmark_mode,dataset,topology,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch,threads
beagle,cuda,FP32,factor-update,synthetic,yule,20260813,91,0,128,255,64,64,64,1
EOF
benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 0
! benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 1
benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 2
benchmark_resume_synthetic_replicate_completed "${new_report}" beagle-cuda \
  FP32 yule 128 64 20260813 0 factor-update 1
! benchmark_resume_synthetic_replicate_completed "${new_report}" beagle-cuda \
  FP32 yule 128 64 20260813 0 fixed-model 1
! benchmark_resume_synthetic_replicate_completed "${new_report}" beagle-cuda \
  FP32 yule 128 64 20260813 0 factor-update 2
mock_rocm="${work_directory}/rocm"
mkdir -p "${mock_rocm}/bin" "${mock_rocm}/lib"
printf '%s\n' '#!/usr/bin/env bash' 'echo "  Name: gfx942"' > \
  "${mock_rocm}/bin/rocminfo"
chmod +x "${mock_rocm}/bin/rocminfo"
TREE_HMM_ALLOW_MISSING_KFD_FOR_TESTS=1 \
  tree_hmm_check_rocm_driver_compatibility "${mock_rocm}" gfx942 \
  >/dev/null
if TREE_HMM_ALLOW_MISSING_KFD_FOR_TESTS=1 \
  tree_hmm_check_rocm_driver_compatibility "${mock_rocm}" gfx90a \
  >/dev/null 2>&1; then
  echo "a missing ROCm target architecture was accepted" >&2
  exit 1
fi
mock_bazel_output="${work_directory}/bazel-output"
mock_bazel_execroot="${mock_bazel_output}/execroot/main"
mock_rocm_repository="${mock_bazel_output}/external/mock_rocm_sdk"
mock_rocm_root="${mock_rocm_repository}/opt/rocm-7.2.3"
mkdir -p "${mock_bazel_execroot}" "${mock_rocm_root}/lib/llvm/bin" \
  "${mock_rocm_root}/bin" "${mock_rocm_root}/toolchain/bin"
cat > "${mock_rocm_root}/lib/llvm/bin/amdllvm" <<'EOF'
#!/usr/bin/env bash
case "$(basename "$0")" in
  amdclang|amdclang++) exit 0 ;;
  *) exit 1 ;;
esac
EOF
chmod +x "${mock_rocm_root}/lib/llvm/bin/amdllvm"
ln -s amdllvm "${mock_rocm_root}/lib/llvm/bin/amdclang"
ln -s amdllvm "${mock_rocm_root}/lib/llvm/bin/amdclang++"
for executable in amdclang amdclang++; do
  cat > "${mock_rocm_root}/toolchain/bin/${executable}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
sdk_root="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd -P)"
entry_point="\${sdk_root}/lib/llvm/bin/${executable}"
exec -a "\${entry_point}" "\${entry_point}" "\$@"
EOF
  chmod +x "${mock_rocm_root}/toolchain/bin/${executable}"
done
cp "${mock_rocm}/bin/rocminfo" "${mock_rocm_root}/bin/rocminfo"
mock_bazel="${work_directory}/bazel"
cat > "${mock_bazel}" <<EOF
#!/usr/bin/env bash
case "\$1" in
  cquery)
    echo external/mock_rocm_sdk/opt/rocm-7.2.3/toolchain/bin/amdclang
    ;;
  info)
    case "\${!#}" in
      execution_root) echo "${mock_bazel_execroot}" ;;
      output_base) echo "${mock_bazel_output}" ;;
      *) exit 2 ;;
    esac
    ;;
  *) exit 2 ;;
esac
EOF
chmod +x "${mock_bazel}"
tree_hmm_resolve_rocm_sdk "${mock_bazel}"
[[ "${TREE_HMM_RESOLVED_AMDCLANG}" == \
   "${mock_rocm_root}/toolchain/bin/amdclang" ]]
[[ "${TREE_HMM_RESOLVED_ROCM_PATH}" == "${mock_rocm_root}" ]]
"${TREE_HMM_RESOLVED_AMDCLANG}" --version
"${TREE_HMM_RESOLVED_AMDCLANGXX}" --version
benchmark_capacity_exhausted=0
success_output="${work_directory}/success.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 1 \
  bash -c 'printf success' > "${success_output}"
[[ "$(<"${success_output}")" == success ]]
[[ "${benchmark_capacity_exhausted}" == 0 ]]

benchmark_capacity_exhausted=0
allocation_output="${work_directory}/allocation.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 2 \
  bash -c 'echo std::bad_alloc >&2; exit 1' > "${allocation_output}" 2>&1
grep -Fq "first_infeasible_site_batch=2" "${allocation_output}"
[[ "${benchmark_capacity_exhausted}" == 1 ]]

benchmark_capacity_exhausted=0
killed_output="${work_directory}/killed.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 4 \
  bash -c 'kill -9 $$' > "${killed_output}" 2>&1
grep -Fq "first_infeasible_site_batch=4" "${killed_output}"
[[ "${benchmark_capacity_exhausted}" == 1 ]]

guard_output="$(benchmark_run_capacity_bounded "${work_directory}" 262144 \
  beagle-cpu FP32 8 bash -c 'ulimit -v')"
[[ "${guard_output}" == 262144 ]]

benchmark_capacity_exhausted=0
segfault_output="${work_directory}/segfault.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 \
  beagle-cpu FP32 16 bash -c 'kill -s SEGV $$' \
  > "${segfault_output}" 2>&1
grep -Fq "first_infeasible_site_batch=16" "${segfault_output}"
grep -Fq "reason=beagle-segfault-under-memory-limit" "${segfault_output}"
[[ "${benchmark_capacity_exhausted}" == 1 ]]

if benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 32 \
  bash -c 'kill -s SEGV $$' >/dev/null 2>&1; then
  echo "a CUDA segmentation fault was misclassified as a capacity limit" >&2
  exit 1
else
  status=$?
  [[ "${status}" == 139 ]]
fi

if benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 64 \
  bash -c 'exit 42'; then
  echo "an unrelated failure was misclassified as a capacity limit" >&2
  exit 1
else
  status=$?
  [[ "${status}" == 42 ]]
fi

launcher_root="${work_directory}/launcher"
source_root="${launcher_root}/input/sources"
working_root="${launcher_root}/working"
mkdir -p "${source_root}/parallel_phylogenetic_inference/scripts" \
  "${source_root}/parallel_tree_hmm" \
  "${source_root}/bidirectional_tree_rake_compress" "${working_root}"
cp "${root}/scripts/kaggle_cuda_notebook.sh" \
  "${source_root}/parallel_phylogenetic_inference/scripts/"
cp "${root}/scripts/kaggle_accelerator_notebook.sh" \
  "${source_root}/parallel_phylogenetic_inference/scripts/"
cp "${root}/scripts/accelerator_environment.sh" \
  "${source_root}/parallel_phylogenetic_inference/scripts/"
printf '%s\n' \
  'parallel_phylogenetic_inference test-revision' \
  'parallel_tree_hmm test-revision' \
  'bidirectional_tree_rake_compress test-revision' > \
  "${source_root}/SOURCE_REVISIONS.txt"
cat > "${source_root}/parallel_phylogenetic_inference/scripts/notebook_cuda.sh" <<'EOF'
#!/usr/bin/env bash
printf 'stub precision=%s skip_fish=%s\n' \
  "${TREE_HMM_PRECISIONS}" "${TREE_HMM_SKIP_FISH_TREE:-0}"
printf 'stub sections=%s\n' "${TREE_HMM_BENCHMARK_SECTIONS:-}"
EOF
cat > "${source_root}/parallel_phylogenetic_inference/scripts/notebook_rocm.sh" <<'EOF'
#!/usr/bin/env bash
printf 'rocm stub precision=%s\n' "${TREE_HMM_PRECISIONS}"
EOF
cache_identity="$({
  echo 'parallel-phylogenetics-benchmark-schema=3'
  echo 'accelerator-backend=cuda'
  sort "${source_root}/SOURCE_REVISIONS.txt"
} | shasum -a 256 | awk '{ print $1 }')"
printf '# cache_identity sha256=%s\nprevious-row\n' "${cache_identity}" > \
  "${source_root}/PREVIOUS_BENCHMARK_REPORT.txt"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_PRECISIONS_OVERRIDE=FP32 TREE_HMM_SKIP_FISH_TREE=1 \
TREE_HMM_BENCHMARK_SECTIONS="fish pandit" \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_root}" \
  > "${launcher_root}/unpacked.log"
grep -Fq 'using unpacked source input' "${launcher_root}/unpacked.log"
grep -Fq 'previous-row' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub precision=FP32 skip_fish=1' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub sections=fish pandit' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_ACCELERATOR_BACKEND_OVERRIDE=rocm \
TREE_HMM_PRECISIONS_OVERRIDE=FP32 \
  bash "${root}/scripts/kaggle_accelerator_notebook.sh" "${source_root}" \
  > "${launcher_root}/rocm.log"
grep -Fq 'selected notebook accelerator: rocm' \
  "${launcher_root}/rocm.log"
grep -Fq 'rocm stub precision=FP32' \
  "${working_root}/parallel_phylogenetics_rocm_report.txt"
printf 'working-only-row\n' >> \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"

source_zip="${launcher_root}/parallel_tree_inference_sources.zip"
(
  cd "${source_root}"
  zip -q -r "${source_zip}" .
)
rm -rf "${working_root}/parallel-tree-inference"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_PRECISIONS_OVERRIDE=FP64 TREE_HMM_SKIP_FISH_TREE=0 \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_zip}" \
  > "${launcher_root}/zip.log"
grep -Fq 'source bundle SHA-256:' "${launcher_root}/zip.log"
grep -Fq 'lines in the working report' "${launcher_root}/zip.log"
grep -Fq 'working-only-row' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub precision=FP64 skip_fish=0' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
