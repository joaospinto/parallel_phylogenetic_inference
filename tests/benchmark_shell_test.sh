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
backend,precision,benchmark_mode,study,dataset,topology,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch
cuda,FP32,factor-update,independent-taxa-pattern-grid,synthetic,yule,20260813,91,0,128,255,64,64,64
cuda,FP32,fixed-model,independent-taxa-pattern-grid,synthetic,yule,20260813,92,2,128,255,64,64,64
cuda,FP32,factor-update,standard,synthetic,yule,20260813,91,0,128,255,64,64,64
cuda,FP32,factor-update,independent-taxa-pattern-grid,synthetic,yule,20260813,93,0,129,257,64,64,64
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch,threads
beagle,cuda,FP32,factor-update,independent-taxa-pattern-grid,synthetic,yule,20260813,91,0,128,255,64,64,64,1
EOF
benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 0 factor-update
! benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 0 fixed-model
! benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 1
benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 2 fixed-model
benchmark_resume_synthetic_replicate_completed "${new_report}" beagle-cuda \
  FP32 yule 128 64 20260813 0 factor-update 1
! benchmark_resume_synthetic_replicate_completed "${new_report}" beagle-cuda \
  FP32 yule 128 64 20260813 0 fixed-model 1
! benchmark_resume_synthetic_replicate_completed "${new_report}" beagle-cuda \
  FP32 yule 128 64 20260813 0 factor-update 2
benchmark_resume_synthetic_replicate_completed "${new_report}" cuda FP32 \
  yule 128 64 20260813 0 factor-update "" standard
benchmark_resume_case_completed "${new_report}" cuda FP32 synthetic yule \
  129 64 64 factor-update "" independent-taxa-pattern-grid
if benchmark_resume_case_completed "${new_report}" cuda FP32 synthetic yule \
  129 64 64 factor-update "" standard; then
  echo "an independent distribution row satisfied a standard-study resume" >&2
  exit 1
fi

mixed_report="${work_directory}/mixed-report.txt"
cat > "${mixed_report}" <<'EOF'
backend,precision,task,dataset,topology,seed_base,seed,replicate,leaves,nodes,unique_patterns,tree_height,normalized_colless,primitive_levels,benchmark_mode,planning_ms,workspace_setup_ms,repeats,median_ms,p25_ms,p75_ms,samples_ms
cuda,FP32,likelihood,synthetic,yule,20260813,91,0,128,255,64,18,0.1,24,full-input-update,0.2,0.3,3,1.0,0.9,1.1,0.9;1.0;1.1
backend,precision,benchmark_mode,dataset,topology,leaves,nodes,sites,unique_patterns,site_batch,cpu_ms,measured_total_ms,max_abs_error,max_relative_error
cuda,FP32,full-input-update,synthetic,yule,128,255,64,64,64,2.0,1.0,0.001,0.0001
EOF
python3 "${root}/scripts/summarize_benchmarks.py" "${mixed_report}" \
  --precision FP32 --benchmark-mode full-input-update \
  > "${work_directory}/mixed-summary.csv"
grep -Fq 'cuda' "${work_directory}/mixed-summary.csv"
! grep -Fq 'likelihood' "${work_directory}/mixed-summary.csv"

coverage_report="${work_directory}/coverage-report.txt"
cat > "${coverage_report}" <<'EOF'
backend,precision,benchmark_mode,study,dataset,topology,leaves,nodes,sites,unique_patterns,site_batch,cpu_ms,measured_total_ms,max_abs_error,max_relative_error
cuda,FP32,full-input-update,coverage-fixture,problem-a,empirical,4,7,8,8,8,2,1,0,0
cuda,FP32,full-input-update,coverage-fixture,problem-b,empirical,4,7,8,8,8,2,1,0,0
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,leaves,nodes,sites,unique_patterns,site_batch,threads,sequential_ms,beagle_total_ms,max_abs_error,max_relative_error
beagle,cpu,FP32,full-input-update,coverage-fixture,problem-a,empirical,4,7,8,8,8,1,2,1.5,0,0
EOF
head -n 3 "${coverage_report}" > "${work_directory}/no-baseline-report.txt"
if python3 "${root}/scripts/summarize_benchmarks.py" \
     "${work_directory}/no-baseline-report.txt" --corpus cuda \
     --required-baseline beagle-cpu --precision FP32 \
     --benchmark-mode full-input-update --max-abs-error 0 \
     --max-relative-error 0 >/dev/null 2>&1; then
  echo "corpus summary accepted a completely absent baseline" >&2
  exit 1
fi
if python3 "${root}/scripts/summarize_benchmarks.py" "${coverage_report}" \
     --corpus cuda --precision FP32 --benchmark-mode full-input-update \
     --required-baseline beagle-cpu \
     --max-abs-error 0 --max-relative-error 0 >/dev/null 2>&1; then
  echo "corpus summary accepted incomplete baseline coverage" >&2
  exit 1
fi
cat >> "${coverage_report}" <<'EOF'
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,leaves,nodes,sites,unique_patterns,site_batch,threads,sequential_ms,beagle_total_ms,max_abs_error,max_relative_error
beagle,cpu,FP32,full-input-update,coverage-fixture,problem-b,empirical,4,7,8,8,8,1,2,1.5,0,0
EOF
python3 "${root}/scripts/summarize_benchmarks.py" "${coverage_report}" \
  --corpus cuda --precision FP32 --benchmark-mode full-input-update \
  --required-baseline beagle-cpu \
  --max-abs-error 0 --max-relative-error 0 > \
  "${work_directory}/coverage-summary.csv"
grep -Fq 'cuda/beagle_cpu_1t' "${work_directory}/coverage-summary.csv"

manifest_directory="${work_directory}/generic-corpus"
mkdir -p "${manifest_directory}"
for name in small.fa small.weights small.nwk large.fa large.weights large.nwk; do
  printf '%s\n' fixture > "${manifest_directory}/${name}"
done
fixture_sha256="$(shasum -a 256 "${manifest_directory}/small.fa" | awk '{print $1}')"
cat > "${manifest_directory}/manifest.csv" <<EOF
dataset,taxa,raw_sites,unique_patterns,alignment,pattern_weights,tree,normalized_alignment_sha256,pattern_weights_sha256,normalized_tree_sha256
small,4,100,100,small.fa,small.weights,small.nwk,${fixture_sha256},${fixture_sha256},${fixture_sha256}
large,10,100000,100000,large.fa,large.weights,large.nwk,${fixture_sha256},${fixture_sha256},${fixture_sha256}
EOF
printf '%s\n' 'corpus_name=generic-test' > \
  "${manifest_directory}/corpus_metadata.txt"
manifest_resume="${manifest_directory}/prior-report.txt"
manifest_sha256="$(shasum -a 256 "${manifest_directory}/manifest.csv" | awk '{print $1}')"
manifest_study="empirical-manifest-${manifest_sha256}-minbrlen-0.000001"
cat > "${manifest_resume}" <<EOF
backend,precision,benchmark_mode,study,dataset,topology,leaves,nodes,sites,unique_patterns,site_batch,max_abs_error,max_relative_error
metal,FP32,full-input-update,${manifest_study},small,empirical,4,7,100,100,100,0,0
# capacity_limit method=metal precision=FP32 dataset=large study=${manifest_study} benchmark_mode=full-input-update threads=none first_infeasible_site_batch=1024 reason=allocation-failure
EOF
TREE_HMM_DRY_RUN=1 TREE_HMM_HOST_MEMORY_GUARD_KIB=262144 \
TREE_HMM_EMPIRICAL_SITE_BATCHES='256 1024 4096' \
TREE_HMM_RESUME_REPORT="${manifest_resume}" \
  bash "${root}/scripts/benchmark_empirical_manifest.sh" metal \
    "${manifest_directory}/manifest.csv" > \
    "${manifest_directory}/dry-run.txt"
grep -Fq '# corpus_name=generic-test' "${manifest_directory}/dry-run.txt"
grep -Fq '# minimum_branch_length=0.000001' \
  "${manifest_directory}/dry-run.txt"
grep -Fq '# planned_cases=5' "${manifest_directory}/dry-run.txt"
grep -Fq '# resume_skip method=metal precision=FP32 dataset=small site_batch=100' \
  "${manifest_directory}/dry-run.txt"
grep -Fq '# resume_capacity_limit method=metal precision=FP32 dataset=large site_batch=1024' \
  "${manifest_directory}/dry-run.txt"
! grep -Fq 'dataset=large taxa=10 raw_sites=100000 unique_patterns=100000 site_batch=4096' \
  "${manifest_directory}/dry-run.txt"
if TREE_HMM_DRY_RUN=1 TREE_HMM_HOST_MEMORY_GUARD_KIB=262144 \
   TREE_HMM_EMPIRICAL_SITE_BATCHES='1024 256' \
   bash "${root}/scripts/benchmark_empirical_manifest.sh" metal \
     "${manifest_directory}/manifest.csv" >/dev/null 2>&1; then
  echo "empirical driver accepted a descending site-batch grid" >&2
  exit 1
fi
printf '%s\n' changed > "${manifest_directory}/small.fa"
if python3 "${root}/scripts/empirical_manifest_rows.py" \
     "${manifest_directory}/manifest.csv" >/dev/null 2>&1; then
  echo "empirical manifest accepted modified prepared data" >&2
  exit 1
fi
printf '%s\n' fixture > "${manifest_directory}/small.fa"
cat >> "${new_report}" <<'EOF'
backend,precision,benchmark_mode,study,dataset,topology,sequence_generation,evolutionary_root_to_tip_distance,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch
cuda,FP32,full-input-update,clock-like-jc69-simulation,synthetic-jc69,yule,jc69,0.001,20260814,910,2,128,255,1024,813,813
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,sequence_generation,evolutionary_root_to_tip_distance,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch,threads
beagle,cpu,FP32,full-input-update,clock-like-jc69-simulation,synthetic-jc69,yule,jc69,0.001,20260814,910,2,128,255,1024,813,813,4
EOF
benchmark_resume_jc69_replicate_completed "${new_report}" cuda FP32 yule \
  128 1024 0.001 20260814 2
! benchmark_resume_jc69_replicate_completed "${new_report}" cuda FP32 yule \
  128 1024 0.01 20260814 2
! benchmark_resume_jc69_replicate_completed "${new_report}" cuda FP32 yule \
  128 1024 0.001 20260814 2 factor-update
! benchmark_resume_jc69_replicate_completed "${new_report}" cuda FP32 yule \
  128 1024 0.001 20260814 3
benchmark_resume_jc69_replicate_completed "${new_report}" beagle-cpu FP32 \
  yule 128 1024 0.001 20260814 2 full-input-update 4
! benchmark_resume_jc69_replicate_completed "${new_report}" beagle-cpu FP32 \
  yule 128 1024 0.001 20260814 2 factor-update 4

TREE_HMM_DRY_RUN=1 TREE_HMM_HOST_MEMORY_GUARD_KIB=262144 \
TREE_HMM_EMPIRICAL_SITE_BATCHES='256 1024 4096' \
  bash "${root}/scripts/benchmark_empirical_manifest.sh" metal \
    "${manifest_directory}/manifest.csv" > \
    "${manifest_directory}/complete-dry-run.txt"
grep -Fq 'dataset=large taxa=10 raw_sites=100000 unique_patterns=100000 site_batch=100000' \
  "${manifest_directory}/complete-dry-run.txt"

study_log="${work_directory}/study.log"
cat > "${study_log}" <<'EOF'
# topology_distributions=yule
# leaf_counts=128
# unique_pattern_counts=64
# topology_replicates=1
# deterministic_seed_base=20260813
backend,precision,benchmark_mode,study,dataset,topology,seed_base,seed,replicate,leaves,nodes,sites,unique_patterns,site_batch,measured_total_ms,max_abs_error,max_relative_error
cuda,FP32,full-input-update,standard,synthetic,yule,20260813,91,0,128,255,64,64,64,2,0,0
cuda,FP32,full-input-update,independent-taxa-pattern-grid,synthetic,yule,20260813,91,0,128,255,64,64,64,1,0,0
EOF
python3 - "${root}" "${study_log}" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "scripts/plot_synthetic_study.py"
spec = importlib.util.spec_from_file_location("plot_synthetic_study", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = module.rows([pathlib.Path(sys.argv[2])])
assert len(rows) == 1
assert rows[0]["study"] == "independent-taxa-pattern-grid"
module.study_design([pathlib.Path(sys.argv[2])])
PY
printf '%s\n' '# leaf_counts=256' >> "${study_log}"
if python3 - "${root}" "${study_log}" 2>/dev/null <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "scripts/plot_synthetic_study.py"
spec = importlib.util.spec_from_file_location("plot_synthetic_study", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.study_design([pathlib.Path(sys.argv[2])])
PY
then
  echo "conflicting synthetic-study declarations were accepted" >&2
  exit 1
fi
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

if [[ "$(uname -s)" == Linux ]]; then
  guard_output="$(benchmark_run_capacity_bounded "${work_directory}" 262144 \
    beagle-cpu FP32 8 bash -c 'ulimit -v')"
  [[ "${guard_output}" == 262144 ]]
fi

benchmark_capacity_exhausted=0
segfault_output="${work_directory}/segfault.txt"
if [[ "$(uname -s)" == Linux ]]; then
  benchmark_run_capacity_bounded "${work_directory}" 262144 \
    beagle-cpu FP32 16 bash -c 'kill -s SEGV $$' \
    > "${segfault_output}" 2>&1
  grep -Fq "first_infeasible_site_batch=16" "${segfault_output}"
  grep -Fq "reason=beagle-segfault-under-memory-limit" "${segfault_output}"
  [[ "${benchmark_capacity_exhausted}" == 1 ]]
else
  if benchmark_run_capacity_bounded "${work_directory}" 262144 \
    beagle-cpu FP32 16 bash -c 'kill -s SEGV $$' \
    > "${segfault_output}" 2>&1; then
    echo "an unbounded BEAGLE segfault was misclassified as capacity" >&2
    exit 1
  else
    status=$?
    [[ "${status}" == 139 ]]
  fi
fi

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
  echo 'parallel-phylogenetics-benchmark-schema=7'
  echo 'benchmark-profile=curated'
  echo 'accelerator-backend=cuda'
  echo 'hardware=test-hardware'
  echo 'resume-scope=hardware-class'
  echo 'session-identity=not-included'
  echo "host-cpu=$(grep -m1 -E 'model name|Hardware' /proc/cpuinfo 2>/dev/null || uname -m)"
  echo 'precisions=FP32'
  echo 'sections=fish pandit'
  echo 'modes=full-input-update'
  echo 'synthetic-modes=full-input-update factor-update fixed-model'
  echo 'fish-modes=full-input-update'
  echo 'pandit-modes=full-input-update'
  echo 'task-modes=full-input-update'
  echo 'jc69-modes=full-input-update'
  echo 'jc69-profile=paper'
  echo 'jc69-topologies-override=unset'
  echo 'jc69-leaves-override=unset'
  echo 'jc69-raw-sites-override=unset'
  echo 'jc69-heights-override=unset'
  echo 'jc69-replicates-override=unset'
  echo 'jc69-seed=20260814'
  echo 'jc69-minimum-repeats-override=unset'
  echo 'jc69-maximum-repeats-override=unset'
  echo 'jc69-work-budget-override=unset'
  echo 'jc69-minimum-site-batch=128'
  echo 'pandit-limit=25'
  echo 'repeats=15'
  echo 'empirical-repeats=3'
  echo 'empirical-minimum-branch-length=0.000001'
  echo 'conditioning-ms=0'
  echo 'beagle-version-label=4.1.0-pre-release-d1e9c62'
  echo 'beagle-source-revision=d1e9c62f922cf544fda4555aedf113519367c07a'
  echo 'beagle-source-url=https://github.com/beagle-dev/beagle-lib/archive/d1e9c62f922cf544fda4555aedf113519367c07a.tar.gz'
  echo 'beagle-source-sha256=55da832b6cde0e65872926b312fcc9f2b03c719b2ebdaabc309e2581c5725705'
  echo 'beagle-build-jobs=1'
  echo 'beagle-cmake-build-opencl=OFF'
  echo 'beagle-cmake-build-jni=OFF'
  echo 'beagle-cmake-build-openmp=OFF'
  echo 'beagle-cmake-build-bit=OFF'
  echo 'beagle-cpu-threads=1 1'
  echo 'fish-minimum-site-batch=256'
  echo 'host-memory-guard-percent=75'
  echo 'host-memory-guard-kib=automatic'
  echo 'pandit-minimum-leaves=100'
  echo 'distribution-topologies=yule beta-critical uniform caterpillar'
  echo 'distribution-leaves=128 512 2048 8192'
  echo 'distribution-patterns=16 64 256 1024'
  echo 'distribution-replicates=30'
  echo 'distribution-seed=20260813'
  echo 'distribution-timing-repeats=5'
  echo 'sanitizer-tools=memcheck racecheck synccheck'
  echo 'skip-sanitizer=0'
  sort "${source_root}/SOURCE_REVISIONS.txt"
} | shasum -a 256 | awk '{ print $1 }')"
printf '# cache_identity sha256=%s\nprevious-row\n' "${cache_identity}" > \
  "${source_root}/PREVIOUS_BENCHMARK_REPORT.txt"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_PRECISIONS_OVERRIDE=FP32 TREE_HMM_SKIP_FISH_TREE=1 \
TREE_HMM_BENCHMARK_SECTIONS="fish pandit" \
TREE_HMM_HARDWARE_IDENTITY_OVERRIDE=test-hardware \
TREE_HMM_RESUME_SCOPE=hardware-class \
TREE_HMM_BEAGLE_CPU_THREADS='1 1' \
BEAGLE_BUILD_JOBS=1 \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_root}" \
  > "${launcher_root}/unpacked.log"
grep -Fq 'using unpacked source input' "${launcher_root}/unpacked.log"
grep -Fq 'previous-row' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub precision=FP32 skip_fish=1' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub sections=fish pandit' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
different_protocol_working="${launcher_root}/different-protocol-working"
mkdir -p "${different_protocol_working}"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${different_protocol_working}" \
TREE_HMM_PRECISIONS_OVERRIDE=FP32 TREE_HMM_SKIP_FISH_TREE=1 \
TREE_HMM_BENCHMARK_SECTIONS="fish pandit" \
TREE_HMM_HARDWARE_IDENTITY_OVERRIDE=test-hardware \
TREE_HMM_RESUME_SCOPE=hardware-class \
TREE_HMM_BEAGLE_CPU_THREADS='1 1' \
TREE_HMM_DISTRIBUTION_TIMING_REPEATS=6 \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_root}" \
  > "${launcher_root}/different-protocol.log"
grep -Fq 'no prior report matches the current source identity' \
  "${launcher_root}/different-protocol.log"
! grep -Fq 'previous-row' \
  "${different_protocol_working}/parallel_phylogenetics_cuda_report.txt"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_ACCELERATOR_BACKEND_OVERRIDE=rocm \
TREE_HMM_RESUME_SCOPE=hardware-class \
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
TREE_HMM_RESUME_SCOPE=hardware-class \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_zip}" \
  > "${launcher_root}/zip.log"
grep -Fq 'source bundle SHA-256:' "${launcher_root}/zip.log"
grep -Fq 'different source identity' "${launcher_root}/zip.log"
! grep -Fq 'working-only-row' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub precision=FP64 skip_fish=0' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
