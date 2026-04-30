#!/bin/bash

#SBATCH --job-name=crack_eval
#SBATCH --output=/data/group_data/r3lit_cot/monitorability/logs/%x.out
#SBATCH --error=/data/group_data/r3lit_cot/monitorability/logs/%x.err

#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=1-00:00:00

#SBATCH --mail-user=jivitesj@andrew.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ${HOME}/batch_jobs.bashrc

# ============================================================
# TODO: Fill in these tokens
# ============================================================
export GITHUB_TOKEN="TODO"
export GEMINI_API_KEY="TODO"

# ============================================================
# Configuration
# ============================================================
CRACK_DIR=${HOME}/projects/CRACK
SCRATCH_DIR=/scratch/jivitesj/${SLURM_JOB_ID}
VLLM_DIR=${SCRATCH_DIR}/vllm
RESULTS_DIR=${CRACK_DIR}/eval/results
PR_NUMBERS="41043 41024 40950 40338 40412 40273 40133 39986 40982 40538 40531 40845"

export TMPDIR=/tmp

# ============================================================
# Activate CRACK environment
# ============================================================
cd ${CRACK_DIR}
source .venv/bin/activate

# ============================================================
# Clone vllm and fetch all PR refs
# ============================================================
mkdir -p "${SCRATCH_DIR}"

echo "Cloning vllm repository..."
git clone https://github.com/vllm-project/vllm.git "${VLLM_DIR}"

cd "${VLLM_DIR}"

echo "Fetching PR refs..."
for pr in ${PR_NUMBERS}; do
    echo "  Fetching PR #${pr}..."
    git fetch origin "pull/${pr}/head:refs/pull/${pr}/head" 2>/dev/null || true
    # Also fetch the merge commit to get all intermediate commits
    MERGE_SHA=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/vllm-project/vllm/pulls/${pr}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('merge_commit_sha',''))" 2>/dev/null)
    if [ -n "${MERGE_SHA}" ]; then
        git fetch origin "${MERGE_SHA}" 2>/dev/null || true
    fi
    # Fetch all individual commits referenced in review comments
    COMMENT_SHAS=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/vllm-project/vllm/pulls/${pr}/comments?per_page=100" \
        | python3 -c "import sys,json; shas=set(c['commit_id'] for c in json.load(sys.stdin)); print(' '.join(shas))" 2>/dev/null)
    for sha in ${COMMENT_SHAS}; do
        git fetch origin "${sha}" 2>/dev/null || true
    done
    # Fetch all commits listed in the PR
    COMMIT_SHAS=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/vllm-project/vllm/pulls/${pr}/commits?per_page=100" \
        | python3 -c "import sys,json; print(' '.join(c['sha'] for c in json.load(sys.stdin)))" 2>/dev/null)
    for sha in ${COMMIT_SHAS}; do
        git fetch origin "${sha}" 2>/dev/null || true
    done
done

# Verify key commits are available
echo ""
echo "Verifying commit availability..."
MISSING=0
for pr in ${PR_NUMBERS}; do
    COMMIT_SHAS=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/vllm-project/vllm/pulls/${pr}/commits?per_page=100" \
        | python3 -c "import sys,json; print(' '.join(c['sha'] for c in json.load(sys.stdin)))" 2>/dev/null)
    for sha in ${COMMIT_SHAS}; do
        if ! git cat-file -t "${sha}" &>/dev/null; then
            echo "  MISSING: PR #${pr} commit ${sha:0:8}"
            MISSING=$((MISSING + 1))
        fi
    done
done
if [ ${MISSING} -eq 0 ]; then
    echo "  All PR commits available."
else
    echo "  WARNING: ${MISSING} commits missing (likely force-pushed, will be skipped)."
fi

# ============================================================
# Run evaluation
# ============================================================
cd "${CRACK_DIR}"
mkdir -p "${RESULTS_DIR}"

echo ""
echo "============================================================"
echo "Starting CRACK evaluation"
echo "============================================================"

python eval/run_eval.py \
    --repo-path "${VLLM_DIR}" \
    --token "${GITHUB_TOKEN}" \
    --model google-gla:gemini-3-flash-preview \
    --skip-embeddings \
    --output-dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/eval.log"

echo ""
echo "Evaluation complete. Results in ${RESULTS_DIR}"

# ============================================================
# Cleanup scratch
# ============================================================
rm -rf "${SCRATCH_DIR}"
echo "Cleaned up ${SCRATCH_DIR}"
