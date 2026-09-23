#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="${ROOT_DIR}/configs/environment/versions.env"

if [[ ! -f "${VERSIONS_FILE}" ]]; then
  echo "Missing ${VERSIONS_FILE}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${VERSIONS_FILE}"

if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<'EOF'
conda was not found.
Install Miniconda or Miniforge first, reopen the shell, then rerun:
  bash scripts/env/setup_conda.sh
EOF
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "[conda] updating existing environment: ${CONDA_ENV_NAME}"
  conda env update     --name "${CONDA_ENV_NAME}"     --file "${ROOT_DIR}/environment.yml"     --prune
else
  echo "[conda] creating environment: ${CONDA_ENV_NAME}"
  conda env create --file "${ROOT_DIR}/environment.yml"
fi

conda activate "${CONDA_ENV_NAME}"

echo "[python] $(python --version)"
python -m pip install --upgrade "pip<26" "wheel" "setuptools<81"

TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "[torch] installing torch=${TORCH_VERSION}, torchvision=${TORCHVISION_VERSION}, ${TORCH_CUDA}"
python -m pip install   "torch==${TORCH_VERSION}"   "torchvision==${TORCHVISION_VERSION}"   --index-url "${TORCH_INDEX_URL}"

echo "[python] installing pilot runtime dependencies"
python -m pip install -r "${ROOT_DIR}/requirements/experiment.txt"

# OpenOOD imports imgaug eagerly even though MCM/NegLabel do not use DRAEM.
# Install it without dependencies so the pinned NumPy/OpenCV stack is preserved.
python -m pip install --no-deps "imgaug==${IMGAUG_VERSION}"

# libmr is also imported eagerly by OpenOOD's postprocessor package.
# It is a small C++/Cython extension; build against the already-pinned NumPy.
python -m pip install "Cython==${CYTHON_VERSION}"
python -m pip install --no-build-isolation "libmr==${LIBMR_VERSION}"

echo "[clip] installing pinned OpenAI CLIP"
python -m pip install --no-deps   "git+${OPENAI_CLIP_REPO}@${OPENAI_CLIP_COMMIT}"

THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
OPENOOD_DIR="${THIRD_PARTY_DIR}/OpenOOD-VLM"
mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${OPENOOD_DIR}/.git" ]]; then
  echo "[openood] cloning ${OPENOOD_REPO}"
  git clone "${OPENOOD_REPO}" "${OPENOOD_DIR}"
fi

echo "[openood] pinning commit ${OPENOOD_COMMIT}"
git -C "${OPENOOD_DIR}" fetch --all --tags --prune
git -C "${OPENOOD_DIR}" checkout --detach "${OPENOOD_COMMIT}"
ACTUAL_OPENOOD_COMMIT="$(git -C "${OPENOOD_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_OPENOOD_COMMIT}" != "${OPENOOD_COMMIT}" ]]; then
  echo "OpenOOD commit mismatch: ${ACTUAL_OPENOOD_COMMIT}" >&2
  exit 3
fi

# --no-deps is intentional. OpenOOD's setup.py still lists old/general
# dependencies (including faiss-gpu); the pilot runtime is pinned above.
python -m pip install --no-deps -e "${OPENOOD_DIR}"

# Immutable reference assets from the original baseline repositories.
MCM_DIR="${THIRD_PARTY_DIR}/MCM"
if [[ ! -d "${MCM_DIR}/.git" ]]; then
  echo "[mcm] cloning ${MCM_REPO}"
  git clone "${MCM_REPO}" "${MCM_DIR}"
fi
git -C "${MCM_DIR}" fetch --all --tags --prune
git -C "${MCM_DIR}" checkout --detach "${MCM_COMMIT}"
if [[ "$(git -C "${MCM_DIR}" rev-parse HEAD)" != "${MCM_COMMIT}" ]]; then
  echo "MCM commit mismatch" >&2
  exit 3
fi

NEGLABEL_DIR="${THIRD_PARTY_DIR}/NegLabel"
if [[ ! -d "${NEGLABEL_DIR}/.git" ]]; then
  echo "[neglabel] cloning ${NEGLABEL_REPO}"
  git clone "${NEGLABEL_REPO}" "${NEGLABEL_DIR}"
fi
git -C "${NEGLABEL_DIR}" fetch --all --tags --prune
git -C "${NEGLABEL_DIR}" checkout --detach "${NEGLABEL_COMMIT}"
if [[ "$(git -C "${NEGLABEL_DIR}" rev-parse HEAD)" != "${NEGLABEL_COMMIT}" ]]; then
  echo "NegLabel commit mismatch" >&2
  exit 3
fi

echo
echo "[verify] running environment smoke test"
python "${ROOT_DIR}/scripts/env/verify_environment.py"

cat <<EOF

Environment setup completed.

Activate it later with:
  conda activate ${CONDA_ENV_NAME}

Then install datasets with:
  python scripts/data/setup_datasets.py --datasets ood

After the first fully verified environment, freeze the critical package fingerprint:
  python scripts/env/environment_lock.py freeze
EOF
