#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CIBW_VERSION="${CIBW_VERSION:-}"

CONTAINER_ENGINE=${CIBW_CONTAINER_ENGINE:-}

if command -v uv >/dev/null 2>&1; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    uv venv "${VENV_DIR}"
  fi
else
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
fi

PYBIN="${VENV_DIR}/bin/python"

if [[ -z "${CONTAINER_ENGINE}" ]]; then
  if command -v docker >/dev/null 2>&1; then
    CONTAINER_ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then
    CONTAINER_ENGINE=podman
  fi
fi

if [[ -z "${CONTAINER_ENGINE}" ]]; then
  cat <<'EOF'
Error: No container engine found for cibuildwheel.

Install Docker or Podman, or set CIBW_CONTAINER_ENGINE=podman.
On Linux, Docker is the most common choice. After installing, ensure the daemon is running.
EOF
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${PYBIN}" --upgrade pip
  if [[ -n "${CIBW_VERSION}" ]]; then
    uv pip install --python "${PYBIN}" "cibuildwheel==${CIBW_VERSION}" maturin
  else
    uv pip install --python "${PYBIN}" cibuildwheel maturin
  fi
else
  "${PYBIN}" -m pip install --upgrade pip
  if [[ -n "${CIBW_VERSION}" ]]; then
    "${PYBIN}" -m pip install "cibuildwheel==${CIBW_VERSION}" maturin
  else
    "${PYBIN}" -m pip install cibuildwheel maturin
  fi
fi

cd "${ROOT_DIR}"
CIBW_CONTAINER_ENGINE="${CONTAINER_ENGINE}" "${VENV_DIR}/bin/cibuildwheel" --output-dir dist
