#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
CXX_COMPILER=${CXX:-/usr/bin/g++}
BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build-demo"}
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DAI_BUILD_TESTS=ON
cmake --build "${BUILD_DIR}" -j"$(nproc)"
(cd "${BUILD_DIR}" && ctest --output-on-failure)
(cd "${ROOT_DIR}" && "${BUILD_DIR}/ai_demo" "${ROOT_DIR}/config/sample_pipeline.json")
