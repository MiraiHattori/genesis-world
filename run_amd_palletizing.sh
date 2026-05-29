#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_SCRIPT=examples/rigid/single_sku_palletizing_amd.py exec "${SCRIPT_DIR}/run_amd_genesis_example.sh" "$@"
