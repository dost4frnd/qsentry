#!/usr/bin/env bash
# Fast end-to-end verification on tiny data/models.
# Usage: bash scripts/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_all.sh configs/experiment/smoke.yaml
