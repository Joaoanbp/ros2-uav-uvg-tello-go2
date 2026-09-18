#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${repo_dir}/docker/compose.yaml"
world="/workspace/simulation/worlds/uav_ugv.sdf"

if [[ "${1:-}" == "--headless" ]]; then
  sim_args=(-s -r -v 4 "${world}")
else
  sim_args=(-r -v 4 "${world}")
fi

if [[ -f /.dockerenv ]] && command -v gz >/dev/null 2>&1; then
  exec gz sim "${sim_args[@]}"
fi

exec docker compose -f "${compose_file}" exec robotics \
  gz sim "${sim_args[@]}"
