#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${repo_dir}/docker/compose.yaml"

if [[ ! -f /.dockerenv ]]; then
  exec docker compose -f "${compose_file}" exec -T robotics \
    python3 /workspace/scripts/missao_de_teste.py
fi

exec python3 /workspace/scripts/missao_de_teste.py
