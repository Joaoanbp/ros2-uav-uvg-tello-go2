#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${repo_dir}/docker/compose.yaml"

check_cmd='models="$(gz model --list 2>/dev/null || true)"
printf "%s\n" "$models"
printf "%s\n" "$models" | grep -Eq "^[[:space:]]*-[[:space:]]+tello$" || { echo "ERROR: tello is not spawned" >&2; exit 1; }
printf "%s\n" "$models" | grep -Eq "^[[:space:]]*-[[:space:]]+go2_edu$" || { echo "ERROR: go2_edu is not spawned" >&2; exit 1; }
ss -lun | grep -qE ":9002[[:space:]]" || { echo "ERROR: ArduPilot UDP 9002 is not bound" >&2; exit 1; }
echo "OK: tello and go2_edu are in Gazebo; UDP 9002 is bound."'

if [[ -f /.dockerenv ]] && command -v gz >/dev/null 2>&1; then
  exec bash -lc "${check_cmd}"
fi

exec docker compose -f "${compose_file}" exec -T robotics bash -lc "${check_cmd}"
