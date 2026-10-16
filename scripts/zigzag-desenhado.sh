#!/usr/bin/env bash
# Escolha o perímetro no navegador e, ao confirmar, execute a missão no Gazebo.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
mission_file="${repo_dir}/artifacts/zigzag-desenhado.json"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Uso: ./scripts/zigzag-desenhado.sh [--host ENDEREÇO] [--port PORTA] [--no-browser]

Abre o editor do perímetro no navegador. Ao clicar em "Iniciar no Gazebo",
salva o polígono e executa a missão no mesmo terminal.
Para usar um celular na mesma rede, passe --host 0.0.0.0.
HELP
  exit 0
fi
editor_args=()
while (($#)); do
  case "$1" in
    --host|--port)
      if (($# < 2)); then echo "Falta valor para $1" >&2; exit 2; fi
      editor_args+=("$1" "$2")
      shift 2 ;;
    --no-browser) editor_args+=("$1"); shift ;;
    *) echo "Opção desconhecida: $1" >&2; exit 2 ;;
  esac
done
python3 "${script_dir}/desenhar_perimetro.py" "${editor_args[@]}" --output "${mission_file}"
exec bash "${script_dir}/zigzag.sh" --shape drawn --polygon-json /workspace/artifacts/zigzag-desenhado.json
