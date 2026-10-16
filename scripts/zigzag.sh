#!/usr/bin/env bash
# Shared one-terminal launcher for ROI missions.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Uso: ./scripts/zigzag.sh --shape {circle|star|random|drawn} [opções da missão]

Inicia o contêiner, abre o Gazebo e executa a missão com mapa e duas câmeras.
Exemplo: ./scripts/zigzag.sh --shape circle --altitude 1.6 --radius 5
Padrão: YOLO + geometria de pilar conhecido, câmeras somente RGB.
Cobertura adaptativa: acompanha solo observado e permite continuar em outra faixa.
--coverage-mode fixed mantém a repetição do zigzag nominal.
Para desenhar o perímetro no navegador, use ./scripts/zigzag-desenhado.sh.
HELP
  exit 0
fi

if [[ "${1:-}" == "--shape" ]]; then
  if (($# < 2)); then echo 'Falta valor para --shape' >&2; exit 2; fi
  shape="$2"
  shift 2
else
  # Accept the previous positional form used by existing launchers.
  shape="${1:-}"
  if (($#)); then shift; fi
fi
case "${shape}" in
  circle|star|random|drawn) ;;
  *) echo "Uso: $0 --shape {circle|star|random|drawn} [opções]" >&2; exit 2 ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<HELP
Uso: ./scripts/zigzag.sh --shape ${shape} [opções da missão]

Inicia o contêiner, abre o Gazebo e executa a missão ${shape} com mapa e duas câmeras ao vivo.
Reutiliza o ambiente local, sem build, download ou recriação do contêiner.
Exemplo: ./scripts/zigzag.sh --shape ${shape} --altitude 1.6 --radius 5 --side-overlap 70
Fonte padrão YOLO + pilar conhecido, sem sensor de profundidade.
Cobertura adaptativa com mapa de solo observado; --coverage-mode fixed usa faixas rígidas.
--depth-source rgb seleciona o experimento denso; gazebo seleciona a linha de base.
Mapa incremental e desvio A* ativos; entradas inválidas interrompem a missão.
Ctrl+C encerra a sessão. Fechar o gráfico aguarda o fim da missão e fecha o Gazebo.
Gráfico: artifacts/zigzag-${shape}-voo.png
Log do Gazebo: artifacts/zigzag-${shape}-gazebo.log
HELP
  exit 0
fi

if [[ "${1:-}" != "--inside" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo 'Erro: DISPLAY não está definido. Execute em um terminal da sessão gráfica.' >&2
    exit 1
  fi
  if ! command -v xauth >/dev/null 2>&1; then
    echo 'Erro: instale xauth no host para compartilhar a autorização da tela com o contêiner.' >&2
    exit 1
  fi
  # Copy only this display's cookie. FamilyWild permits the different container
  # hostname; the file remains private and is removed when this session ends.
  xauth_file="$(mktemp "${repo_dir}/docker/.xauth.XXXXXX")"
  trap 'rm -f -- "${xauth_file}"' EXIT
  xauth nlist "${DISPLAY}" | sed 's/^..../ffff/' | xauth -f "${xauth_file}" nmerge -
  if [[ ! -s "${xauth_file}" ]]; then
    echo 'Erro: nenhuma autorização X11 encontrada para o DISPLAY atual. Verifique XAUTHORITY na sessão gráfica.' >&2
    exit 1
  fi
  compose=(docker compose -f "${repo_dir}/docker/compose.yaml")
  # Missions are bind-mounted from the workspace. Running them does not need a
  # new image or registry access; preserve an existing simulation container.
  echo 'Iniciando o contêiner com o ambiente local (sem build ou download).'
  if ! "${compose[@]}" up -d --no-build --pull never --no-recreate robotics; then
    echo 'Não foi possível iniciar o ambiente local. Verifique o erro do Docker acima.' >&2
    echo 'Se a imagem ainda não foi preparada, execute o comando abaixo com acesso ao Docker Hub:' >&2
    printf '  docker compose -f %q up -d --build robotics\n' "${repo_dir}/docker/compose.yaml" >&2
    exit 1
  fi
  # A separate transport partition prevents interference with other worlds.
  "${compose[@]}" exec \
    -e "DISPLAY=${DISPLAY}" -e "XAUTHORITY=/workspace/docker/${xauth_file##*/}" \
    -e QT_QPA_PLATFORM=xcb -e MPLBACKEND=Qt5Agg \
    -e "GZ_PARTITION=zigzag-${shape}-${UID}-$$" \
    robotics bash /workspace/scripts/zigzag.sh --shape "${shape}" --inside "$@"
  exit 0
fi
shift

# The image also contains pip packages used by MAVProxy. Prefer the matched
# Ubuntu Matplotlib/NumPy/Qt stack for this graphical session only.
export PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export TELLO_GUI_SYSTEM_PYTHON=1
python3 "${repo_dir}/scripts/check_graphics.py"

mkdir -p "${repo_dir}/artifacts"
perception_mode=yolo
previous_arg=""
for mission_arg in "$@"; do
  if [[ "${previous_arg}" == "--depth-source" ]]; then perception_mode="${mission_arg}"; fi
  case "${mission_arg}" in --depth-source=*) perception_mode="${mission_arg#*=}" ;; esac
  previous_arg="${mission_arg}"
done
world_file="${repo_dir}/simulation/worlds/tello_only.sdf"
if [[ "${perception_mode}" == yolo ]]; then
  world_file="${repo_dir}/artifacts/tello-yolo-rgb.sdf"
  python3 "${repo_dir}/scripts/yolo_fixture.py" --shape "${shape}" "$@" --output "${world_file}"
fi
gazebo_log="${repo_dir}/artifacts/zigzag-${shape}-gazebo.log"
gazebo_pid=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${gazebo_pid}" ]]; then
    # Only stop the process group created by this launcher.
    kill -TERM -- "-${gazebo_pid}" 2>/dev/null || true
    timeout 5s tail --pid="${gazebo_pid}" -f /dev/null 2>/dev/null || true
    kill -KILL -- "-${gazebo_pid}" 2>/dev/null || true
    wait "${gazebo_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Abrindo Gazebo. Log: ${gazebo_log}"
setsid gz sim -r "${world_file}" >"${gazebo_log}" 2>&1 &
gazebo_pid=$!

# Wait for a responding world, rather than assuming a fixed startup delay.
if ! python3 - "${gazebo_pid}" <<'PY'
import os
import sys
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node

node = Node()
deadline = time.monotonic() + 60
print('Aguardando o mundo tello_only ficar pronto...', flush=True)
while time.monotonic() < deadline:
    try:
        os.kill(int(sys.argv[1]), 0)
    except ProcessLookupError:
        sys.exit('O Gazebo encerrou antes de carregar o mundo.')
    executed, response = node.request(
        '/world/tello_only/control', WorldControl(pause=False),
        WorldControl, Boolean, 500,
    )
    if executed and response.data:
        print('Gazebo pronto. Iniciando a missão e o gráfico.', flush=True)
        break
    time.sleep(0.5)
else:
    sys.exit('O mundo não respondeu em 60 segundos.')
PY
then
  tail -n 30 "${gazebo_log}" >&2
  exit 1
fi

python3 "${repo_dir}/scripts/drone_trajetoria.py" \
  --shape "${shape}" --run-search \
  --output "${repo_dir}/artifacts/zigzag-${shape}-voo.png" "$@"
