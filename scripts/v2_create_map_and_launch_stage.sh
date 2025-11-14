#!/bin/bash
set -euo pipefail

NVANTS="${1:-5}"
SEP_PX="${2:-5}"
MAX_WH="${3:-400}"

PKG="$(rospack find airspace_control)"
GRAFO="$PKG/graph/sistema_logistico/grafo.txt"
BITMAP="$PKG/graph/sistema_logistico/grafo_fundo_branco.png"
WORLD="$PKG/worlds/airspace.world"
GEN="$PKG/scripts/gen_stage_world.py"   # ajuste o nome se seu gerador tiver outro arquivo

echo "[INFO] Gerando mapa (nvants=$NVANTS, sep_px=$SEP_PX, max_wh=$MAX_WH)..."
python3 "$GEN" \
  --grafo  "$GRAFO" \
  --bitmap "$BITMAP" \
  --nvants "$NVANTS" \
  --sep_px "$SEP_PX" \
  --max_wh "$MAX_WH"

echo "[INFO] Confirmando arquivo: $WORLD"
if [[ ! -f "$WORLD" ]]; then
  echo "[ERRO] $WORLD não foi criado." >&2
  exit 1
fi

# Deixa o caminho no parâmetro para a central ler
rosparam set /airspace/world_path "$WORLD"

echo "[INFO] Iniciando Stage..."
exec rosrun stage_ros stageros "$WORLD"
