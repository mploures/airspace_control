#!/bin/bash
set -euo pipefail

NVANTS="${1:-5}"
SEP_PX="${2:-5}"
MAX_WH="${3:-4096}"

PKG="$(rospack find airspace_control)"
GRAFO="$PKG/graph/sistema_logistico/grafo.txt"
BITMAP="$PKG/graph/sistema_logistico/grafo_fundo_branco.png"
WORLD="$PKG/worlds/airspace.world"

echo "[INFO] Gerando mapa (nvants=$NVANTS, sep_px=$SEP_PX, max_wh=$MAX_WH)..."
python3 "$PKG/scripts/gen_stage_world.py" \
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

echo "[INFO] Iniciando Stage..."
exec rosrun stage_ros stageros "$WORLD"
