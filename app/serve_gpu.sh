#!/bin/bash
# SketchFade on a shared GPU server: pick the GPU by hand, run the app on it,
# and release the GPU when the app stops.
#
#   bash serve_gpu.sh [port] [gpu]      # without gpu: lists the GPUs and asks
#
# Normally started by connect.bat on the laptop (ssh -t ... with a tunnel).
# The app only listens on 127.0.0.1 of this server; nothing else is opened.
# The GPU is released when the app process ends:
#   - Ctrl+C in the laptop window           -> SIGINT
#   - closing the laptop window / ssh exits -> SIGHUP from the terminal
#   - connection lost without a clean close -> no ping from the laptop for
#     SKETCHFADE_IDLE_EXIT seconds (default 120) -> the app exits by itself

cd "$(dirname "$0")" || exit 1
PORT="${1:-9100}"
GPU="$2"

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found on $(hostname): no NVIDIA GPU or driver."
  exit 1
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "Port $PORT is already in use on $(hostname) (another SketchFade?)."
  echo "Pick another port in connect.cfg, or stop the other one first."
  exit 1
fi

# GPUs in nvidia-smi order; CUDA_DEVICE_ORDER below makes CUDA use the same order
COUNT=$(nvidia-smi -L | wc -l)
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
echo
echo "GPUs on $(hostname):"
printf "  %-3s %-34s %17s %6s  %s\n" "#" "name" "memory used/total" "util" "in use by"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,uuid \
           --format=csv,noheader,nounits |
while IFS=',' read -r idx name used total util uuid; do
  uuid="${uuid# }"
  users=$(echo "$APPS" | grep -F "$uuid" | cut -d',' -f2 |
          xargs -r -n1 ps -o user= -p 2>/dev/null | sort -u | paste -sd' ')
  printf "  %-3s %-34s %7s/%6s MiB %5s%%  %s\n" \
         "${idx# }" "${name# }" "${used# }" "${total# }" "${util# }" "${users:--}"
done
echo

while :; do
  if [ -z "$GPU" ]; then
    read -r -p "Pick a GPU (0-$((COUNT - 1)), q = quit): " GPU || exit 1
  fi
  [ "$GPU" = q ] && exit 0
  if [[ "$GPU" =~ ^[0-9]+$ ]] && [ "$GPU" -lt "$COUNT" ]; then
    break
  fi
  echo "  '$GPU' is not a GPU on this server."
  GPU=""
done

FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if [ "$FREE" -lt 4096 ]; then
  read -r -p "GPU $GPU has only $FREE MiB free (SketchFade needs ~4 GB). Use it anyway? [y/N] " a
  [ "$a" = y ] || [ "$a" = Y ] || exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID   # GPU numbers as nvidia-smi shows them
export CUDA_VISIBLE_DEVICES="$GPU"    # the app sees only this GPU
export SKETCHFADE_IDLE_EXIT="${SKETCHFADE_IDLE_EXIT:-120}"

echo
echo "Starting SketchFade on GPU $GPU, port $PORT (only 127.0.0.1 of this server)."
echo "Ctrl+C or closing the window stops it and releases the GPU."
echo
# exec: the app replaces this shell, so Ctrl+C and hang-up reach it directly
exec "$PY" server.py "$PORT"
