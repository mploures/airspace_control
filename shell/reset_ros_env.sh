#!/bin/bash

echo "🔧 [ROS Reset] Terminating all ROS nodes (except /rosout)..."
NODES=$(rosnode list 2>/dev/null | grep -v "^/rosout$")

if [ -z "$NODES" ]; then
  echo "✅ No active ROS nodes found (except /rosout)."
else
  for NODE in $NODES; do
    echo "⛔ Terminating node: $NODE"
    rosnode kill "$NODE" 2>/dev/null
  done
fi

echo ""
echo "🔍 Checking and killing lingering ROS-related processes..."

PROCESS_LIST=("python3" "roslaunch" "roscore" "rosmaster" "rosout")
for PROC in "${PROCESS_LIST[@]}"; do
  PIDS=$(pgrep -f "$PROC")
  if [ -n "$PIDS" ]; then
    echo "⚠️  Killing process [$PROC]: $PIDS"
    kill -9 $PIDS 2>/dev/null
  else
    echo "✅ No [$PROC] process running."
  fi
done

echo ""
echo "🧹 ROS environment cleanup completed."

# Ask to restart roscore
echo ""
read -p "🔁 Do you want to restart roscore now? [y/N]: " ANSWER
if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
  echo "🚀 Starting roscore in background..."
  roscore > /dev/null 2>&1 &
  sleep 2
  echo "✅ roscore started."
else
  echo "⏹ roscore was not started."
fi
