#!/bin/bash
set -e

TOPICS=(
    "/navsatfix"
)

OUTPUT_DIR="rosbag_recordings"
BAG_NAME="$OUTPUT_DIR/rosbag"

echo "Recording ROS2 bag to $BAG_NAME"
# ros2 bag record -o $BAG_NAME "${TOPICS[@]}"
ros2 bag record -o $BAG_NAME
echo "Recording complete."
