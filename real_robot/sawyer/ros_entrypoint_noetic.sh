
#!/bin/bash
set -e
# Source ROS
source /opt/ros/noetic/setup.bash
# Source workspace if present
if [ -f "/home/developer/catkin_ws/devel/setup.bash" ]; then
  source /home/developer/catkin_ws/devel/setup.bash
fi
exec "$@"