# Use Ubuntu 22.04 (Jammy) since ROS2 Humble targets it
FROM ubuntu:22.04

# Prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# --------------------
# 1. Basic dependencies
# --------------------
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    curl \
    git \
    lsb-release \
    gnupg2 \
    build-essential \
    python3-pip \
    locales \
    software-properties-common \
    iputils-ping \
    tmux \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libglu1-mesa \
    freeglut3 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set UTF-8 locale
RUN locale-gen en_US en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# --------------------
# 2. Install ROS2 Humble (desktop + tools)
# --------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | apt-key add - \
    && echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2-latest.list


RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-moveit \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    ros-humble-ur-moveit-config \
    ros-humble-geometric-shapes \
    ros-humble-srdfdom \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers

    

# # --------------------
# # 3. MoveIt Python API (rosdep sources)
# # --------------------
# RUN mkdir -p /etc/ros/rosdep/sources.list.d/ \
#     && echo "deb [trusted=yes] https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/ ./" \
#         | tee /etc/apt/sources.list.d/moveit_moveit2_packages.list \
#     && echo "yaml https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/local.yaml humble" \
#         | tee /etc/ros/rosdep/sources.list.d/1-moveit_moveit2_packages.list

# RUN apt-get update && apt-get upgrade -y \
#     && apt-get install -y ros-humble-moveit-py \
#     && rm -rf /var/lib/apt/lists/*

# --------------------
# 4. Python dependencies
# --------------------
RUN pip3 install --no-cache-dir dotmap

# --------------------
# 5. Workspace setup
# --------------------
WORKDIR /root/ws
RUN mkdir -p src
RUN mkdir -p src/robot_control_cloth/
# # --------------------
# # 7. Environment variables
# # --------------------
# ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
# # --------------------
# # 8. Default entrypoint
# # --------------------
# ENTRYPOINT ["/bin/bash"]
