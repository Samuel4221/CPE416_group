FROM arm64v8/ubuntu:jammy

SHELL ["/bin/bash", "-c"]

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set up ROS 2 Humble from scratch
RUN apt update && apt install -y curl gnupg lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble (include rosdep and other essential tools)
RUN apt update && apt install -y \
    ros-humble-ros-base \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-colcon-common-extensions \
    git wget python3-pip ros-dev-tools vim

# Install Ignition Fortress (Gazebo)
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt update && apt -y install ignition-fortress
RUN apt update && apt install -y \
    ros-humble-turtlebot4-description \
    ros-humble-ros-gz 

# Initialize and update rosdep (this was missing)
RUN rosdep init || true
RUN rosdep update

# Source ROS in shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Make a directory for the ros workspace
# Install the necessary packages from source
RUN git clone --recursive https://github.com/ambulantelab/gobilda.git

# Compile the template code
RUN cd /gobilda/gobilda_ws/ && \
    source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -i -y --rosdistro humble && \
    colcon build --symlink-install

# Reset DEBIAN_FRONTEND to default
ENV DEBIAN_FRONTEND=

ENV DISPLAY=novnc:0.0