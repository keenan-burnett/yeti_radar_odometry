# yeti

This repo contains radar odometry code which removes motion distortion and Doppler effects from Navtech radar data. For our odometry experiments, we use the [Oxford Radar Robotcar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/). For our metric localization experiments, we use data taken using our own platform, Boreas, shown below.

## Boreas
![Boreas](figs/boreas.JPG "Boreas")

## Odometry Performance
![Odom](figs/trajectory.png "Odom")

## Build Instructions

Dependencies:

```
ROS Kinetic
Eigen 3
Boost 1.5.8
OpenCV 3
libnabo
libpointmatcher
yaml-cpp
steam
```

Note: we provide a Dockerfile which can be used to build a docker image with all the required deps already installed.

These build instructions assume you are building your project using a catkin workspace with catkin build.

Note: Eigen and OpenCV can be installed with ROS through ros-kinetic-eigen and ros-kinetic-opencv
Just remember to add "source /opt/ros/kinetic/setup.bash" to your .bashrc

Install Eigen, OpenCV and catkin tools:
```
sudo apt-get install -y ros-kinetic-eigen* ros-kinetic-opencv* python-catkin-tools
```

Installing libnabo:

```
git clone https://github.com/ethz-asl/libnabo.git
cd libnabo && mkdir build && cd build && cmake .. && make && make install
```

Installing libpointmatcher:

```
git clone https://github.com/ethz-asl/libpointmatcher.git
cd libpointmatcher && mkdir build && cd build && cmake .. && make && make install
```

Installing yaml-cpp:

```
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp && mkdir build && cd build && cmake .. && make && make install
```

Installing steam:

```
mkdir \~/steam_ws && cd \~steam_ws
git clone https://github.com/utiasASRL/steam.git
cd steam && git submodule update --init --remote
cd deps/catkin && catkin build && cd ../.. && catkin build
```

Building yeti:

```
mkdir -p \~/yeti_ws/src/ && cd \~/yeti_ws/src
git clone https://github.com/keenan-burnett/yeti.git
cd ../.. && catkin init && catkin config --extend \~/steam_ws/devel/repo
catkin build
```
