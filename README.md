# yeti

Dependencies:

Eigen 3
Boost 1.5.8
libnabo 1.0.7
STEAM

Installing STEAM:

Follow the instructions to download and build steam using catkin build

How to overlay catkin workspace:
- Create a new workspace where your project will exist, ex: ~/catkin_ws
- mkdir src
- Your project will be located under ~/catkin_ws/src/...
- catkin init
- catkin config --extend ~/steam_ws/devel/repo

Installing libnabo:

sudo add-apt-repository ppa:stephane.magnenat/xenial
sudo apt-get update
sudo apt-get install libnabo*
