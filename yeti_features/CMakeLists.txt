project(yeti_features)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -Og")

SET(GCC_COVERAGE_LINK_FLAGS    "-lboost_filesystem -lboost_system")

add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=8)

find_package(catkin REQUIRED COMPONENTS
	steam
)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)
find_package(PkgConfig)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)

add_library(yeti_features_lib
  src/features.cpp
  src/radar_utils.cpp
)

target_link_libraries(yeti_features_lib
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)

add_executable(visualization src/visualization.cpp)
target_link_libraries(visualization
    yeti_features_lib
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)
