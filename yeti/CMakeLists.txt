project(yeti)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -Og")

add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=8)

find_package(catkin REQUIRED COMPONENTS
	steam
)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)
find_package(PkgConfig)

find_package(libpointmatcher REQUIRED)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
    ${libpointmatcher_INCLUDE_DIRS}
)

add_library(features
  src/features.cpp
)

target_link_libraries(features
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)

add_library(radar_utils
  src/radar_utils.cpp
)

target_link_libraries(radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)

add_library(association
  src/association.cpp
)

target_link_libraries(association
	${catkin_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)

add_executable(visualization src/visualization.cpp)
target_link_libraries(visualization
    features
    radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)

add_executable(mapping src/mapping.cpp)
target_link_libraries(mapping
	${DEPEND_LIBS}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)

add_executable(scan_matching_basic src/scan_matching_basic.cpp)
target_link_libraries(scan_matching_basic
    features
    radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    ${libpointmatcher_LIBRARIES}
)

add_executable(scan_matching_test src/test_scan_matching.cpp)
target_link_libraries(scan_matching_test
    features
    radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    ${libpointmatcher_LIBRARIES}
)

add_executable(build_map src/build_map.cpp)
target_link_libraries(build_map
    features
    radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    ${libpointmatcher_LIBRARIES}
)

add_executable(test_association src/test_association.cpp)
target_link_libraries(test_association
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    ${libpointmatcher_LIBRARIES}
)
