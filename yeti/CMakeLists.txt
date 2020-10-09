project(yeti)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")

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

find_package(PythonLibs REQUIRED)

include_directories(
	include
    third_party/matplotlib-cpp
	third_party/nanoflann
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
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)

find_library(yaml-cpp REQUIRED)

add_executable(test_feature_matching src/test_feature_matching.cpp)
target_link_libraries(test_feature_matching
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    yaml-cpp
)

add_library(matplotlibcpp INTERFACE)
target_include_directories(matplotlibcpp INTERFACE include)
target_include_directories(matplotlibcpp INTERFACE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(matplotlibcpp INTERFACE ${PYTHON_LIBRARIES})

add_executable(test_motiondistortion src/test_motiondistortion.cpp)
target_link_libraries(test_motiondistortion
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	matplotlibcpp
)
target_compile_definitions(test_motiondistortion PRIVATE "-DWITHOUT_NUMPY")

add_executable(test_doppler_distortion src/test_doppler_distortion.cpp)
target_link_libraries(test_doppler_distortion
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	matplotlibcpp
)
target_compile_definitions(test_doppler_distortion PRIVATE "-DWITHOUT_NUMPY")

add_executable(showcase_distortion src/showcase_distortion.cpp)
target_link_libraries(showcase_distortion
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	matplotlibcpp
)
target_compile_definitions(showcase_distortion PRIVATE "-DWITHOUT_NUMPY")

add_executable(test_localization src/test_localization.cpp)
target_link_libraries(test_localization
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    yaml-cpp
)
