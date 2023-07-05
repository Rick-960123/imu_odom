cmake_minimum_required(VERSION 3.0.2)
project(imu_odom)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-Wall -g")
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  roslib
  sensor_msgs
)

## catkin specific configuration ##
catkin_package(
 INCLUDE_DIRS
 include
#  LIBRARIES 
#  imu_to_odom
 CATKIN_DEPENDS 
 geometry_msgs 
 nav_msgs 
 roscpp 
 roslib 
 sensor_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/imu_to_odom.cpp
# )

## Specify libraries to link a library or executable target against
add_executable(imu_odom src/imu_odom.cpp)
target_link_libraries(imu_odom
  ${catkin_LIBRARIES}
)
install(TARGETS imu_odom
              ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
              LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
              RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(PROGRAMS scripts/imu_butterworth_filter.py
                  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY scripts
                  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
                  config
                  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})