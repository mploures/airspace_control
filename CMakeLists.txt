cmake_minimum_required(VERSION 3.0.2)
project(airspace_control)

## ===========================================================
## ===============  CATKIN CONFIGURATION  =====================
## ===========================================================

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_generation
)

## ===========================================================
## ===============  PYTHON PACKAGE SETUP  =====================
## ===========================================================
# catkin_python_setup() deve vir ANTES de generate_messages()
catkin_python_setup()

## ===========================================================
## ===============  SERVICE AND MESSAGE GENERATION  ===========
## ===========================================================

add_service_files(
  FILES
    GetBattery.srv
    GotoXY.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
)

## ===========================================================
## ===============  INCLUDE DIRECTORIES  ======================
## ===========================================================

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## ===========================================================
## ===============  INSTALLATION TARGETS  =====================
## ===========================================================

# === Scripts Python (nós ROS) ===
file(GLOB PY_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py"
)

catkin_install_python(PROGRAMS
  ${PY_SCRIPTS}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# === Script shell (não usar catkin_install_python) ===
install(PROGRAMS
  scripts/create_map_and_launch_stage.sh    
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# === Módulos Python (ultrades_lib) ===
install(DIRECTORY ultrades_lib/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/ultrades_lib
  FILES_MATCHING PATTERN "*.py"
)

# === Recursos de grafo e scripts auxiliares ===
install(DIRECTORY graph/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/graph
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  FILES_MATCHING PATTERN "*.py"
)
