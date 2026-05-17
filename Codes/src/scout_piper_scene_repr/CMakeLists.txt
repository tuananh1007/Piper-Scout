cmake_minimum_required(VERSION 3.16)
project(scout_piper_scene_repr)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(pluginlib REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# ---------------------------------------------------------------------------
# Semantic collision plugin (MoveIt 2 pluginlib)
# ---------------------------------------------------------------------------
add_library(semantic_collision_plugin SHARED
  src/semantic_collision_plugin.cpp
)
target_include_directories(semantic_collision_plugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(semantic_collision_plugin
  rclcpp
  pluginlib
  moveit_core
  moveit_ros_planning
  geometry_msgs
)

install(TARGETS semantic_collision_plugin
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

pluginlib_export_plugin_description_file(moveit_core plugin_description.xml)

# ---------------------------------------------------------------------------
# Python helpers (class_demux_node lives here for fast iteration)
# ---------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME}_py
  PACKAGE_DIR python/${PROJECT_NAME}_py
)

install(PROGRAMS
  python/class_demux_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
install(DIRECTORY config launch docs
  DESTINATION share/${PROJECT_NAME}
)

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(semantic_collision_plugin)
ament_export_dependencies(rclcpp pluginlib moveit_core moveit_ros_planning)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
