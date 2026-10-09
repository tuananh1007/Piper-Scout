cmake_minimum_required(VERSION 3.8)
project(scout_piper_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/piper_joint_state_relay.py
    scripts/piper_servo_bridge.py
    scripts/fake_piper_driver.py
    scripts/fake_scout_base.py
    scripts/servo_chain_check.py
    scripts/mpc_chain_check.py
    scripts/grasp_chain_check.py
    scripts/print_tcp.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(bringup_python test TIMEOUT 120)
endif()

ament_package()
