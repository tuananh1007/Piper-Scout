from setuptools import find_packages, setup

package_name = "scout_piper_whole_body_mpc"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/whole_body_mpc.launch.py"]),
        ("share/" + package_name + "/config", ["config/whole_body_mpc.yaml", "config/whole_body_mpc_orin.yaml",
                                            "config/whole_body_mpc_gpu.yaml",
                                            "config/whole_body_mpc_orin_gpu.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vu Tuan Anh",
    maintainer_email="tuananh05253@gmail.com",
    description="Non-holonomic Scout + Piper whole-body MPC (geometry-only baseline).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "whole_body_mpc_node = scout_piper_whole_body_mpc.controller_node:main",
        "calibrate_slip = scout_piper_whole_body_mpc.calibration_nodes:slip_main",
        "calibrate_tcp = scout_piper_whole_body_mpc.calibration_nodes:tcp_main",
        "calibrate_hand_eye = scout_piper_whole_body_mpc.calibration_nodes:hand_eye_main",
        "calibrate_effort = scout_piper_whole_body_mpc.calibration_nodes:effort_main",
        "effort_force_node = scout_piper_whole_body_mpc.effort_force_node:main",
    ]},
)
