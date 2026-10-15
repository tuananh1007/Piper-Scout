from glob import glob

from setuptools import find_packages, setup

package_name = "piper_quickstart"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/urdf", glob("urdf/*.xacro")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sciarm",
    maintainer_email="sciarm@todo.todo",
    description="Piper + wrist RealSense quickstart: MoveIt control, preflight checks, episode recording.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "preflight = piper_quickstart.preflight:main",
            "moveit_bridge = piper_quickstart.moveit_bridge:main",
            "health_monitor = piper_quickstart.health_monitor:main",
            "camera_mount_publisher = piper_quickstart.camera_mount_publisher:main",
            "import_handeye = piper_quickstart.import_handeye:main",
            "record_episode = piper_quickstart.record_episode:main",
            "export_episode = piper_quickstart.export_episode:main",
            "scene_objects = piper_quickstart.scene_objects:main",
            "arm_safety = piper_quickstart.arm_safety:main",
            "safety_panel = piper_quickstart.safety_panel:main",
        ],
    },
)
