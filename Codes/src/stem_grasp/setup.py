from setuptools import find_packages, setup

package_name = "stem_grasp"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/stem_grasp.launch.py"]),
        ("share/" + package_name + "/config", ["config/pipeline.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sciarm",
    maintainer_email="sciarm@todo.todo",
    description="ROS 2 port of stem_grasp_ros1 pipeline.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pipeline_node = stem_grasp.pipeline_node:main",
            "segmentation_node = stem_grasp.segmentation_node:main",
            "pointcloud_node = stem_grasp.pointcloud_node:main",
            "hotkey_stop_and_zero = stem_grasp.hotkey_stop_and_zero:main",
        ],
    },
)
