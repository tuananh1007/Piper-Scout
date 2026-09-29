from setuptools import find_packages, setup

package_name = "plant_twin"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/plant_twin.launch.py"]),
        ("share/" + package_name + "/config", ["config/plant_twin.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vu Tuan Anh",
    maintainer_email="tuananh05253@gmail.com",
    description="Deformable leaf + stem digital twin fitted to RGB-D tracks during Piper grasps.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "twin_node = plant_twin.twin_node:main",
        ],
    },
)
