from glob import glob

from setuptools import find_packages, setup

package_name = "scout_piper_language"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["web/*"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Vu Tuan Anh",
    maintainer_email="tuananh05253@gmail.com",
    description="Phase 5: natural-language instructions grounded in the segmentation, reachability filter "
                "with plain-language reasons, confirmation countdown, and the operator web GUI.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "instruction_node = scout_piper_language.instruction_node:main",
            "operator_gui_node = scout_piper_language.operator_gui_node:main",
            "operator_gui_demo = scout_piper_language.gui_server:main",
        ],
    },
)
